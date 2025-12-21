#include <iostream>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "hnswlib.h"
#include <thread>
#include <atomic>
#include <stdlib.h>
#include <assert.h>

namespace py = pybind11;
using namespace pybind11::literals;  // needed to bring in _a literal

/*
 * replacement for the openmp '#pragma omp parallel for' directive
 * only handles a subset of functionality (no reductions etc)
 * Process ids from start (inclusive) to end (EXCLUSIVE)
 *
 * The method is borrowed from nmslib
 */
template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}


inline void assert_true(bool expr, const std::string & msg) {
    if (expr == false) throw std::runtime_error("Unpickle Error: " + msg);
    return;
}


class CustomFilterFunctor: public hnswlib::BaseFilterFunctor {
    std::function<bool(hnswlib::labeltype)> filter;

 public:
    explicit CustomFilterFunctor(const std::function<bool(hnswlib::labeltype)>& f) {
        filter = f;
    }

    bool operator()(hnswlib::labeltype id) {
        return filter(id);
    }
};


inline void get_input_array_shapes(const py::buffer_info& buffer, size_t* rows, size_t* features) {
    if (buffer.ndim != 2 && buffer.ndim != 1) {
        char msg[256];
        snprintf(msg, sizeof(msg),
            "Input vector data wrong shape. Number of dimensions %d. Data must be a 1D or 2D array.",
            buffer.ndim);
        throw std::runtime_error(msg);
    }
    if (buffer.ndim == 2) {
        *rows = buffer.shape[0];
        *features = buffer.shape[1];
    } else {
        *rows = 1;
        *features = buffer.shape[0];
    }
}


inline std::vector<size_t> get_input_ids_and_check_shapes(const py::object& ids_, size_t feature_rows) {
    std::vector<size_t> ids;
    if (!ids_.is_none()) {
        py::array_t < size_t, py::array::c_style | py::array::forcecast > items(ids_);
        auto ids_numpy = items.request();
        // check shapes
        if (!((ids_numpy.ndim == 1 && ids_numpy.shape[0] == feature_rows) ||
              (ids_numpy.ndim == 0 && feature_rows == 1))) {
            char msg[256];
            snprintf(msg, sizeof(msg),
                "The input label shape %d does not match the input data vector shape %d",
                ids_numpy.ndim, feature_rows);
            throw std::runtime_error(msg);
        }
        // extract data
        if (ids_numpy.ndim == 1) {
            std::vector<size_t> ids1(ids_numpy.shape[0]);
            for (size_t i = 0; i < ids1.size(); i++) {
                ids1[i] = items.data()[i];
            }
            ids.swap(ids1);
        } else if (ids_numpy.ndim == 0) {
            ids.push_back(*items.data());
        }
    }

    return ids;
}


template<typename dist_t, typename data_t = float>
class Index {
 public:
    static const int ser_version = 1;  // serialization version

    std::string space_name;
    int dim;
    size_t seed;
    size_t default_ef;

    bool index_inited;
    bool ep_added;
    bool normalize;
    int num_threads_default;
    hnswlib::labeltype cur_l;
    hnswlib::HierarchicalNSW<dist_t>* appr_alg;
    hnswlib::SpaceInterface<float>* l2space;


    Index(const std::string &space_name, const int dim) : space_name(space_name), dim(dim) {
        normalize = false;
        if (space_name == "l2") {
            l2space = new hnswlib::L2Space(dim);
        } else if (space_name == "ip") {
            l2space = new hnswlib::InnerProductSpace(dim);
        } else if (space_name == "cosine") {
            l2space = new hnswlib::InnerProductSpace(dim);
            normalize = true;
        } else {
            throw std::runtime_error("Space name must be one of l2, ip, or cosine.");
        }
        appr_alg = NULL;
        ep_added = true;
        index_inited = false;
        num_threads_default = std::thread::hardware_concurrency();

        default_ef = 10;
    }


    ~Index() {
        delete l2space;
        if (appr_alg)
            delete appr_alg;
    }


    void init_new_index(
        size_t maxElements,
        size_t M,
        size_t efConstruction,
        size_t random_seed,
        bool allow_replace_deleted) {
        if (appr_alg) {
            throw std::runtime_error("The index is already initiated.");
        }
        cur_l = 0;
        appr_alg = new hnswlib::HierarchicalNSW<dist_t>(l2space, maxElements, M, efConstruction, random_seed, allow_replace_deleted);
        index_inited = true;
        ep_added = false;
        appr_alg->ef_ = default_ef;
        seed = random_seed;
    }


    void set_ef(size_t ef) {
      default_ef = ef;
      if (appr_alg)
          appr_alg->ef_ = ef;
    }


    void set_num_threads(int num_threads) {
        this->num_threads_default = num_threads;
    }

    size_t indexFileSize() const {
        return appr_alg->indexFileSize();
    }

    void saveIndex(const std::string &path_to_index) {
        appr_alg->saveIndex(path_to_index);
    }


    void loadIndex(const std::string &path_to_index, size_t max_elements, bool allow_replace_deleted) {
      if (appr_alg) {
          std::cerr << "Warning: Calling load_index for an already inited index. Old index is being deallocated." << std::endl;
          delete appr_alg;
      }
      appr_alg = new hnswlib::HierarchicalNSW<dist_t>(l2space, path_to_index, false, max_elements, allow_replace_deleted);
      cur_l = appr_alg->cur_element_count;
      index_inited = true;
    }


    /**
     * Initialize a new index with mmap-backed storage.
     *
     * This creates a file-backed mmap instead of allocating memory with malloc,
     * allowing you to build indexes larger than available RAM.
     *
     * @param maxElements Maximum number of elements in the index
     * @param mmap_path Path to the mmap file (will be created/overwritten)
     * @param M HNSW M parameter
     * @param efConstruction Build quality parameter
     * @param random_seed Random seed for reproducibility
     * @param allow_replace_deleted Whether to allow replacing deleted elements
     */
    void init_new_index_mmap(
        size_t maxElements,
        const std::string &mmap_path,
        size_t M,
        size_t efConstruction,
        size_t random_seed,
        bool allow_replace_deleted) {
        if (appr_alg) {
            throw std::runtime_error("The index is already initiated.");
        }
        cur_l = 0;
        appr_alg = new hnswlib::HierarchicalNSW<dist_t>(l2space, maxElements, mmap_path, M, efConstruction, random_seed, allow_replace_deleted);
        index_inited = true;
        ep_added = false;
        appr_alg->ef_ = default_ef;
        seed = random_seed;
    }


    /**
     * Load an existing index using mmap for read-only access.
     *
     * This mmaps the index file directly instead of reading it into RAM,
     * useful for very large indexes where you want to rely on OS paging.
     *
     * @param path_to_index Path to the index file
     * @param read_only If true, map as read-only (default). If false, map read-write.
     */
    void loadIndexMmap(const std::string &path_to_index, bool read_only = true) {
      if (appr_alg) {
          std::cerr << "Warning: Calling load_index_mmap for an already inited index. Old index is being deallocated." << std::endl;
          delete appr_alg;
      }
      appr_alg = new hnswlib::HierarchicalNSW<dist_t>(l2space);
      appr_alg->loadIndexMmap(path_to_index, l2space, read_only);
      cur_l = appr_alg->cur_element_count;
      index_inited = true;
    }


    /**
     * Initialize a graph-only index with external vectors.
     *
     * This allocates only the graph structure (~68 bytes/element for M=8)
     * instead of graph + vectors (~1612 bytes/element for M=8, dim=384).
     * Vectors are fetched from the provided numpy array during build and search.
     *
     * @param maxElements Maximum number of elements
     * @param external_vectors Numpy array of shape (max_elements, dim) containing all vectors
     * @param M HNSW M parameter
     * @param efConstruction Build quality parameter
     * @param random_seed Random seed
     */
    void init_graph_only(
        size_t maxElements,
        py::array_t<float, py::array::c_style | py::array::forcecast> external_vectors,
        size_t M,
        size_t efConstruction,
        size_t random_seed) {
        if (appr_alg) {
            throw std::runtime_error("The index is already initiated.");
        }

        auto buffer = external_vectors.request();
        if (buffer.ndim != 2) {
            throw std::runtime_error("External vectors must be a 2D array");
        }
        if (buffer.shape[1] != dim) {
            throw std::runtime_error("External vectors dimension mismatch");
        }

        cur_l = 0;
        const float* vectors_ptr = static_cast<const float*>(buffer.ptr);
        appr_alg = new hnswlib::HierarchicalNSW<dist_t>(
            l2space, maxElements, vectors_ptr, M, efConstruction, random_seed, false);
        index_inited = true;
        ep_added = false;
        appr_alg->ef_ = default_ef;
        seed = random_seed;
    }


    /**
     * Set external vectors for an existing graph-only index.
     * Use this after loading a graph with load_graph().
     */
    void set_external_vectors(py::array_t<float, py::array::c_style | py::array::forcecast> external_vectors) {
        if (!appr_alg) {
            throw std::runtime_error("Index not initialized");
        }

        auto buffer = external_vectors.request();
        if (buffer.ndim != 2) {
            throw std::runtime_error("External vectors must be a 2D array");
        }
        if (buffer.shape[1] != dim) {
            throw std::runtime_error("External vectors dimension mismatch");
        }

        const float* vectors_ptr = static_cast<const float*>(buffer.ptr);
        appr_alg->setExternalVectorPointer(vectors_ptr, dim * sizeof(float));
    }


    /**
     * Set external vectors in fp16 format.
     * Vectors will be converted to fp32 on-the-fly during distance calculations.
     *
     * This is useful when you have fp16 vectors mmap'd from disk and want to avoid
     * doubling memory usage by converting the entire array to fp32.
     *
     * Note: The numpy array must have dtype=float16 and remain valid for the
     * lifetime of the index.
     */
    void set_external_vectors_fp16(py::array_t<uint16_t, py::array::c_style> external_vectors) {
        if (!appr_alg) {
            throw std::runtime_error("Index not initialized");
        }

        auto buffer = external_vectors.request();
        if (buffer.ndim != 2) {
            throw std::runtime_error("External vectors must be a 2D array");
        }
        if (buffer.shape[1] != dim) {
            throw std::runtime_error("External vectors dimension mismatch (got " +
                                     std::to_string(buffer.shape[1]) + ", expected " +
                                     std::to_string(dim) + ")");
        }

        const uint16_t* vectors_ptr = static_cast<const uint16_t*>(buffer.ptr);
        appr_alg->setExternalVectorPointerFp16(vectors_ptr, dim);
    }


    /**
     * Save only the graph structure (no vectors).
     * The resulting file is much smaller than save_index().
     */
    void saveGraph(const std::string &path_to_graph) {
        if (!appr_alg) {
            throw std::runtime_error("Index not initialized");
        }
        appr_alg->saveGraph(path_to_graph);
    }


    /**
     * Load a graph-only index. Call set_external_vectors() afterwards to provide vectors.
     */
    void loadGraph(const std::string &path_to_graph) {
        if (appr_alg) {
            std::cerr << "Warning: Calling load_graph for an already inited index. Old index is being deallocated." << std::endl;
            delete appr_alg;
        }
        appr_alg = new hnswlib::HierarchicalNSW<dist_t>(l2space);
        appr_alg->loadGraph(path_to_graph, l2space);
        cur_l = appr_alg->cur_element_count;
        index_inited = true;
    }


    void normalize_vector(float* data, float* norm_array) {
        float norm = 0.0f;
        for (int i = 0; i < dim; i++)
            norm += data[i] * data[i];
        norm = 1.0f / (sqrtf(norm) + 1e-30f);
        for (int i = 0; i < dim; i++)
            norm_array[i] = data[i] * norm;
    }


    void addItems(py::object input, py::object ids_ = py::none(), int num_threads = -1, bool replace_deleted = false) {
        py::array_t < dist_t, py::array::c_style | py::array::forcecast > items(input);
        auto buffer = items.request();
        if (num_threads <= 0)
            num_threads = num_threads_default;

        size_t rows, features;
        get_input_array_shapes(buffer, &rows, &features);

        if (features != dim)
            throw std::runtime_error("Wrong dimensionality of the vectors");

        // avoid using threads when the number of additions is small:
        if (rows <= num_threads * 4) {
            num_threads = 1;
        }

        std::vector<size_t> ids = get_input_ids_and_check_shapes(ids_, rows);

        {
            int start = 0;
            if (!ep_added) {
                size_t id = ids.size() ? ids.at(0) : (cur_l);
                float* vector_data = (float*)items.data(0);
                std::vector<float> norm_array(dim);
                if (normalize) {
                    normalize_vector(vector_data, norm_array.data());
                    vector_data = norm_array.data();
                }
                appr_alg->addPoint((void*)vector_data, (size_t)id, replace_deleted);
                start = 1;
                ep_added = true;
            }

            py::gil_scoped_release l;
            if (normalize == false) {
                ParallelFor(start, rows, num_threads, [&](size_t row, size_t threadId) {
                    size_t id = ids.size() ? ids.at(row) : (cur_l + row);
                    appr_alg->addPoint((void*)items.data(row), (size_t)id, replace_deleted);
                    });
            } else {
                std::vector<float> norm_array(num_threads * dim);
                ParallelFor(start, rows, num_threads, [&](size_t row, size_t threadId) {
                    // normalize vector:
                    size_t start_idx = threadId * dim;
                    normalize_vector((float*)items.data(row), (norm_array.data() + start_idx));

                    size_t id = ids.size() ? ids.at(row) : (cur_l + row);
                    appr_alg->addPoint((void*)(norm_array.data() + start_idx), (size_t)id, replace_deleted);
                    });
            }
            cur_l += rows;
        }
    }


    py::object getData(py::object ids_ = py::none(), std::string return_type = "numpy") {
        std::vector<std::string> return_types{"numpy", "list"};
        if (std::find(std::begin(return_types), std::end(return_types), return_type) == std::end(return_types)) {
            throw std::invalid_argument("return_type should be \"numpy\" or \"list\"");
        }
        std::vector<size_t> ids;
        if (!ids_.is_none()) {
            py::array_t < size_t, py::array::c_style | py::array::forcecast > items(ids_);
            auto ids_numpy = items.request();

            if (ids_numpy.ndim == 0) {
                throw std::invalid_argument("get_items accepts a list of indices and returns a list of vectors");
            } else {
                std::vector<size_t> ids1(ids_numpy.shape[0]);
                for (size_t i = 0; i < ids1.size(); i++) {
                    ids1[i] = items.data()[i];
                }
                ids.swap(ids1);
            }
        }

        std::vector<std::vector<data_t>> data;
        for (auto id : ids) {
            data.push_back(appr_alg->template getDataByLabel<data_t>(id));
        }
        if (return_type == "list") {
            return py::cast(data);
        }
        if (return_type == "numpy") {
            return py::array_t< data_t, py::array::c_style | py::array::forcecast >(py::cast(data));
        }
    }


    std::vector<hnswlib::labeltype> getIdsList() {
        std::vector<hnswlib::labeltype> ids;

        for (auto kv : appr_alg->label_lookup_) {
            ids.push_back(kv.first);
        }
        return ids;
    }


    py::dict getAnnData() const { /* WARNING: Index::getAnnData is not thread-safe with Index::addItems */
        std::unique_lock <std::mutex> templock(appr_alg->global);

        size_t level0_npy_size = appr_alg->cur_element_count * appr_alg->size_data_per_element_;
        size_t link_npy_size = 0;
        std::vector<size_t> link_npy_offsets(appr_alg->cur_element_count);

        for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
            size_t linkListSize = appr_alg->element_levels_[i] > 0 ? appr_alg->size_links_per_element_ * appr_alg->element_levels_[i] : 0;
            link_npy_offsets[i] = link_npy_size;
            if (linkListSize)
                link_npy_size += linkListSize;
        }

        char* data_level0_npy = (char*)malloc(level0_npy_size);
        char* link_list_npy = (char*)malloc(link_npy_size);
        int* element_levels_npy = (int*)malloc(appr_alg->element_levels_.size() * sizeof(int));

        hnswlib::labeltype* label_lookup_key_npy = (hnswlib::labeltype*)malloc(appr_alg->label_lookup_.size() * sizeof(hnswlib::labeltype));
        hnswlib::tableint* label_lookup_val_npy = (hnswlib::tableint*)malloc(appr_alg->label_lookup_.size() * sizeof(hnswlib::tableint));

        memset(label_lookup_key_npy, -1, appr_alg->label_lookup_.size() * sizeof(hnswlib::labeltype));
        memset(label_lookup_val_npy, -1, appr_alg->label_lookup_.size() * sizeof(hnswlib::tableint));

        size_t idx = 0;
        for (auto it = appr_alg->label_lookup_.begin(); it != appr_alg->label_lookup_.end(); ++it) {
            label_lookup_key_npy[idx] = it->first;
            label_lookup_val_npy[idx] = it->second;
            idx++;
        }

        memset(link_list_npy, 0, link_npy_size);

        memcpy(data_level0_npy, appr_alg->data_level0_memory_, level0_npy_size);
        memcpy(element_levels_npy, appr_alg->element_levels_.data(), appr_alg->element_levels_.size() * sizeof(int));

        for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
            size_t linkListSize = appr_alg->element_levels_[i] > 0 ? appr_alg->size_links_per_element_ * appr_alg->element_levels_[i] : 0;
            if (linkListSize) {
                memcpy(link_list_npy + link_npy_offsets[i], appr_alg->linkLists_[i], linkListSize);
            }
        }

        py::capsule free_when_done_l0(data_level0_npy, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_lvl(element_levels_npy, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_lb(label_lookup_key_npy, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_id(label_lookup_val_npy, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_ll(link_list_npy, [](void* f) {
            delete[] f;
            });

        /*  TODO: serialize state of random generators appr_alg->level_generator_ and appr_alg->update_probability_generator_  */
        /*        for full reproducibility / to avoid re-initializing generators inside Index::createFromParams         */

        return py::dict(
            "offset_level0"_a = appr_alg->offsetLevel0_,
            "max_elements"_a = appr_alg->max_elements_,
            "cur_element_count"_a = (size_t)appr_alg->cur_element_count,
            "size_data_per_element"_a = appr_alg->size_data_per_element_,
            "label_offset"_a = appr_alg->label_offset_,
            "offset_data"_a = appr_alg->offsetData_,
            "max_level"_a = appr_alg->maxlevel_,
            "enterpoint_node"_a = appr_alg->enterpoint_node_,
            "max_M"_a = appr_alg->maxM_,
            "max_M0"_a = appr_alg->maxM0_,
            "M"_a = appr_alg->M_,
            "mult"_a = appr_alg->mult_,
            "ef_construction"_a = appr_alg->ef_construction_,
            "ef"_a = appr_alg->ef_,
            "has_deletions"_a = (bool)appr_alg->num_deleted_,
            "size_links_per_element"_a = appr_alg->size_links_per_element_,
            "allow_replace_deleted"_a = appr_alg->allow_replace_deleted_,

            "label_lookup_external"_a = py::array_t<hnswlib::labeltype>(
                { appr_alg->label_lookup_.size() },  // shape
                { sizeof(hnswlib::labeltype) },  // C-style contiguous strides for each index
                label_lookup_key_npy,  // the data pointer
                free_when_done_lb),

            "label_lookup_internal"_a = py::array_t<hnswlib::tableint>(
                { appr_alg->label_lookup_.size() },  // shape
                { sizeof(hnswlib::tableint) },  // C-style contiguous strides for each index
                label_lookup_val_npy,  // the data pointer
                free_when_done_id),

            "element_levels"_a = py::array_t<int>(
                { appr_alg->element_levels_.size() },  // shape
                { sizeof(int) },  // C-style contiguous strides for each index
                element_levels_npy,  // the data pointer
                free_when_done_lvl),

            // linkLists_,element_levels_,data_level0_memory_
            "data_level0"_a = py::array_t<char>(
                { level0_npy_size },  // shape
                { sizeof(char) },  // C-style contiguous strides for each index
                data_level0_npy,  // the data pointer
                free_when_done_l0),

            "link_lists"_a = py::array_t<char>(
                { link_npy_size },  // shape
                { sizeof(char) },  // C-style contiguous strides for each index
                link_list_npy,  // the data pointer
                free_when_done_ll));
    }


    py::dict getIndexParams() const { /* WARNING: Index::getAnnData is not thread-safe with Index::addItems */
        auto params = py::dict(
            "ser_version"_a = py::int_(Index<float>::ser_version),  // serialization version
            "space"_a = space_name,
            "dim"_a = dim,
            "index_inited"_a = index_inited,
            "ep_added"_a = ep_added,
            "normalize"_a = normalize,
            "num_threads"_a = num_threads_default,
            "seed"_a = seed);

        if (index_inited == false)
            return py::dict(**params, "ef"_a = default_ef);

        auto ann_params = getAnnData();

        return py::dict(**params, **ann_params);
    }


    static Index<float>* createFromParams(const py::dict d) {
        // check serialization version
        assert_true(((int)py::int_(Index<float>::ser_version)) >= d["ser_version"].cast<int>(), "Invalid serialization version!");

        auto space_name_ = d["space"].cast<std::string>();
        auto dim_ = d["dim"].cast<int>();
        auto index_inited_ = d["index_inited"].cast<bool>();

        Index<float>* new_index = new Index<float>(space_name_, dim_);

        /*  TODO: deserialize state of random generators into new_index->level_generator_ and new_index->update_probability_generator_  */
        /*        for full reproducibility / state of generators is serialized inside Index::getIndexParams                      */
        new_index->seed = d["seed"].cast<size_t>();

        if (index_inited_) {
            new_index->appr_alg = new hnswlib::HierarchicalNSW<dist_t>(
                new_index->l2space,
                d["max_elements"].cast<size_t>(),
                d["M"].cast<size_t>(),
                d["ef_construction"].cast<size_t>(),
                new_index->seed);
            new_index->cur_l = d["cur_element_count"].cast<size_t>();
        }

        new_index->index_inited = index_inited_;
        new_index->ep_added = d["ep_added"].cast<bool>();
        new_index->num_threads_default = d["num_threads"].cast<int>();
        new_index->default_ef = d["ef"].cast<size_t>();

        if (index_inited_)
            new_index->setAnnData(d);

        return new_index;
    }


    static Index<float> * createFromIndex(const Index<float> & index) {
        return createFromParams(index.getIndexParams());
    }


    void setAnnData(const py::dict d) { /* WARNING: Index::setAnnData is not thread-safe with Index::addItems */
        std::unique_lock <std::mutex> templock(appr_alg->global);

        assert_true(appr_alg->offsetLevel0_ == d["offset_level0"].cast<size_t>(), "Invalid value of offsetLevel0_ ");
        assert_true(appr_alg->max_elements_ == d["max_elements"].cast<size_t>(), "Invalid value of max_elements_ ");

        appr_alg->cur_element_count = d["cur_element_count"].cast<size_t>();

        assert_true(appr_alg->size_data_per_element_ == d["size_data_per_element"].cast<size_t>(), "Invalid value of size_data_per_element_ ");
        assert_true(appr_alg->label_offset_ == d["label_offset"].cast<size_t>(), "Invalid value of label_offset_ ");
        assert_true(appr_alg->offsetData_ == d["offset_data"].cast<size_t>(), "Invalid value of offsetData_ ");

        appr_alg->maxlevel_ = d["max_level"].cast<int>();
        appr_alg->enterpoint_node_ = d["enterpoint_node"].cast<hnswlib::tableint>();

        assert_true(appr_alg->maxM_ == d["max_M"].cast<size_t>(), "Invalid value of maxM_ ");
        assert_true(appr_alg->maxM0_ == d["max_M0"].cast<size_t>(), "Invalid value of maxM0_ ");
        assert_true(appr_alg->M_ == d["M"].cast<size_t>(), "Invalid value of M_ ");
        assert_true(appr_alg->mult_ == d["mult"].cast<double>(), "Invalid value of mult_ ");
        assert_true(appr_alg->ef_construction_ == d["ef_construction"].cast<size_t>(), "Invalid value of ef_construction_ ");

        appr_alg->ef_ = d["ef"].cast<size_t>();

        assert_true(appr_alg->size_links_per_element_ == d["size_links_per_element"].cast<size_t>(), "Invalid value of size_links_per_element_ ");

        auto label_lookup_key_npy = d["label_lookup_external"].cast<py::array_t < hnswlib::labeltype, py::array::c_style | py::array::forcecast > >();
        auto label_lookup_val_npy = d["label_lookup_internal"].cast<py::array_t < hnswlib::tableint, py::array::c_style | py::array::forcecast > >();
        auto element_levels_npy = d["element_levels"].cast<py::array_t < int, py::array::c_style | py::array::forcecast > >();
        auto data_level0_npy = d["data_level0"].cast<py::array_t < char, py::array::c_style | py::array::forcecast > >();
        auto link_list_npy = d["link_lists"].cast<py::array_t < char, py::array::c_style | py::array::forcecast > >();

        for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
            if (label_lookup_val_npy.data()[i] < 0) {
                throw std::runtime_error("Internal id cannot be negative!");
            } else {
                appr_alg->label_lookup_.insert(std::make_pair(label_lookup_key_npy.data()[i], label_lookup_val_npy.data()[i]));
            }
        }

        memcpy(appr_alg->element_levels_.data(), element_levels_npy.data(), element_levels_npy.nbytes());

        size_t link_npy_size = 0;
        std::vector<size_t> link_npy_offsets(appr_alg->cur_element_count);

        for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
            size_t linkListSize = appr_alg->element_levels_[i] > 0 ? appr_alg->size_links_per_element_ * appr_alg->element_levels_[i] : 0;
            link_npy_offsets[i] = link_npy_size;
            if (linkListSize)
                link_npy_size += linkListSize;
        }

        memcpy(appr_alg->data_level0_memory_, data_level0_npy.data(), data_level0_npy.nbytes());

        for (size_t i = 0; i < appr_alg->max_elements_; i++) {
            size_t linkListSize = appr_alg->element_levels_[i] > 0 ? appr_alg->size_links_per_element_ * appr_alg->element_levels_[i] : 0;
            if (linkListSize == 0) {
                appr_alg->linkLists_[i] = nullptr;
            } else {
                appr_alg->linkLists_[i] = (char*)malloc(linkListSize);
                if (appr_alg->linkLists_[i] == nullptr)
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");

                memcpy(appr_alg->linkLists_[i], link_list_npy.data() + link_npy_offsets[i], linkListSize);
            }
        }

        // process deleted elements
        bool allow_replace_deleted = false;
        if (d.contains("allow_replace_deleted")) {
            allow_replace_deleted = d["allow_replace_deleted"].cast<bool>();
        }
        appr_alg->allow_replace_deleted_= allow_replace_deleted;

        appr_alg->num_deleted_ = 0;
        bool has_deletions = d["has_deletions"].cast<bool>();
        if (has_deletions) {
            for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
                if (appr_alg->isMarkedDeleted(i)) {
                    appr_alg->num_deleted_ += 1;
                    if (allow_replace_deleted) appr_alg->deleted_elements.insert(i);
                }
            }
        }
    }


    py::object knnQuery_return_numpy(
        py::object input,
        size_t k = 1,
        int num_threads = -1,
        const std::function<bool(hnswlib::labeltype)>& filter = nullptr) {
        py::array_t < dist_t, py::array::c_style | py::array::forcecast > items(input);
        auto buffer = items.request();
        hnswlib::labeltype* data_numpy_l;
        dist_t* data_numpy_d;
        size_t rows, features;

        if (num_threads <= 0)
            num_threads = num_threads_default;

        {
            py::gil_scoped_release l;
            get_input_array_shapes(buffer, &rows, &features);

            // avoid using threads when the number of searches is small:
            if (rows <= num_threads * 4) {
                num_threads = 1;
            }

            data_numpy_l = new hnswlib::labeltype[rows * k];
            data_numpy_d = new dist_t[rows * k];

            // Warning: search with a filter works slow in python in multithreaded mode. For best performance set num_threads=1
            CustomFilterFunctor idFilter(filter);
            CustomFilterFunctor* p_idFilter = filter ? &idFilter : nullptr;

            if (normalize == false) {
                ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                    std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = appr_alg->searchKnn(
                        (void*)items.data(row), k, p_idFilter);
                    if (result.size() != k)
                        throw std::runtime_error(
                            "Cannot return the results in a contiguous 2D array. Probably ef or M is too small");
                    for (int i = k - 1; i >= 0; i--) {
                        auto& result_tuple = result.top();
                        data_numpy_d[row * k + i] = result_tuple.first;
                        data_numpy_l[row * k + i] = result_tuple.second;
                        result.pop();
                    }
                });
            } else {
                std::vector<float> norm_array(num_threads * features);
                ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                    float* data = (float*)items.data(row);

                    size_t start_idx = threadId * dim;
                    normalize_vector((float*)items.data(row), (norm_array.data() + start_idx));

                    std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = appr_alg->searchKnn(
                        (void*)(norm_array.data() + start_idx), k, p_idFilter);
                    if (result.size() != k)
                        throw std::runtime_error(
                            "Cannot return the results in a contiguous 2D array. Probably ef or M is too small");
                    for (int i = k - 1; i >= 0; i--) {
                        auto& result_tuple = result.top();
                        data_numpy_d[row * k + i] = result_tuple.first;
                        data_numpy_l[row * k + i] = result_tuple.second;
                        result.pop();
                    }
                });
            }
        }
        py::capsule free_when_done_l(data_numpy_l, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_d(data_numpy_d, [](void* f) {
            delete[] f;
            });

        return py::make_tuple(
            py::array_t<hnswlib::labeltype>(
                { rows, k },  // shape
                { k * sizeof(hnswlib::labeltype),
                  sizeof(hnswlib::labeltype) },  // C-style contiguous strides for each index
                data_numpy_l,  // the data pointer
                free_when_done_l),
            py::array_t<dist_t>(
                { rows, k },  // shape
                { k * sizeof(dist_t), sizeof(dist_t) },  // C-style contiguous strides for each index
                data_numpy_d,  // the data pointer
                free_when_done_d));
    }


    void markDeleted(size_t label) {
        appr_alg->markDelete(label);
    }


    void unmarkDeleted(size_t label) {
        appr_alg->unmarkDelete(label);
    }


    void resizeIndex(size_t new_size) {
        appr_alg->resizeIndex(new_size);
    }


    size_t getMaxElements() const {
        return appr_alg->max_elements_;
    }


    size_t getCurrentCount() const {
        return appr_alg->cur_element_count;
    }
};

template<typename dist_t, typename data_t = float>
class BFIndex {
 public:
    static const int ser_version = 1;  // serialization version

    std::string space_name;
    int dim;
    bool index_inited;
    bool normalize;
    int num_threads_default;

    hnswlib::labeltype cur_l;
    hnswlib::BruteforceSearch<dist_t>* alg;
    hnswlib::SpaceInterface<float>* space;


    BFIndex(const std::string &space_name, const int dim) : space_name(space_name), dim(dim) {
        normalize = false;
        if (space_name == "l2") {
            space = new hnswlib::L2Space(dim);
        } else if (space_name == "ip") {
            space = new hnswlib::InnerProductSpace(dim);
        } else if (space_name == "cosine") {
            space = new hnswlib::InnerProductSpace(dim);
            normalize = true;
        } else {
            throw std::runtime_error("Space name must be one of l2, ip, or cosine.");
        }
        alg = NULL;
        index_inited = false;

        num_threads_default = std::thread::hardware_concurrency();
    }


    ~BFIndex() {
        delete space;
        if (alg)
            delete alg;
    }


    size_t getMaxElements() const {
        return alg->maxelements_;
    }


    size_t getCurrentCount() const {
        return alg->cur_element_count;
    }


    void set_num_threads(int num_threads) {
        this->num_threads_default = num_threads;
    }


    void init_new_index(const size_t maxElements) {
        if (alg) {
            throw std::runtime_error("The index is already initiated.");
        }
        cur_l = 0;
        alg = new hnswlib::BruteforceSearch<dist_t>(space, maxElements);
        index_inited = true;
    }


    void normalize_vector(float* data, float* norm_array) {
        float norm = 0.0f;
        for (int i = 0; i < dim; i++)
            norm += data[i] * data[i];
        norm = 1.0f / (sqrtf(norm) + 1e-30f);
        for (int i = 0; i < dim; i++)
            norm_array[i] = data[i] * norm;
    }


    void addItems(py::object input, py::object ids_ = py::none()) {
        py::array_t < dist_t, py::array::c_style | py::array::forcecast > items(input);
        auto buffer = items.request();
        size_t rows, features;
        get_input_array_shapes(buffer, &rows, &features);

        if (features != dim)
            throw std::runtime_error("Wrong dimensionality of the vectors");

        std::vector<size_t> ids = get_input_ids_and_check_shapes(ids_, rows);

        {
            for (size_t row = 0; row < rows; row++) {
                size_t id = ids.size() ? ids.at(row) : cur_l + row;
                if (!normalize) {
                    alg->addPoint((void *) items.data(row), (size_t) id);
                } else {
                    std::vector<float> normalized_vector(dim);
                    normalize_vector((float *)items.data(row), normalized_vector.data());
                    alg->addPoint((void *) normalized_vector.data(), (size_t) id);
                }
            }
            cur_l+=rows;
        }
    }


    void deleteVector(size_t label) {
        alg->removePoint(label);
    }


    void saveIndex(const std::string &path_to_index) {
        alg->saveIndex(path_to_index);
    }


    void loadIndex(const std::string &path_to_index, size_t max_elements) {
        if (alg) {
            std::cerr << "Warning: Calling load_index for an already inited index. Old index is being deallocated." << std::endl;
            delete alg;
        }
        alg = new hnswlib::BruteforceSearch<dist_t>(space, path_to_index);
        cur_l = alg->cur_element_count;
        index_inited = true;
    }


    py::object knnQuery_return_numpy(
        py::object input,
        size_t k = 1,
        int num_threads = -1,
        const std::function<bool(hnswlib::labeltype)>& filter = nullptr) {
        py::array_t < dist_t, py::array::c_style | py::array::forcecast > items(input);
        auto buffer = items.request();
        hnswlib::labeltype *data_numpy_l;
        dist_t *data_numpy_d;
        size_t rows, features;

        if (num_threads <= 0)
            num_threads = num_threads_default;

        {
            py::gil_scoped_release l;
            get_input_array_shapes(buffer, &rows, &features);

            data_numpy_l = new hnswlib::labeltype[rows * k];
            data_numpy_d = new dist_t[rows * k];

            CustomFilterFunctor idFilter(filter);
            CustomFilterFunctor* p_idFilter = filter ? &idFilter : nullptr;

            ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = alg->searchKnn(
                    (void*)items.data(row), k, p_idFilter);
                for (int i = k - 1; i >= 0; i--) {
                    auto& result_tuple = result.top();
                    data_numpy_d[row * k + i] = result_tuple.first;
                    data_numpy_l[row * k + i] = result_tuple.second;
                    result.pop();
                }
            });
        }

        py::capsule free_when_done_l(data_numpy_l, [](void *f) {
            delete[] f;
        });
        py::capsule free_when_done_d(data_numpy_d, [](void *f) {
            delete[] f;
        });


        return py::make_tuple(
                py::array_t<hnswlib::labeltype>(
                        { rows, k },  // shape
                        { k * sizeof(hnswlib::labeltype),
                          sizeof(hnswlib::labeltype)},  // C-style contiguous strides for each index
                        data_numpy_l,  // the data pointer
                        free_when_done_l),
                py::array_t<dist_t>(
                        { rows, k },  // shape
                        { k * sizeof(dist_t), sizeof(dist_t) },  // C-style contiguous strides for each index
                        data_numpy_d,  // the data pointer
                        free_when_done_d));
    }
};


/**
 * ShardedIndex - Python wrapper for building and searching sharded HNSW indexes.
 *
 * This enables building indexes larger than available RAM by splitting data into
 * multiple shards, each built independently.
 */
template<typename dist_t, typename data_t = float>
class ShardedIndexPy {
public:
    std::string space_name;
    int dim;
    bool normalize;
    int num_threads_default;
    size_t default_ef;

    hnswlib::SpaceInterface<float>* space;
    std::unique_ptr<hnswlib::ShardedIndex<dist_t>> sharded_index;
    size_t n_shards;
    size_t total_elements;

    ShardedIndexPy(const std::string& space_name, int dim)
        : space_name(space_name), dim(dim), n_shards(0), total_elements(0) {
        normalize = false;
        if (space_name == "l2") {
            space = new hnswlib::L2Space(dim);
        } else if (space_name == "ip") {
            space = new hnswlib::InnerProductSpace(dim);
        } else if (space_name == "cosine") {
            space = new hnswlib::InnerProductSpace(dim);
            normalize = true;
        } else {
            throw std::runtime_error("Space name must be one of l2, ip, or cosine.");
        }
        sharded_index = std::make_unique<hnswlib::ShardedIndex<dist_t>>(space, dim);
        num_threads_default = std::thread::hardware_concurrency();
        default_ef = 50;
    }

    ~ShardedIndexPy() {
        sharded_index.reset();
        delete space;
    }

    void set_ef(size_t ef) {
        default_ef = ef;
        if (n_shards > 0) {
            sharded_index->setEf(ef);
        }
    }

    void set_num_threads(int num_threads) {
        num_threads_default = num_threads;
    }

    void set_labels_are_global(bool value) {
        sharded_index->setLabelsAreGlobal(value);
    }

    void normalize_vector(float* data, float* norm_array) {
        float norm = 0.0f;
        for (int i = 0; i < dim; i++)
            norm += data[i] * data[i];
        norm = 1.0f / (sqrtf(norm) + 1e-30f);
        for (int i = 0; i < dim; i++)
            norm_array[i] = data[i] * norm;
    }

    /**
     * Build a single shard of the index.
     *
     * Call this once per shard, with only that shard's vectors loaded in RAM.
     */
    void buildShard(
        py::array_t<float, py::array::c_style | py::array::forcecast> vectors,
        const std::string& output_dir,
        size_t shard_idx,
        size_t start_idx,
        size_t M = 8,
        size_t ef_construction = 200,
        int num_threads = -1
    ) {
        if (num_threads <= 0) num_threads = num_threads_default;

        auto buffer = vectors.request();
        if (buffer.ndim != 2) {
            throw std::runtime_error("Vectors must be a 2D array");
        }
        if (buffer.shape[1] != dim) {
            throw std::runtime_error("Vector dimension mismatch");
        }

        size_t n_elements = buffer.shape[0];
        const float* vectors_ptr = static_cast<const float*>(buffer.ptr);

        // If normalizing, we need to make a copy
        std::vector<float> normalized;
        if (normalize) {
            normalized.resize(n_elements * dim);
            for (size_t i = 0; i < n_elements; i++) {
                normalize_vector(const_cast<float*>(vectors_ptr + i * dim),
                                normalized.data() + i * dim);
            }
            vectors_ptr = normalized.data();
        }

        py::gil_scoped_release release;
        sharded_index->buildShard(
            vectors_ptr, n_elements, output_dir, shard_idx, start_idx,
            M, ef_construction, num_threads
        );
    }

    /**
     * Save global index metadata after all shards are built.
     */
    void saveMetadata(
        const std::string& output_dir,
        size_t n_shards,
        size_t total_elements,
        size_t M,
        size_t ef_construction
    ) {
        sharded_index->saveMetadata(
            output_dir, n_shards, total_elements, dim, M, ef_construction, space_name
        );
        this->n_shards = n_shards;
        this->total_elements = total_elements;
    }

    /**
     * Load all shards for search with fp32 vectors.
     */
    void loadShards(
        const std::string& output_dir,
        py::array_t<float, py::array::c_style | py::array::forcecast> external_vectors,
        size_t n_shards
    ) {
        auto buffer = external_vectors.request();
        if (buffer.ndim != 2) {
            throw std::runtime_error("External vectors must be a 2D array");
        }
        if (buffer.shape[1] != dim) {
            throw std::runtime_error("Vector dimension mismatch");
        }

        const float* vectors_ptr = static_cast<const float*>(buffer.ptr);
        total_elements = buffer.shape[0];

        py::gil_scoped_release release;
        sharded_index->loadShards(output_dir, vectors_ptr, n_shards);
        sharded_index->setEf(default_ef);
        this->n_shards = n_shards;
    }

    /**
     * Load all shards for search with fp16 vectors.
     */
    void loadShardsFp16(
        const std::string& output_dir,
        py::array_t<uint16_t, py::array::c_style> external_vectors,
        size_t n_shards,
        bool use_mmap_graphs = false
    ) {
        auto buffer = external_vectors.request();
        if (buffer.ndim != 2) {
            throw std::runtime_error("External vectors must be a 2D array");
        }
        if (buffer.shape[1] != dim) {
            throw std::runtime_error("Vector dimension mismatch");
        }

        const uint16_t* vectors_ptr = static_cast<const uint16_t*>(buffer.ptr);
        total_elements = buffer.shape[0];

        py::gil_scoped_release release;
        sharded_index->loadShardsFp16(output_dir, vectors_ptr, n_shards, use_mmap_graphs);
        sharded_index->setEf(default_ef);
        this->n_shards = n_shards;
    }

    /**
     * Load all shards for search with PQ (Product Quantization) codes.
     *
     * PQ compresses vectors to n_subvectors bytes each, enabling search on
     * datasets too large to fit in memory. Distance computation uses asymmetric
     * lookup tables computed from the query.
     */
    void loadShardsPQ(
        const std::string& output_dir,
        py::array_t<uint8_t, py::array::c_style> pq_codes,
        py::array_t<float, py::array::c_style | py::array::forcecast> codebooks,
        size_t n_shards,
        size_t n_subvectors,
        size_t n_centroids,
        size_t subvector_dim,
        bool use_mmap_graphs = false,
        size_t n_ram_shards = 0,
        py::object sort_order_obj = py::none()
    ) {
        // Validate PQ codes array
        auto codes_buffer = pq_codes.request();
        if (codes_buffer.ndim != 2) {
            throw std::runtime_error("PQ codes must be a 2D array (n_vectors, n_subvectors)");
        }
        if ((size_t)codes_buffer.shape[1] != n_subvectors) {
            throw std::runtime_error("PQ codes dimension mismatch (got " +
                                     std::to_string(codes_buffer.shape[1]) + ", expected " +
                                     std::to_string(n_subvectors) + ")");
        }

        // Validate codebooks array
        auto cb_buffer = codebooks.request();
        if (cb_buffer.ndim != 3) {
            throw std::runtime_error("Codebooks must be a 3D array (n_subvectors, n_centroids, subvector_dim)");
        }
        if ((size_t)cb_buffer.shape[0] != n_subvectors ||
            (size_t)cb_buffer.shape[1] != n_centroids ||
            (size_t)cb_buffer.shape[2] != subvector_dim) {
            throw std::runtime_error("Codebooks shape mismatch");
        }

        // Verify dimension consistency
        if (n_subvectors * subvector_dim != (size_t)dim) {
            throw std::runtime_error("PQ dimension mismatch: n_subvectors * subvector_dim (" +
                                     std::to_string(n_subvectors * subvector_dim) +
                                     ") != dim (" + std::to_string(dim) + ")");
        }

        // Handle optional sort_order for cluster-based sharding
        const int64_t* sort_order_ptr = nullptr;
        if (!sort_order_obj.is_none()) {
            py::array_t<int64_t, py::array::c_style> sort_order = sort_order_obj.cast<py::array_t<int64_t, py::array::c_style>>();
            auto so_buffer = sort_order.request();
            if (so_buffer.ndim != 1) {
                throw std::runtime_error("sort_order must be a 1D array");
            }
            if ((size_t)so_buffer.shape[0] != (size_t)codes_buffer.shape[0]) {
                throw std::runtime_error("sort_order length must match number of vectors");
            }
            sort_order_ptr = static_cast<const int64_t*>(so_buffer.ptr);
        }

        const uint8_t* codes_ptr = static_cast<const uint8_t*>(codes_buffer.ptr);
        const float* codebooks_ptr = static_cast<const float*>(cb_buffer.ptr);
        total_elements = codes_buffer.shape[0];

        py::gil_scoped_release release;
        sharded_index->loadShardsPQ(
            output_dir, codes_ptr, codebooks_ptr, n_shards,
            n_subvectors, n_centroids, subvector_dim, use_mmap_graphs, n_ram_shards,
            sort_order_ptr
        );
        sharded_index->setEf(default_ef);
        this->n_shards = n_shards;
    }

    /**
     * Search across all shards.
     */
    py::object knnQuery(
        py::object input,
        size_t k = 1,
        int num_threads = -1
    ) {
        py::array_t<dist_t, py::array::c_style | py::array::forcecast> items(input);
        auto buffer = items.request();

        if (num_threads <= 0) num_threads = num_threads_default;

        size_t rows, features;
        get_input_array_shapes(buffer, &rows, &features);

        if (features != (size_t)dim) {
            throw std::runtime_error("Query dimension mismatch");
        }

        hnswlib::labeltype* data_numpy_l = new hnswlib::labeltype[rows * k];
        dist_t* data_numpy_d = new dist_t[rows * k];

        {
            py::gil_scoped_release release;

            if (normalize) {
                std::vector<float> norm_array(num_threads * dim);
                ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                    size_t start_idx = threadId * dim;
                    normalize_vector(const_cast<float*>(static_cast<const float*>(items.data(row))),
                                    norm_array.data() + start_idx);

                    auto result = sharded_index->searchKnn(
                        (void*)(norm_array.data() + start_idx), k);

                    size_t i = k - 1;
                    while (!result.empty() && i < k) {
                        auto& top = result.top();
                        data_numpy_d[row * k + i] = top.first;
                        data_numpy_l[row * k + i] = top.second;
                        result.pop();
                        i--;
                    }
                });
            } else {
                ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                    auto result = sharded_index->searchKnn(
                        (void*)items.data(row), k);

                    size_t i = k - 1;
                    while (!result.empty() && i < k) {
                        auto& top = result.top();
                        data_numpy_d[row * k + i] = top.first;
                        data_numpy_l[row * k + i] = top.second;
                        result.pop();
                        i--;
                    }
                });
            }
        }

        py::capsule free_when_done_l(data_numpy_l, [](void* f) { delete[] f; });
        py::capsule free_when_done_d(data_numpy_d, [](void* f) { delete[] f; });

        return py::make_tuple(
            py::array_t<hnswlib::labeltype>(
                { rows, k },
                { k * sizeof(hnswlib::labeltype), sizeof(hnswlib::labeltype) },
                data_numpy_l,
                free_when_done_l),
            py::array_t<dist_t>(
                { rows, k },
                { k * sizeof(dist_t), sizeof(dist_t) },
                data_numpy_d,
                free_when_done_d));
    }

    /**
     * Search only specific shards (for IVF-style routing).
     */
    py::object knnQuerySelective(
        py::object input,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> shard_ids,
        size_t k = 1,
        int num_threads = -1
    ) {
        py::array_t<dist_t, py::array::c_style | py::array::forcecast> items(input);
        auto buffer = items.request();

        if (num_threads <= 0) num_threads = num_threads_default;

        size_t rows, features;
        get_input_array_shapes(buffer, &rows, &features);

        if (features != (size_t)dim) {
            throw std::runtime_error("Query dimension mismatch");
        }

        // Extract shard ids
        auto shard_buffer = shard_ids.request();
        std::vector<size_t> shard_vec;
        if (shard_buffer.ndim == 1) {
            for (size_t i = 0; i < (size_t)shard_buffer.shape[0]; i++) {
                shard_vec.push_back(static_cast<size_t>(shard_ids.at(i)));
            }
        } else {
            throw std::runtime_error("shard_ids must be a 1D array");
        }

        hnswlib::labeltype* data_numpy_l = new hnswlib::labeltype[rows * k];
        dist_t* data_numpy_d = new dist_t[rows * k];

        // Initialize with zeros in case we get fewer results
        std::memset(data_numpy_l, 0, rows * k * sizeof(hnswlib::labeltype));
        std::memset(data_numpy_d, 0, rows * k * sizeof(dist_t));

        {
            py::gil_scoped_release release;

            if (normalize) {
                std::vector<float> norm_array(num_threads * dim);
                ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                    size_t start_idx = threadId * dim;
                    normalize_vector(const_cast<float*>(static_cast<const float*>(items.data(row))),
                                    norm_array.data() + start_idx);

                    auto result = sharded_index->searchKnnSelective(
                        (void*)(norm_array.data() + start_idx), shard_vec, k);

                    size_t i = k - 1;
                    while (!result.empty() && i < k) {
                        auto& top = result.top();
                        data_numpy_d[row * k + i] = top.first;
                        data_numpy_l[row * k + i] = top.second;
                        result.pop();
                        i--;
                    }
                });
            } else {
                ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                    auto result = sharded_index->searchKnnSelective(
                        (void*)items.data(row), shard_vec, k);

                    size_t i = k - 1;
                    while (!result.empty() && i < k) {
                        auto& top = result.top();
                        data_numpy_d[row * k + i] = top.first;
                        data_numpy_l[row * k + i] = top.second;
                        result.pop();
                        i--;
                    }
                });
            }
        }

        py::capsule free_when_done_l(data_numpy_l, [](void* f) { delete[] f; });
        py::capsule free_when_done_d(data_numpy_d, [](void* f) { delete[] f; });

        return py::make_tuple(
            py::array_t<hnswlib::labeltype>(
                { rows, k },
                { k * sizeof(hnswlib::labeltype), sizeof(hnswlib::labeltype) },
                data_numpy_l,
                free_when_done_l),
            py::array_t<dist_t>(
                { rows, k },
                { k * sizeof(dist_t), sizeof(dist_t) },
                data_numpy_d,
                free_when_done_d));
    }

    size_t getTotalElements() const { return total_elements; }
    size_t getNumShards() const { return n_shards; }
};


PYBIND11_PLUGIN(hnswlib) {
        py::module m("hnswlib");

        py::class_<Index<float>>(m, "Index")
        .def(py::init(&Index<float>::createFromParams), py::arg("params"))
           /* WARNING: Index::createFromIndex is not thread-safe with Index::addItems */
        .def(py::init(&Index<float>::createFromIndex), py::arg("index"))
        .def(py::init<const std::string &, const int>(), py::arg("space"), py::arg("dim"))
        .def("init_index",
            &Index<float>::init_new_index,
            py::arg("max_elements"),
            py::arg("M") = 16,
            py::arg("ef_construction") = 200,
            py::arg("random_seed") = 100,
            py::arg("allow_replace_deleted") = false)
        .def("knn_query",
            &Index<float>::knnQuery_return_numpy,
            py::arg("data"),
            py::arg("k") = 1,
            py::arg("num_threads") = -1,
            py::arg("filter") = py::none())
        .def("add_items",
            &Index<float>::addItems,
            py::arg("data"),
            py::arg("ids") = py::none(),
            py::arg("num_threads") = -1,
            py::arg("replace_deleted") = false)
        .def("get_items", &Index<float>::getData, py::arg("ids") = py::none(), py::arg("return_type") = "numpy")
        .def("get_ids_list", &Index<float>::getIdsList)
        .def("set_ef", &Index<float>::set_ef, py::arg("ef"))
        .def("set_num_threads", &Index<float>::set_num_threads, py::arg("num_threads"))
        .def("index_file_size", &Index<float>::indexFileSize)
        .def("save_index", &Index<float>::saveIndex, py::arg("path_to_index"))
        .def("load_index",
            &Index<float>::loadIndex,
            py::arg("path_to_index"),
            py::arg("max_elements") = 0,
            py::arg("allow_replace_deleted") = false)
        .def("init_index_mmap",
            &Index<float>::init_new_index_mmap,
            py::arg("max_elements"),
            py::arg("mmap_path"),
            py::arg("M") = 16,
            py::arg("ef_construction") = 200,
            py::arg("random_seed") = 100,
            py::arg("allow_replace_deleted") = false,
            R"pbdoc(
            Initialize a new index with mmap-backed storage.

            This creates a file-backed mmap instead of allocating memory with malloc,
            allowing you to build indexes larger than available RAM.

            Args:
                max_elements: Maximum number of elements in the index
                mmap_path: Path to the mmap file (will be created/overwritten)
                M: HNSW M parameter (default: 16)
                ef_construction: Build quality parameter (default: 200)
                random_seed: Random seed for reproducibility (default: 100)
                allow_replace_deleted: Whether to allow replacing deleted elements (default: False)
            )pbdoc")
        .def("load_index_mmap",
            &Index<float>::loadIndexMmap,
            py::arg("path_to_index"),
            py::arg("read_only") = true,
            R"pbdoc(
            Load an existing index using mmap for read-only access.

            This mmaps the index file directly instead of reading it into RAM,
            useful for very large indexes where you want to rely on OS paging.

            Args:
                path_to_index: Path to the index file
                read_only: If True, map as read-only (default). If False, map read-write.
            )pbdoc")
        .def("init_index_graph_only",
            &Index<float>::init_graph_only,
            py::arg("max_elements"),
            py::arg("external_vectors"),
            py::arg("M") = 16,
            py::arg("ef_construction") = 200,
            py::arg("random_seed") = 100,
            R"pbdoc(
            Initialize a graph-only index with external vectors.

            This allocates only the graph structure (~68 bytes/element for M=8)
            instead of graph + vectors (~1612 bytes/element for M=8, dim=384).
            Memory savings: ~96% for dim=384!

            Vectors are fetched from the provided numpy array during build and search.
            The array must stay valid for the lifetime of the index.

            Args:
                max_elements: Maximum number of elements in the index
                external_vectors: Numpy array of shape (max_elements, dim) with all vectors
                M: HNSW M parameter (default: 16)
                ef_construction: Build quality parameter (default: 200)
                random_seed: Random seed for reproducibility (default: 100)

            Example:
                vectors = np.random.randn(1000000, 384).astype('float32')
                index = hnswlib.Index(space='cosine', dim=384)
                index.init_index_graph_only(1000000, vectors, M=8)
                index.add_items(vectors)  # vectors already set, just builds graph
                index.save_graph('index.graph')  # Much smaller than save_index!
            )pbdoc")
        .def("set_external_vectors",
            &Index<float>::set_external_vectors,
            py::arg("external_vectors"),
            R"pbdoc(
            Set external vectors for a graph-only index.

            Use this after loading a graph with load_graph() to provide the vectors
            needed for search.

            Args:
                external_vectors: Numpy array of shape (n_elements, dim), dtype=float32
            )pbdoc")
        .def("set_external_vectors_fp16",
            &Index<float>::set_external_vectors_fp16,
            py::arg("external_vectors"),
            R"pbdoc(
            Set external vectors in fp16 format for a graph-only index.

            Vectors will be converted to fp32 on-the-fly during distance calculations.
            This is useful when you have fp16 vectors mmap'd from disk and want to avoid
            doubling memory usage by converting the entire array to fp32.

            Args:
                external_vectors: Numpy array of shape (n_elements, dim), dtype=float16
                                  Pass the array.view(np.uint16) to this method.

            Example:
                # Load fp16 vectors from mmap
                vectors_fp16 = np.memmap('vectors.mmap', dtype=np.float16, mode='r',
                                         shape=(n_elements, dim))

                index = hnswlib.Index(space='cosine', dim=dim)
                index.load_graph('index.graph')
                index.set_external_vectors_fp16(vectors_fp16.view(np.uint16))
                labels, distances = index.knn_query(query, k=10)  # query must be fp32
            )pbdoc")
        .def("save_graph",
            &Index<float>::saveGraph,
            py::arg("path_to_graph"),
            R"pbdoc(
            Save only the graph structure (no vectors).

            The resulting file is MUCH smaller than save_index() since it excludes
            the vectors. Graph-only files are typically ~95% smaller.

            Load with load_graph() and call set_external_vectors() to provide vectors.

            Args:
                path_to_graph: Path to save the graph file
            )pbdoc")
        .def("load_graph",
            &Index<float>::loadGraph,
            py::arg("path_to_graph"),
            R"pbdoc(
            Load a graph-only index saved with save_graph().

            After loading, you MUST call set_external_vectors() to provide the vectors
            before searching.

            Args:
                path_to_graph: Path to the graph file

            Example:
                index = hnswlib.Index(space='cosine', dim=384)
                index.load_graph('index.graph')
                index.set_external_vectors(my_vectors)  # mmap'd numpy array
                labels, distances = index.knn_query(query, k=10)
            )pbdoc")
        .def("mark_deleted", &Index<float>::markDeleted, py::arg("label"))
        .def("unmark_deleted", &Index<float>::unmarkDeleted, py::arg("label"))
        .def("resize_index", &Index<float>::resizeIndex, py::arg("new_size"))
        .def("get_max_elements", &Index<float>::getMaxElements)
        .def("get_current_count", &Index<float>::getCurrentCount)
        .def_readonly("space", &Index<float>::space_name)
        .def_readonly("dim", &Index<float>::dim)
        .def_readwrite("num_threads", &Index<float>::num_threads_default)
        .def_property("ef",
          [](const Index<float> & index) {
            return index.index_inited ? index.appr_alg->ef_ : index.default_ef;
          },
          [](Index<float> & index, const size_t ef_) {
            index.default_ef = ef_;
            if (index.appr_alg)
              index.appr_alg->ef_ = ef_;
        })
        .def_property_readonly("max_elements", [](const Index<float> & index) {
            return index.index_inited ? index.appr_alg->max_elements_ : 0;
        })
        .def_property_readonly("element_count", [](const Index<float> & index) {
            return index.index_inited ? (size_t)index.appr_alg->cur_element_count : 0;
        })
        .def_property_readonly("ef_construction", [](const Index<float> & index) {
          return index.index_inited ? index.appr_alg->ef_construction_ : 0;
        })
        .def_property_readonly("M",  [](const Index<float> & index) {
          return index.index_inited ? index.appr_alg->M_ : 0;
        })

        .def(py::pickle(
            [](const Index<float> &ind) {  // __getstate__
                return py::make_tuple(ind.getIndexParams()); /* Return dict (wrapped in a tuple) that fully encodes state of the Index object */
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 1)
                    throw std::runtime_error("Invalid state!");
                return Index<float>::createFromParams(t[0].cast<py::dict>());
            }))

        .def("__repr__", [](const Index<float> &a) {
            return "<hnswlib.Index(space='" + a.space_name + "', dim="+std::to_string(a.dim)+")>";
        });

        py::class_<BFIndex<float>>(m, "BFIndex")
        .def(py::init<const std::string &, const int>(), py::arg("space"), py::arg("dim"))
        .def("init_index", &BFIndex<float>::init_new_index, py::arg("max_elements"))
        .def("knn_query",
            &BFIndex<float>::knnQuery_return_numpy,
            py::arg("data"),
            py::arg("k") = 1,
            py::arg("num_threads") = -1,
            py::arg("filter") = py::none())
        .def("add_items", &BFIndex<float>::addItems, py::arg("data"), py::arg("ids") = py::none())
        .def("delete_vector", &BFIndex<float>::deleteVector, py::arg("label"))
        .def("set_num_threads", &BFIndex<float>::set_num_threads, py::arg("num_threads"))
        .def("save_index", &BFIndex<float>::saveIndex, py::arg("path_to_index"))
        .def("load_index", &BFIndex<float>::loadIndex, py::arg("path_to_index"), py::arg("max_elements") = 0)
        .def("__repr__", [](const BFIndex<float> &a) {
            return "<hnswlib.BFIndex(space='" + a.space_name + "', dim="+std::to_string(a.dim)+")>";
        })
        .def("get_max_elements", &BFIndex<float>::getMaxElements)
        .def("get_current_count", &BFIndex<float>::getCurrentCount)
        .def_readwrite("num_threads", &BFIndex<float>::num_threads_default);

        py::class_<ShardedIndexPy<float>>(m, "ShardedIndex")
        .def(py::init<const std::string &, const int>(), py::arg("space"), py::arg("dim"),
            R"pbdoc(
            Create a sharded HNSW index for large-scale vector search.

            Sharding enables building indexes larger than available RAM by splitting
            the data into multiple smaller indexes (shards) that are built independently.

            For very large datasets, a monolithic index would exceed RAM during build.
            With shards of ~10M vectors each, you only need ~15GB RAM per shard.

            Args:
                space: Distance metric ('l2', 'ip', or 'cosine')
                dim: Vector dimension

            Example (building shards):
                import hnswlib
                import numpy as np

                index = hnswlib.ShardedIndex('cosine', dim=1536)

                # Build shards one at a time
                for shard_idx in range(n_shards):
                    start = shard_idx * shard_size
                    end = min(start + shard_size, n_total)

                    # Load only this shard's vectors
                    shard_vectors = load_vectors(start, end)  # ~15GB RAM
                    index.build_shard(shard_vectors, 'index_dir/', shard_idx, start,
                                      M=8, ef_construction=200)

                index.save_metadata('index_dir/', n_shards, n_total, M=8, ef_construction=200)

            Example (searching):
                index = hnswlib.ShardedIndex('cosine', dim=1536)

                # Load all vectors as mmap (no RAM used for vectors)
                vectors = np.memmap('vectors.mmap', dtype=np.float16, mode='r',
                                    shape=(n_total, dim))
                index.load_shards_fp16('index_dir/', vectors.view(np.uint16), n_shards)

                index.set_ef(50)
                labels, distances = index.knn_query(query, k=10)
            )pbdoc")
        .def("build_shard",
            &ShardedIndexPy<float>::buildShard,
            py::arg("vectors"),
            py::arg("output_dir"),
            py::arg("shard_idx"),
            py::arg("start_idx"),
            py::arg("M") = 8,
            py::arg("ef_construction") = 200,
            py::arg("num_threads") = -1,
            R"pbdoc(
            Build a single shard of the index.

            Call once per shard, loading only that shard's vectors into RAM.
            The graph is saved to disk immediately.

            Args:
                vectors: Numpy array of shape (n_elements, dim), dtype=float32
                output_dir: Directory to save shard graphs
                shard_idx: Index of this shard (0, 1, 2, ...)
                start_idx: Global starting index for element IDs
                M: HNSW M parameter (default: 8)
                ef_construction: Build quality parameter (default: 200)
                num_threads: Number of threads for building (default: all CPUs)
            )pbdoc")
        .def("save_metadata",
            &ShardedIndexPy<float>::saveMetadata,
            py::arg("output_dir"),
            py::arg("n_shards"),
            py::arg("total_elements"),
            py::arg("M"),
            py::arg("ef_construction"),
            R"pbdoc(
            Save global index metadata after all shards are built.

            Args:
                output_dir: Directory containing shard graphs
                n_shards: Total number of shards
                total_elements: Total number of vectors across all shards
                M: HNSW M parameter used during build
                ef_construction: Build quality parameter used during build
            )pbdoc")
        .def("load_shards",
            &ShardedIndexPy<float>::loadShards,
            py::arg("output_dir"),
            py::arg("external_vectors"),
            py::arg("n_shards"),
            R"pbdoc(
            Load all shards for search with fp32 vectors.

            Args:
                output_dir: Directory containing shard graphs
                external_vectors: Numpy array of all vectors, shape (n_total, dim), dtype=float32
                n_shards: Number of shards to load
            )pbdoc")
        .def("load_shards_fp16",
            &ShardedIndexPy<float>::loadShardsFp16,
            py::arg("output_dir"),
            py::arg("external_vectors"),
            py::arg("n_shards"),
            py::arg("use_mmap_graphs") = false,
            R"pbdoc(
            Load all shards for search with fp16 vectors.

            Vectors are converted to fp32 on-the-fly during search. This halves
            memory usage compared to fp32.

            Args:
                output_dir: Directory containing shard graphs
                external_vectors: Numpy array as uint16 view of float16 data.
                                  Pass your_fp16_array.view(np.uint16)
                n_shards: Number of shards to load
                use_mmap_graphs: If True, memory-map graph files instead of loading
                                 into RAM. Uses ~4GB instead of ~28GB for 64 shards,
                                 but may be slower due to disk I/O during search.
            )pbdoc")
        .def("load_shards_pq",
            &ShardedIndexPy<float>::loadShardsPQ,
            py::arg("output_dir"),
            py::arg("pq_codes"),
            py::arg("codebooks"),
            py::arg("n_shards"),
            py::arg("n_subvectors"),
            py::arg("n_centroids"),
            py::arg("subvector_dim"),
            py::arg("use_mmap_graphs") = false,
            py::arg("n_ram_shards") = 0,
            py::arg("sort_order") = py::none(),
            R"pbdoc(
            Load all shards for search with Product Quantization codes.

            PQ compresses vectors to n_subvectors bytes each, enabling search on
            datasets much larger than RAM. For hundreds of millions of vectors,
            PQ provides ~8x compression with ~94% recall@10.

            During search, a distance lookup table is computed from the query,
            then each candidate distance is just n_subvectors table lookups.

            Args:
                output_dir: Directory containing shard graphs
                pq_codes: Numpy array of shape (n_vectors, n_subvectors), dtype=uint8
                          Memory-map this for minimal RAM usage!
                codebooks: Numpy array of shape (n_subvectors, n_centroids, subvector_dim),
                           dtype=float32. Typically ~1-2MB for 192 subvectors.
                n_shards: Number of shards to load
                n_subvectors: Number of PQ subvectors (M)
                n_centroids: Centroids per subspace (K, typically 256)
                subvector_dim: Dimension of each subspace (dim / n_subvectors)
                use_mmap_graphs: If True, memory-map graph files instead of loading into RAM
                n_ram_shards: Hybrid mode - load first N shards into RAM, mmap the rest (0=disabled)
                sort_order: Optional mapping for cluster-based sharding. Numpy array of shape
                           (n_vectors,) dtype=int64 where sort_order[reordered_idx] = original_idx.
                           Required when shards group vectors by cluster rather than sequential order.

            Example:
                import numpy as np
                import hnswlib

                # Load PQ data (mmap for minimal RAM)
                pq_codes = np.memmap('pq_codes.mmap', dtype=np.uint8, mode='r',
                                      shape=(313628622, 192))
                codebooks_data = np.load('pq_codebooks.npz')
                codebooks = codebooks_data['codebooks']  # (192, 256, 2)

                # Create index and load with PQ
                index = hnswlib.ShardedIndex('cosine', dim=384)
                index.load_shards_pq('index_dir/', pq_codes, codebooks, n_shards=64,
                                     n_subvectors=192, n_centroids=256, subvector_dim=2)
                index.set_ef(100)

                # Search (query must be normalized fp32)
                query = normalize(query_vector)  # shape (1, 384)
                labels, distances = index.knn_query(query, k=10)
            )pbdoc")
        .def("knn_query",
            &ShardedIndexPy<float>::knnQuery,
            py::arg("data"),
            py::arg("k") = 1,
            py::arg("num_threads") = -1,
            R"pbdoc(
            Search across all shards and return top-k results.

            Results are returned with global element indices (not shard-local).

            Args:
                data: Query vector(s), shape (n_queries, dim) or (dim,)
                k: Number of results per query
                num_threads: Number of threads (default: all CPUs)

            Returns:
                Tuple of (labels, distances), each shape (n_queries, k)
            )pbdoc")
        .def("knn_query_selective",
            &ShardedIndexPy<float>::knnQuerySelective,
            py::arg("data"),
            py::arg("shard_ids"),
            py::arg("k") = 1,
            py::arg("num_threads") = -1,
            R"pbdoc(
            Search only specific shards and return top-k results.

            This enables IVF-style search where only relevant clusters are searched,
            dramatically reducing I/O and compute. Use with cluster centroids to
            determine which shards to search.

            Args:
                data: Query vector(s), shape (n_queries, dim) or (dim,)
                shard_ids: Array of shard indices to search, shape (n_shards_to_search,)
                k: Number of results per query
                num_threads: Number of threads (default: all CPUs)

            Returns:
                Tuple of (labels, distances), each shape (n_queries, k)

            Example:
                # Load centroids and find nearest clusters
                centroids = np.load('centroids.npy')
                similarities = query @ centroids.T
                top_clusters = np.argsort(-similarities)[:n_probe]

                # Search only those clusters
                labels, distances = index.knn_query_selective(query, top_clusters, k=10)
            )pbdoc")
        .def("set_ef", &ShardedIndexPy<float>::set_ef, py::arg("ef"),
            R"pbdoc(Set ef search parameter for all shards.)pbdoc")
        .def("set_num_threads", &ShardedIndexPy<float>::set_num_threads, py::arg("num_threads"))
        .def("set_labels_are_global", &ShardedIndexPy<float>::set_labels_are_global, py::arg("value"),
            R"pbdoc(
            Set whether stored labels in graphs are already global indices.

            When True, search methods return labels as-is without adding the shard's
            start_idx offset. Use this when graphs were built with global labels
            (e.g., IVF-HNSW where labels are reordered indices).

            Args:
                value: True if labels are already global, False for local 0-based labels
            )pbdoc")
        .def("get_total_elements", &ShardedIndexPy<float>::getTotalElements)
        .def("get_num_shards", &ShardedIndexPy<float>::getNumShards)
        .def_readonly("space", &ShardedIndexPy<float>::space_name)
        .def_readonly("dim", &ShardedIndexPy<float>::dim)
        .def_readwrite("num_threads", &ShardedIndexPy<float>::num_threads_default)
        .def_property("ef",
            [](const ShardedIndexPy<float>& idx) { return idx.default_ef; },
            [](ShardedIndexPy<float>& idx, size_t ef) { idx.set_ef(ef); })
        .def("__repr__", [](const ShardedIndexPy<float>& a) {
            return "<hnswlib.ShardedIndex(space='" + a.space_name +
                   "', dim=" + std::to_string(a.dim) +
                   ", shards=" + std::to_string(a.n_shards) +
                   ", elements=" + std::to_string(a.total_elements) + ")>";
        });

        return m.ptr();
}
