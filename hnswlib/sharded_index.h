#pragma once

#include "hnswalg.h"
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <queue>
#include <algorithm>
#include <sys/stat.h>
#include <sys/types.h>

namespace hnswlib {

/**
 * @brief Sharded HNSW index for large-scale vector search.
 *
 * This class enables building indexes larger than available RAM by splitting
 * the data into multiple shards, each built independently. At search time,
 * all shards are queried and results are merged.
 *
 * For very large datasets, a monolithic index would exceed RAM during build.
 * With shards of ~10M vectors each, you only need ~15GB RAM per shard.
 *
 * Example usage:
 *   ShardedIndex<float> index(space, dim);
 *
 *   // Build shards one at a time (can be done on limited RAM)
 *   for (int i = 0; i < n_shards; i++) {
 *       index.buildShard(shard_vectors, shard_ids, output_dir, i, M, ef_construction);
 *   }
 *
 *   // Load all shards for search
 *   index.loadShards(output_dir, external_vectors_fp16, n_shards);
 *
 *   // Search
 *   auto results = index.searchKnn(query, k);
 *
 * @tparam dist_t Distance type (typically float)
 */
template<typename dist_t>
class ShardedIndex {
public:
    struct ShardInfo {
        size_t start_idx;           // Global starting index for this shard
        size_t end_idx;             // Global ending index (exclusive)
        size_t n_elements;          // Number of elements in shard
        std::string graph_path;     // Path to saved graph
        std::unique_ptr<HierarchicalNSW<dist_t>> index;  // The loaded shard index
    };

    ShardedIndex(SpaceInterface<float>* s, size_t dim)
        : space_(s), dim_(dim), total_elements_(0) {}

    ~ShardedIndex() {
        // Clear shards first (they reference space_)
        shards_.clear();
    }

    /**
     * @brief Build a single shard of the index.
     *
     * This loads only the vectors for this shard into RAM, builds the graph,
     * and saves it to disk. Memory usage is O(shard_size * dim * sizeof(float)).
     *
     * @param vectors Pointer to vectors for this shard (contiguous, row-major)
     * @param n_elements Number of elements in this shard
     * @param output_dir Directory to save shard graphs
     * @param shard_idx Index of this shard
     * @param start_idx Global starting index for element IDs
     * @param M HNSW M parameter
     * @param ef_construction Build quality parameter
     * @param num_threads Number of threads for parallel insertion
     */
    void buildShard(
        const float* vectors,
        size_t n_elements,
        const std::string& output_dir,
        size_t shard_idx,
        size_t start_idx,
        size_t M = 8,
        size_t ef_construction = 200,
        size_t num_threads = 8
    ) {
        // Create graph-only index for this shard
        auto shard = std::make_unique<HierarchicalNSW<dist_t>>(
            space_, n_elements, vectors, M, ef_construction, 100 + shard_idx, false
        );

        // Add all elements in parallel using OpenMP
        #pragma omp parallel for schedule(dynamic) num_threads(num_threads)
        for (size_t i = 0; i < n_elements; i++) {
            shard->addPoint(vectors + i * dim_, i);
        }

        // Create output directory (simple mkdir, parent must exist)
        mkdir(output_dir.c_str(), 0755);
        std::string graph_path = output_dir + "/shard_" +
            std::to_string(shard_idx) + ".graph";
        shard->saveGraph(graph_path);

        // Update metadata
        ShardInfo info;
        info.start_idx = start_idx;
        info.end_idx = start_idx + n_elements;
        info.n_elements = n_elements;
        info.graph_path = graph_path;

        // Save shard metadata
        std::string meta_path = output_dir + "/shard_" +
            std::to_string(shard_idx) + ".meta";
        std::ofstream meta_out(meta_path, std::ios::binary);
        writeBinaryPOD(meta_out, info.start_idx);
        writeBinaryPOD(meta_out, info.end_idx);
        writeBinaryPOD(meta_out, info.n_elements);
    }

    /**
     * @brief Save global index metadata.
     *
     * Call after building all shards to save the index configuration.
     */
    void saveMetadata(
        const std::string& output_dir,
        size_t n_shards,
        size_t total_elements,
        size_t dim,
        size_t M,
        size_t ef_construction,
        const std::string& space_name
    ) {
        std::string meta_path = output_dir + "/index.meta";
        std::ofstream out(meta_path, std::ios::binary);

        // Magic number for validation
        uint32_t magic = 0x53484458;  // "SHDX"
        writeBinaryPOD(out, magic);
        writeBinaryPOD(out, n_shards);
        writeBinaryPOD(out, total_elements);
        writeBinaryPOD(out, dim);
        writeBinaryPOD(out, M);
        writeBinaryPOD(out, ef_construction);

        // Space name (with length prefix)
        size_t name_len = space_name.size();
        writeBinaryPOD(out, name_len);
        out.write(space_name.c_str(), name_len);
    }

    /**
     * @brief Load all shards from disk with fp32 external vectors.
     *
     * @param output_dir Directory containing shard graphs
     * @param external_vectors Pointer to all vectors (fp32, contiguous)
     * @param n_shards Number of shards to load
     */
    void loadShards(
        const std::string& output_dir,
        const float* external_vectors,
        size_t n_shards
    ) {
        shards_.clear();
        total_elements_ = 0;
        external_vectors_fp32_ = external_vectors;
        external_vectors_fp16_ = nullptr;
        use_fp16_ = false;

        for (size_t i = 0; i < n_shards; i++) {
            loadSingleShard(output_dir, i);
        }
    }

    /**
     * @brief Load all shards from disk with fp16 external vectors.
     *
     * Vectors will be converted to fp32 on-the-fly during search.
     *
     * @param output_dir Directory containing shard graphs
     * @param external_vectors Pointer to all vectors (fp16 as uint16_t, contiguous)
     * @param n_shards Number of shards to load
     * @param use_mmap_graphs If true, memory-map graph files instead of loading into RAM
     */
    void loadShardsFp16(
        const std::string& output_dir,
        const uint16_t* external_vectors,
        size_t n_shards,
        bool use_mmap_graphs = false
    ) {
        shards_.clear();
        total_elements_ = 0;
        external_vectors_fp32_ = nullptr;
        external_vectors_fp16_ = external_vectors;
        use_fp16_ = true;
        use_pq_ = false;
        use_mmap_graphs_ = use_mmap_graphs;

        for (size_t i = 0; i < n_shards; i++) {
            loadSingleShard(output_dir, i);
        }
    }

    /**
     * @brief Load all shards from disk with Product Quantization codes.
     *
     * PQ enables searching very large datasets by compressing vectors to
     * n_subvectors bytes each. Distance computation uses asymmetric lookup
     * tables computed from the query vector.
     *
     * @param output_dir Directory containing shard graphs
     * @param pq_codes Pointer to all PQ codes (n_vectors, n_subvectors) uint8
     * @param codebooks Pointer to codebooks (n_subvectors, n_centroids, subvector_dim) float32
     * @param n_shards Number of shards to load
     * @param n_subvectors Number of PQ subvectors (M)
     * @param n_centroids Number of centroids per subspace (K, typically 256)
     * @param subvector_dim Dimension of each subspace (dim / n_subvectors)
     * @param use_mmap_graphs If true, memory-map graph files instead of loading into RAM
     * @param n_ram_shards For hybrid mode: load first N shards into RAM, rest mmap'd (0 = disabled)
     * @param sort_order Optional mapping for cluster-based sharding: sort_order[reordered_idx] = original_idx.
     *                   If provided, PQ lookup uses sort_order[start_idx + label] instead of direct indexing.
     */
    void loadShardsPQ(
        const std::string& output_dir,
        const uint8_t* pq_codes,
        const float* codebooks,
        size_t n_shards,
        size_t n_subvectors,
        size_t n_centroids,
        size_t subvector_dim,
        bool use_mmap_graphs = false,
        size_t n_ram_shards = 0,
        const int64_t* sort_order = nullptr
    ) {
        shards_.clear();
        total_elements_ = 0;
        external_vectors_fp32_ = nullptr;
        external_vectors_fp16_ = nullptr;
        use_fp16_ = false;
        use_pq_ = true;
        use_mmap_graphs_ = use_mmap_graphs;
        n_ram_shards_ = n_ram_shards;

        // Store PQ parameters
        external_pq_codes_ = pq_codes;
        pq_codebooks_ = codebooks;
        pq_n_subvectors_ = n_subvectors;
        pq_n_centroids_ = n_centroids;
        pq_subvector_dim_ = subvector_dim;
        pq_sort_order_ = sort_order;

        for (size_t i = 0; i < n_shards; i++) {
            loadSingleShard(output_dir, i);
        }
    }

    /**
     * @brief Set ef search parameter for all shards.
     */
    void setEf(size_t ef) {
        for (auto& shard : shards_) {
            shard.index->ef_ = ef;
        }
    }

    /**
     * @brief Set whether stored labels are already global indices.
     *
     * When true, the search methods will return labels as-is without adding
     * the shard's start_idx. Use this when graphs were built with global
     * labels (e.g., IVF-HNSW where labels are reordered indices).
     */
    void setLabelsAreGlobal(bool value) {
        labels_are_global_ = value;
    }

    /**
     * @brief Search across all shards and return top-k results.
     *
     * Results are global element indices (not shard-local).
     *
     * @param query Query vector (normalized for cosine)
     * @param k Number of results
     * @return Priority queue of (distance, global_id) pairs
     */
    std::priority_queue<std::pair<dist_t, labeltype>> searchKnn(
        const void* query,
        size_t k
    ) const {
        // Collect results from all shards
        std::vector<std::pair<dist_t, labeltype>> all_results;
        all_results.reserve(k * shards_.size());

        for (const auto& shard : shards_) {
            auto shard_results = shard.index->searchKnn(query, k);

            // Convert to global indices (unless labels are already global)
            while (!shard_results.empty()) {
                auto& top = shard_results.top();
                labeltype global_id = labels_are_global_ ? top.second : (top.second + shard.start_idx);
                all_results.push_back({top.first, global_id});
                shard_results.pop();
            }
        }

        // Sort by distance and take top-k
        std::sort(all_results.begin(), all_results.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

        // Build result priority queue (max-heap for hnswlib compatibility)
        std::priority_queue<std::pair<dist_t, labeltype>> result;
        size_t count = std::min(k, all_results.size());
        for (size_t i = 0; i < count; i++) {
            result.push(all_results[i]);
        }

        return result;
    }

    /**
     * @brief Search only specific shards and return top-k results.
     *
     * This enables IVF-style search where only clusters near the query are searched,
     * dramatically reducing I/O and compute.
     *
     * @param query Query vector (normalized for cosine)
     * @param shard_ids Vector of shard indices to search
     * @param k Number of results
     * @return Priority queue of (distance, global_id) pairs
     */
    std::priority_queue<std::pair<dist_t, labeltype>> searchKnnSelective(
        const void* query,
        const std::vector<size_t>& shard_ids,
        size_t k
    ) const {
        // Collect results from selected shards only
        std::vector<std::pair<dist_t, labeltype>> all_results;
        all_results.reserve(k * shard_ids.size());

        for (size_t shard_idx : shard_ids) {
            if (shard_idx >= shards_.size()) {
                continue;  // Skip invalid shard indices
            }
            const auto& shard = shards_[shard_idx];
            auto shard_results = shard.index->searchKnn(query, k);

            // Convert to global indices (unless labels are already global)
            while (!shard_results.empty()) {
                auto& top = shard_results.top();
                labeltype global_id = labels_are_global_ ? top.second : (top.second + shard.start_idx);
                all_results.push_back({top.first, global_id});
                shard_results.pop();
            }
        }

        // Sort by distance and take top-k
        std::sort(all_results.begin(), all_results.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

        // Build result priority queue (max-heap for hnswlib compatibility)
        std::priority_queue<std::pair<dist_t, labeltype>> result;
        size_t count = std::min(k, all_results.size());
        for (size_t i = 0; i < count; i++) {
            result.push(all_results[i]);
        }

        return result;
    }

    /**
     * @brief Get the total number of elements across all shards.
     */
    size_t getTotalElements() const { return total_elements_; }

    /**
     * @brief Get the number of loaded shards.
     */
    size_t getNumShards() const { return shards_.size(); }

private:
    void loadSingleShard(const std::string& output_dir, size_t shard_idx) {
        // Load shard metadata
        std::string meta_path = output_dir + "/shard_" +
            std::to_string(shard_idx) + ".meta";
        std::ifstream meta_in(meta_path, std::ios::binary);
        if (!meta_in) {
            throw std::runtime_error("Cannot open shard metadata: " + meta_path);
        }

        ShardInfo info;
        readBinaryPOD(meta_in, info.start_idx);
        readBinaryPOD(meta_in, info.end_idx);
        readBinaryPOD(meta_in, info.n_elements);
        info.graph_path = output_dir + "/shard_" + std::to_string(shard_idx) + ".graph";

        // Load graph - either mmap'd or into RAM
        // Hybrid mode: load first n_ram_shards_ into RAM, rest mmap'd
        info.index = std::make_unique<HierarchicalNSW<dist_t>>(space_);
        bool use_mmap_for_this_shard = use_mmap_graphs_ ||
            (n_ram_shards_ > 0 && shard_idx >= n_ram_shards_);

        if (use_mmap_for_this_shard) {
            // Memory-map graph file (minimal RAM usage)
            info.index->loadGraphMmap(info.graph_path, space_);
        } else {
            // Load into RAM in read-only mode (skips mutex/label_lookup_)
            info.index->loadGraph(info.graph_path, space_, true /* read_only */);
        }

        // Set external vectors for this shard
        if (use_pq_) {
            if (pq_sort_order_) {
                // Cluster-based sharding: use sort_order mapping
                // PQ codes are in original order, sort_order maps reordered -> original
                info.index->setExternalVectorPointerPQ(
                    external_pq_codes_,  // Full PQ codes array (in original order)
                    pq_codebooks_,
                    pq_n_subvectors_,
                    pq_n_centroids_,
                    pq_subvector_dim_,
                    pq_sort_order_,      // Mapping array
                    info.start_idx       // Offset for this shard
                );
            } else {
                // Sequential sharding: direct indexing with offset
                const uint8_t* shard_codes = external_pq_codes_ + info.start_idx * pq_n_subvectors_;
                info.index->setExternalVectorPointerPQ(
                    shard_codes,
                    pq_codebooks_,
                    pq_n_subvectors_,
                    pq_n_centroids_,
                    pq_subvector_dim_
                );
            }
        } else if (use_fp16_) {
            // Point to the correct offset in the global fp16 vector array
            const uint16_t* shard_vectors = external_vectors_fp16_ + info.start_idx * dim_;
            info.index->setExternalVectorPointerFp16(shard_vectors, dim_);
        } else {
            // Point to the correct offset in the global fp32 vector array
            const float* shard_vectors = external_vectors_fp32_ + info.start_idx * dim_;
            info.index->setExternalVectorPointer(shard_vectors, dim_ * sizeof(float));
        }

        total_elements_ += info.n_elements;
        shards_.push_back(std::move(info));
    }

    SpaceInterface<float>* space_;
    size_t dim_;
    size_t total_elements_;
    bool use_fp16_ = false;
    bool use_pq_ = false;
    bool use_mmap_graphs_ = false;
    bool labels_are_global_ = false;  // If true, stored labels are already global (don't add start_idx)
    size_t n_ram_shards_ = 0;  // For hybrid mode: load first N shards into RAM
    const float* external_vectors_fp32_ = nullptr;
    const uint16_t* external_vectors_fp16_ = nullptr;

    // PQ parameters
    const uint8_t* external_pq_codes_ = nullptr;
    const float* pq_codebooks_ = nullptr;
    size_t pq_n_subvectors_ = 0;
    size_t pq_n_centroids_ = 0;
    size_t pq_subvector_dim_ = 0;
    const int64_t* pq_sort_order_ = nullptr;  // For cluster sharding: maps reordered_idx -> original_idx

    std::vector<ShardInfo> shards_;
};

}  // namespace hnswlib
