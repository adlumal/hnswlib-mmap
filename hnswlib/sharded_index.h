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
 * For 314M vectors with 1536 dimensions:
 * - Full index would need ~480GB RAM during build
 * - With 32 shards of ~10M vectors each: ~15GB RAM per shard
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
     */
    void loadShardsFp16(
        const std::string& output_dir,
        const uint16_t* external_vectors,
        size_t n_shards
    ) {
        shards_.clear();
        total_elements_ = 0;
        external_vectors_fp32_ = nullptr;
        external_vectors_fp16_ = external_vectors;
        use_fp16_ = true;

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

            // Convert to global indices
            while (!shard_results.empty()) {
                auto& top = shard_results.top();
                labeltype global_id = top.second + shard.start_idx;
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

        // Load graph
        info.index = std::make_unique<HierarchicalNSW<dist_t>>(space_);
        info.index->loadGraph(info.graph_path, space_);

        // Set external vectors for this shard
        if (use_fp16_) {
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
    const float* external_vectors_fp32_ = nullptr;
    const uint16_t* external_vectors_fp16_ = nullptr;
    std::vector<ShardInfo> shards_;
};

}  // namespace hnswlib
