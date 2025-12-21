#pragma once

#include <cstdint>
#include <cstring>
#include <vector>
#include <stdexcept>

namespace hnswlib {

/**
 * Product Quantization distance computation for asymmetric search.
 *
 * PQ splits each vector into M subvectors and quantizes each subspace
 * independently using K centroids. For search, we precompute a distance
 * lookup table from the query to all centroids, then each candidate
 * distance is just M table lookups + additions.
 *
 * Memory layout:
 * - Codebooks: (n_subvectors, n_centroids, subvector_dim) float32
 * - PQ codes: (n_vectors, n_subvectors) uint8
 * - Distance table: (n_subvectors, n_centroids) float32 per query
 */
class PQDistance {
public:
    size_t n_subvectors_;     // M: number of subvectors
    size_t n_centroids_;      // K: centroids per subspace (typically 256)
    size_t subvector_dim_;    // D/M: dimension of each subspace
    size_t dim_;              // D: full vector dimension

    const float* codebooks_;  // (n_subvectors, n_centroids, subvector_dim) float32

    PQDistance() : n_subvectors_(0), n_centroids_(0), subvector_dim_(0),
                   dim_(0), codebooks_(nullptr) {}

    PQDistance(const float* codebooks, size_t n_subvectors, size_t n_centroids,
               size_t subvector_dim)
        : n_subvectors_(n_subvectors), n_centroids_(n_centroids),
          subvector_dim_(subvector_dim), dim_(n_subvectors * subvector_dim),
          codebooks_(codebooks) {}

    /**
     * Precompute distance table from query to all centroids.
     *
     * For cosine similarity with normalized vectors, we compute squared L2
     * distances (which are equivalent to 2 - 2*cos_sim for unit vectors).
     *
     * @param query Query vector (dim,) - should be normalized for cosine
     * @param table Output distance table (n_subvectors * n_centroids)
     */
    void computeDistanceTable(const float* query, float* table) const {
        // For each subspace
        for (size_t m = 0; m < n_subvectors_; m++) {
            const float* query_sub = query + m * subvector_dim_;
            const float* centroids = codebooks_ + m * n_centroids_ * subvector_dim_;
            float* table_sub = table + m * n_centroids_;

            // Squared L2 distance to each centroid
            for (size_t k = 0; k < n_centroids_; k++) {
                const float* centroid = centroids + k * subvector_dim_;
                float dist = 0.0f;
                for (size_t d = 0; d < subvector_dim_; d++) {
                    float diff = query_sub[d] - centroid[d];
                    dist += diff * diff;
                }
                table_sub[k] = dist;
            }
        }
    }

    /**
     * Compute asymmetric distance using precomputed table.
     *
     * @param table Precomputed distance table (n_subvectors * n_centroids)
     * @param code PQ code for the database vector (n_subvectors uint8)
     * @return Approximate squared L2 distance
     */
    inline float asymmetricDistance(const float* table, const uint8_t* code) const {
        float dist = 0.0f;
        for (size_t m = 0; m < n_subvectors_; m++) {
            dist += table[m * n_centroids_ + code[m]];
        }
        return dist;
    }

    /**
     * Get the size of the distance table in bytes.
     */
    size_t getDistanceTableSize() const {
        return n_subvectors_ * n_centroids_ * sizeof(float);
    }

    /**
     * Get the size of a PQ code in bytes.
     */
    size_t getCodeSize() const {
        return n_subvectors_;
    }
};

/**
 * Thread-local storage for per-query distance tables.
 * Each thread maintains its own distance table to avoid allocation per query.
 */
class PQDistanceTableCache {
public:
    std::vector<float> table_;
    bool valid_{false};  // Whether table has been computed for current query

    void ensureSize(size_t n_subvectors, size_t n_centroids) {
        size_t required = n_subvectors * n_centroids;
        if (table_.size() < required) {
            table_.resize(required);
        }
        valid_ = false;
    }

    float* data() { return table_.data(); }
    void invalidate() { valid_ = false; }
    void setValid() { valid_ = true; }
    bool isValid() const { return valid_; }
};

}  // namespace hnswlib
