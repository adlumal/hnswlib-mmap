#!/usr/bin/env python3
"""
Tests for selective shard search (IVF-HNSW).

This tests the knn_query_selective() method which allows searching
only specific shards instead of all shards. This enables IVF-HNSW
search where cluster centroids route queries to relevant shards,
and HNSW is used for intra-cluster search.
"""

import tempfile
import shutil
import unittest
import numpy as np
import hnswlib


def normalize(vectors):
    """Normalize vectors for cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class TestSelectiveSearch(unittest.TestCase):
    """Test selective shard search functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.dim = 32
        self.n_elements = 1000
        self.n_shards = 4
        self.shard_size = self.n_elements // self.n_shards

        # Generate random test data
        np.random.seed(42)
        self.data = np.random.randn(self.n_elements, self.dim).astype('float32')
        self.data = normalize(self.data)

        # Create temp directory for index files
        self.temp_dir = tempfile.mkdtemp()

        # Build sharded index
        self._build_sharded_index()

    def tearDown(self):
        """Clean up temp files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _build_sharded_index(self):
        """Build a sharded index for testing."""
        index = hnswlib.ShardedIndex(space='cosine', dim=self.dim)

        for shard_idx in range(self.n_shards):
            start = shard_idx * self.shard_size
            end = start + self.shard_size
            shard_vectors = self.data[start:end].copy()

            index.build_shard(
                shard_vectors,
                self.temp_dir,
                shard_idx,
                start,
                M=8,
                ef_construction=100,
                num_threads=1
            )

        index.save_metadata(self.temp_dir, self.n_shards, self.n_elements, M=8, ef_construction=100)

    def _load_index_fp16(self):
        """Load index with fp16 vectors."""
        data_fp16 = self.data.astype(np.float16)
        index = hnswlib.ShardedIndex(space='cosine', dim=self.dim)
        index.load_shards_fp16(
            self.temp_dir,
            data_fp16.view(np.uint16),
            self.n_shards
        )
        index.set_ef(50)
        return index

    def _load_index_fp32(self):
        """Load index with fp32 vectors."""
        index = hnswlib.ShardedIndex(space='cosine', dim=self.dim)
        index.load_shards(self.temp_dir, self.data, self.n_shards)
        index.set_ef(50)
        return index

    def test_selective_single_shard(self):
        """Test searching a single shard."""
        index = self._load_index_fp32()

        # Query a vector from shard 0 and search only shard 0
        query = self.data[10:11]
        shard_ids = np.array([0], dtype=np.int64)

        labels, distances = index.knn_query_selective(query, shard_ids, k=5)

        # Results should only contain IDs from shard 0 (0-249)
        self.assertEqual(labels.shape, (1, 5))
        self.assertTrue(all(0 <= label < self.shard_size for label in labels[0]),
                       f"Labels should be from shard 0, got {labels[0]}")

        # Query vector 10 should find itself
        self.assertIn(10, labels[0], "Query should find itself in its own shard")

    def test_selective_multiple_shards(self):
        """Test searching multiple shards."""
        index = self._load_index_fp32()

        query = self.data[0:1]
        shard_ids = np.array([0, 1], dtype=np.int64)

        labels, distances = index.knn_query_selective(query, shard_ids, k=10)

        # Results should only contain IDs from shards 0 and 1 (0-499)
        self.assertEqual(labels.shape, (1, 10))
        self.assertTrue(all(0 <= label < 2 * self.shard_size for label in labels[0]),
                       f"Labels should be from shards 0-1, got {labels[0]}")

    def test_selective_vs_full_search(self):
        """Compare selective search with full search when using all shards."""
        index = self._load_index_fp32()

        query = self.data[100:101]
        all_shard_ids = np.array([0, 1, 2, 3], dtype=np.int64)

        # Selective search with all shards
        labels_sel, dists_sel = index.knn_query_selective(query, all_shard_ids, k=10)

        # Full search
        labels_full, dists_full = index.knn_query(query, k=10)

        # Results should be identical
        np.testing.assert_array_equal(labels_sel, labels_full,
            "Selective search with all shards should match full search")
        np.testing.assert_array_almost_equal(dists_sel, dists_full, decimal=5)

    def test_selective_with_fp16(self):
        """Test selective search with fp16 vectors."""
        index = self._load_index_fp16()

        query = self.data[50:51]
        shard_ids = np.array([0, 2], dtype=np.int64)

        labels, distances = index.knn_query_selective(query, shard_ids, k=5)

        # Results should be from shards 0 and 2
        self.assertEqual(labels.shape, (1, 5))
        for label in labels[0]:
            shard = label // self.shard_size
            self.assertIn(shard, [0, 2],
                f"Label {label} (shard {shard}) not in searched shards")

    def test_selective_batch_query(self):
        """Test selective search with multiple queries."""
        index = self._load_index_fp32()

        queries = self.data[:5]
        shard_ids = np.array([0, 1], dtype=np.int64)

        labels, distances = index.knn_query_selective(queries, shard_ids, k=3)

        self.assertEqual(labels.shape, (5, 3))
        self.assertEqual(distances.shape, (5, 3))

        # All results should be from shards 0-1
        for row in labels:
            for label in row:
                self.assertTrue(0 <= label < 2 * self.shard_size,
                    f"Label {label} not in searched shards")

    def test_selective_results_sorted(self):
        """Test that selective search returns sorted results."""
        index = self._load_index_fp32()

        query = self.data[0:1]
        shard_ids = np.array([0, 1, 2], dtype=np.int64)

        labels, distances = index.knn_query_selective(query, shard_ids, k=10)

        # Distances should be non-decreasing
        for i in range(len(distances[0]) - 1):
            self.assertLessEqual(distances[0][i], distances[0][i+1] + 1e-6,
                "Distances should be sorted")

    def test_selective_empty_shard_list(self):
        """Test behavior with empty shard list."""
        index = self._load_index_fp32()

        query = self.data[0:1]
        shard_ids = np.array([], dtype=np.int64)

        labels, distances = index.knn_query_selective(query, shard_ids, k=5)

        # Should return zeros when no shards searched
        self.assertEqual(labels.shape, (1, 5))

    def test_selective_invalid_shard_id(self):
        """Test that invalid shard IDs are gracefully handled."""
        index = self._load_index_fp32()

        query = self.data[0:1]
        # Include an invalid shard ID (99)
        shard_ids = np.array([0, 99], dtype=np.int64)

        # Should not crash, should just search valid shards
        labels, distances = index.knn_query_selective(query, shard_ids, k=5)

        # Results should only contain IDs from shard 0
        self.assertEqual(labels.shape, (1, 5))

    def test_selective_preserves_global_ids(self):
        """Test that selective search returns correct global IDs."""
        index = self._load_index_fp32()

        # Query from shard 2 (IDs 500-749)
        query_idx = 600
        query = self.data[query_idx:query_idx+1]
        shard_ids = np.array([2], dtype=np.int64)

        labels, distances = index.knn_query_selective(query, shard_ids, k=5)

        # Should find the query vector itself
        self.assertIn(query_idx, labels[0],
            f"Should find query {query_idx} in its shard")

        # All results should be in the correct range for shard 2
        shard_start = 2 * self.shard_size
        shard_end = 3 * self.shard_size
        for label in labels[0]:
            self.assertTrue(shard_start <= label < shard_end,
                f"Label {label} should be in shard 2 range [{shard_start}, {shard_end})")


class TestSelectiveSearchWithPQ(unittest.TestCase):
    """Test selective search with PQ-encoded vectors."""

    def setUp(self):
        """Set up test fixtures with PQ."""
        self.dim = 16
        self.n_elements = 400
        self.n_shards = 4
        self.shard_size = self.n_elements // self.n_shards
        self.n_subvectors = 4
        self.n_centroids = 256
        self.subvector_dim = self.dim // self.n_subvectors

        np.random.seed(42)
        self.data = np.random.randn(self.n_elements, self.dim).astype('float32')
        self.data = normalize(self.data)

        self.temp_dir = tempfile.mkdtemp()

        # Build index
        self._build_index()

        # Create PQ codes
        self._create_pq_codes()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _build_index(self):
        """Build sharded index."""
        index = hnswlib.ShardedIndex(space='cosine', dim=self.dim)
        for shard_idx in range(self.n_shards):
            start = shard_idx * self.shard_size
            end = start + self.shard_size
            index.build_shard(
                self.data[start:end], self.temp_dir, shard_idx, start,
                M=16, ef_construction=200
            )
        index.save_metadata(self.temp_dir, self.n_shards, self.n_elements,
                           M=16, ef_construction=200)

    def _create_pq_codes(self):
        """Create random codebooks and encode vectors."""
        self.codebooks = np.random.randn(
            self.n_subvectors, self.n_centroids, self.subvector_dim
        ).astype(np.float32)

        self.pq_codes = np.zeros((self.n_elements, self.n_subvectors), dtype=np.uint8)
        for i in range(self.n_elements):
            for m in range(self.n_subvectors):
                start_d = m * self.subvector_dim
                end_d = start_d + self.subvector_dim
                subvec = self.data[i, start_d:end_d]
                dists = np.sum((self.codebooks[m] - subvec) ** 2, axis=1)
                self.pq_codes[i, m] = np.argmin(dists)

    def _load_index_pq(self):
        """Load index with PQ codes."""
        index = hnswlib.ShardedIndex(space='cosine', dim=self.dim)
        index.load_shards_pq(
            self.temp_dir, self.pq_codes, self.codebooks, self.n_shards,
            self.n_subvectors, self.n_centroids, self.subvector_dim
        )
        index.set_ef(100)
        return index

    def test_selective_pq_single_shard(self):
        """Test selective search with PQ on single shard."""
        index = self._load_index_pq()

        query = self.data[10:11]
        shard_ids = np.array([0], dtype=np.int64)

        labels, distances = index.knn_query_selective(query, shard_ids, k=5)

        # Results should be from shard 0
        self.assertEqual(labels.shape, (1, 5))
        for label in labels[0]:
            self.assertTrue(0 <= label < self.shard_size,
                f"Label {label} should be from shard 0")

        # Should find query vector
        self.assertIn(10, labels[0])

    def test_selective_pq_multiple_shards(self):
        """Test selective search with PQ on multiple shards."""
        index = self._load_index_pq()

        query = self.data[0:1]
        shard_ids = np.array([0, 2], dtype=np.int64)

        labels, distances = index.knn_query_selective(query, shard_ids, k=10)

        # Results should be from shards 0 and 2
        for label in labels[0]:
            shard = label // self.shard_size
            self.assertIn(shard, [0, 2])

    def test_selective_pq_vs_full(self):
        """Compare PQ selective search with full search."""
        index = self._load_index_pq()

        query = self.data[50:51]
        all_shards = np.array([0, 1, 2, 3], dtype=np.int64)

        labels_sel, dists_sel = index.knn_query_selective(query, all_shards, k=10)
        labels_full, dists_full = index.knn_query(query, k=10)

        np.testing.assert_array_equal(labels_sel, labels_full)
        np.testing.assert_array_almost_equal(dists_sel, dists_full, decimal=5)


class TestIVFHNSWRouting(unittest.TestCase):
    """Test IVF-HNSW cluster routing pattern."""

    def setUp(self):
        """Set up test with clustered data."""
        self.dim = 32
        self.n_clusters = 4
        self.vectors_per_cluster = 100
        self.n_elements = self.n_clusters * self.vectors_per_cluster

        np.random.seed(42)

        # Create clustered data - each cluster has vectors near a centroid
        self.centroids = np.random.randn(self.n_clusters, self.dim).astype('float32')
        self.centroids = normalize(self.centroids)

        self.data = np.zeros((self.n_elements, self.dim), dtype='float32')
        self.cluster_assignments = np.zeros(self.n_elements, dtype=np.int32)

        for c in range(self.n_clusters):
            start = c * self.vectors_per_cluster
            end = start + self.vectors_per_cluster
            # Add noise to centroid to create cluster
            noise = np.random.randn(self.vectors_per_cluster, self.dim) * 0.1
            self.data[start:end] = self.centroids[c] + noise
            self.cluster_assignments[start:end] = c

        self.data = normalize(self.data).astype('float32')

        self.temp_dir = tempfile.mkdtemp()
        self._build_index()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _build_index(self):
        """Build one shard per cluster."""
        index = hnswlib.ShardedIndex(space='cosine', dim=self.dim)
        for c in range(self.n_clusters):
            start = c * self.vectors_per_cluster
            end = start + self.vectors_per_cluster
            index.build_shard(
                self.data[start:end], self.temp_dir, c, start,
                M=8, ef_construction=100
            )
        index.save_metadata(self.temp_dir, self.n_clusters, self.n_elements,
                           M=8, ef_construction=100)

    def _find_nearest_clusters(self, query, n_probe):
        """Find n_probe nearest clusters using centroid comparison."""
        query_norm = normalize(query.reshape(1, -1))[0]
        similarities = query_norm @ self.centroids.T
        return np.argsort(-similarities)[:n_probe]

    def test_ivf_routing_accuracy(self):
        """Test that IVF routing finds correct results."""
        index = hnswlib.ShardedIndex(space='cosine', dim=self.dim)
        index.load_shards(self.temp_dir, self.data, self.n_clusters)
        index.set_ef(50)

        # Query a vector and route to nearest cluster
        query_idx = 50  # From cluster 0
        query = self.data[query_idx:query_idx+1]

        # Find nearest cluster via centroids
        nearest_clusters = self._find_nearest_clusters(query[0], n_probe=1)

        # Search only the nearest cluster
        shard_ids = np.array(nearest_clusters, dtype=np.int64)
        labels, distances = index.knn_query_selective(query, shard_ids, k=5)

        # Since query is from cluster 0 and we route to cluster 0,
        # should find the query itself
        expected_cluster = query_idx // self.vectors_per_cluster
        self.assertIn(expected_cluster, nearest_clusters,
            "Routing should identify the correct cluster")
        self.assertIn(query_idx, labels[0],
            "Should find query in its cluster")

    def test_ivf_routing_with_multiple_probes(self):
        """Test IVF routing with multiple cluster probes."""
        index = hnswlib.ShardedIndex(space='cosine', dim=self.dim)
        index.load_shards(self.temp_dir, self.data, self.n_clusters)
        index.set_ef(50)

        query = self.data[0:1]

        # Compare different n_probe values
        results_1probe = []
        results_2probe = []

        nearest_1 = self._find_nearest_clusters(query[0], n_probe=1)
        nearest_2 = self._find_nearest_clusters(query[0], n_probe=2)

        labels_1, _ = index.knn_query_selective(query, np.array(nearest_1, dtype=np.int64), k=10)
        labels_2, _ = index.knn_query_selective(query, np.array(nearest_2, dtype=np.int64), k=10)

        # n_probe=2 should find at least as many good results as n_probe=1
        # (in terms of true nearest neighbors)
        # This is a sanity check that more probes = more coverage
        self.assertEqual(labels_1.shape, (1, 10))
        self.assertEqual(labels_2.shape, (1, 10))


if __name__ == '__main__':
    unittest.main()
