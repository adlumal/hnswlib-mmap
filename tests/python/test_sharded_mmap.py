"""
Tests for sharded index with memory-mapped graphs.
"""

import tempfile
import shutil
import unittest
import numpy as np
import hnswlib


class TestShardedIndexMmapGraphs(unittest.TestCase):
    """Test sharded index with mmap'd graph files."""

    def setUp(self):
        """Set up test fixtures."""
        self.dim = 64
        self.n_elements = 10000
        self.n_shards = 4
        self.shard_size = self.n_elements // self.n_shards

        # Generate random test data
        np.random.seed(42)
        self.data = np.random.randn(self.n_elements, self.dim).astype('float32')

        # Normalize for cosine
        norms = np.linalg.norm(self.data, axis=1, keepdims=True)
        self.data = self.data / np.maximum(norms, 1e-12)

        # Create temp directory for index files
        self.temp_dir = tempfile.mkdtemp()

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

        return index

    def test_load_shards_ram_vs_mmap(self):
        """Compare RAM-loaded vs mmap-loaded search results."""
        # Build index
        self._build_sharded_index()

        # Convert to fp16 for testing
        data_fp16 = self.data.astype(np.float16)

        # Load with RAM (default)
        index_ram = hnswlib.ShardedIndex(space='cosine', dim=self.dim)
        index_ram.load_shards_fp16(
            self.temp_dir,
            data_fp16.view(np.uint16),
            self.n_shards,
            use_mmap_graphs=False
        )
        index_ram.set_ef(50)

        # Load with mmap
        index_mmap = hnswlib.ShardedIndex(space='cosine', dim=self.dim)
        index_mmap.load_shards_fp16(
            self.temp_dir,
            data_fp16.view(np.uint16),
            self.n_shards,
            use_mmap_graphs=True
        )
        index_mmap.set_ef(50)

        # Query both
        query = self.data[:10]
        k = 5

        labels_ram, dists_ram = index_ram.knn_query(query, k=k)
        labels_mmap, dists_mmap = index_mmap.knn_query(query, k=k)

        # Results should be identical
        np.testing.assert_array_equal(labels_ram, labels_mmap)
        np.testing.assert_array_almost_equal(dists_ram, dists_mmap, decimal=5)

    def test_mmap_self_recall(self):
        """Test that mmap mode achieves good self-recall."""
        self._build_sharded_index()

        data_fp16 = self.data.astype(np.float16)

        # Load with mmap
        index = hnswlib.ShardedIndex(space='cosine', dim=self.dim)
        index.load_shards_fp16(
            self.temp_dir,
            data_fp16.view(np.uint16),
            self.n_shards,
            use_mmap_graphs=True
        )
        index.set_ef(100)

        # Query first 100 vectors
        n_test = 100
        queries = self.data[:n_test]
        labels, _ = index.knn_query(queries, k=1)

        # Check self-recall (should find themselves)
        expected = np.arange(n_test)
        recall = np.mean(labels.flatten() == expected)

        # Allow some tolerance due to fp16 quantization
        self.assertGreater(recall, 0.5,
            f"Self-recall too low: {recall:.2%}")

    def test_mmap_search_quality(self):
        """Test search quality with mmap graphs."""
        self._build_sharded_index()

        data_fp16 = self.data.astype(np.float16)

        index = hnswlib.ShardedIndex(space='cosine', dim=self.dim)
        index.load_shards_fp16(
            self.temp_dir,
            data_fp16.view(np.uint16),
            self.n_shards,
            use_mmap_graphs=True
        )
        index.set_ef(50)

        # Do a search
        query = self.data[0:1]
        labels, distances = index.knn_query(query, k=10)

        # Should return valid results
        self.assertEqual(labels.shape, (1, 10))
        self.assertEqual(distances.shape, (1, 10))

        # Distances should be non-negative and sorted
        self.assertTrue(np.all(distances >= 0))
        self.assertTrue(np.all(np.diff(distances[0]) >= -1e-6))  # Allow small tolerance

    def test_mmap_multiple_queries(self):
        """Test batch queries with mmap graphs."""
        self._build_sharded_index()

        data_fp16 = self.data.astype(np.float16)

        index = hnswlib.ShardedIndex(space='cosine', dim=self.dim)
        index.load_shards_fp16(
            self.temp_dir,
            data_fp16.view(np.uint16),
            self.n_shards,
            use_mmap_graphs=True
        )
        index.set_ef(50)

        # Batch query
        queries = self.data[:50]
        labels, distances = index.knn_query(queries, k=5)

        self.assertEqual(labels.shape, (50, 5))
        self.assertEqual(distances.shape, (50, 5))


class TestShardedIndexFp32Mmap(unittest.TestCase):
    """Test sharded index with fp32 vectors (non-fp16 path)."""

    def setUp(self):
        self.dim = 32
        self.n_elements = 5000
        self.n_shards = 2
        self.shard_size = self.n_elements // self.n_shards

        np.random.seed(123)
        self.data = np.random.randn(self.n_elements, self.dim).astype('float32')
        norms = np.linalg.norm(self.data, axis=1, keepdims=True)
        self.data = self.data / np.maximum(norms, 1e-12)

        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_fp32_build_and_search(self):
        """Test basic fp32 sharded index."""
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
                ef_construction=100
            )

        # Load with fp32
        index2 = hnswlib.ShardedIndex(space='cosine', dim=self.dim)
        index2.load_shards(self.temp_dir, self.data, self.n_shards)
        index2.set_ef(50)

        # Search
        labels, distances = index2.knn_query(self.data[:10], k=5)

        self.assertEqual(labels.shape, (10, 5))
        self.assertTrue(np.all(distances >= 0))


if __name__ == '__main__':
    unittest.main()
