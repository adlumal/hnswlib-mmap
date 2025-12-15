import os
import tempfile
import unittest

import numpy as np

import hnswlib


class ShardedIndexTestCase(unittest.TestCase):
    """Tests for sharded index functionality."""

    def setUp(self):
        self.dim = 128
        self.num_elements = 10000
        self.n_shards = 4
        np.random.seed(42)
        self.data = np.float32(np.random.random((self.num_elements, self.dim)))
        # Normalize for cosine
        self.data = self.data / np.linalg.norm(self.data, axis=1, keepdims=True)

    def test_build_and_search_shards(self):
        """Test building shards and searching across them."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Build shards
            index = hnswlib.ShardedIndex(space='cosine', dim=self.dim)

            shard_size = (self.num_elements + self.n_shards - 1) // self.n_shards

            for shard_idx in range(self.n_shards):
                start_idx = shard_idx * shard_size
                end_idx = min(start_idx + shard_size, self.num_elements)
                shard_vectors = self.data[start_idx:end_idx]

                index.build_shard(
                    shard_vectors, tmpdir, shard_idx, start_idx,
                    M=16, ef_construction=100, num_threads=4
                )

            index.save_metadata(tmpdir, self.n_shards, self.num_elements, M=16, ef_construction=100)

            # Verify shard files exist
            for i in range(self.n_shards):
                graph_path = os.path.join(tmpdir, f'shard_{i}.graph')
                meta_path = os.path.join(tmpdir, f'shard_{i}.meta')
                self.assertTrue(os.path.exists(graph_path), f"Missing graph file: {graph_path}")
                self.assertTrue(os.path.exists(meta_path), f"Missing meta file: {meta_path}")

            # Load shards with fp32 vectors
            index2 = hnswlib.ShardedIndex(space='cosine', dim=self.dim)
            index2.load_shards(tmpdir, self.data, self.n_shards)
            index2.set_ef(50)

            self.assertEqual(index2.get_num_shards(), self.n_shards)
            self.assertEqual(index2.get_total_elements(), self.num_elements)

            # Search - each vector should find itself
            labels, distances = index2.knn_query(self.data[:100], k=1)

            # Check recall
            expected = np.arange(100).reshape(-1, 1)
            recall = np.mean(labels == expected)
            self.assertGreater(recall, 0.95, f"Recall too low: {recall}")

    def test_sharded_fp16_search(self):
        """Test searching with fp16 external vectors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Build shards with fp32
            index = hnswlib.ShardedIndex(space='cosine', dim=self.dim)

            shard_size = (self.num_elements + self.n_shards - 1) // self.n_shards

            for shard_idx in range(self.n_shards):
                start_idx = shard_idx * shard_size
                end_idx = min(start_idx + shard_size, self.num_elements)
                shard_vectors = self.data[start_idx:end_idx]

                index.build_shard(
                    shard_vectors, tmpdir, shard_idx, start_idx,
                    M=16, ef_construction=100
                )

            index.save_metadata(tmpdir, self.n_shards, self.num_elements, M=16, ef_construction=100)

            # Load with fp16 vectors
            data_fp16 = self.data.astype(np.float16)

            index2 = hnswlib.ShardedIndex(space='cosine', dim=self.dim)
            index2.load_shards_fp16(tmpdir, data_fp16.view(np.uint16), self.n_shards)
            index2.set_ef(50)

            # Search all vectors for themselves
            labels, distances = index2.knn_query(self.data, k=1)

            # High recall expected (allow some loss from fp16)
            recall = np.mean(labels.flatten() == np.arange(self.num_elements))
            self.assertGreater(recall, 0.90, f"FP16 recall too low: {recall}")

    def test_sharded_k_results(self):
        """Test that we get correct k results from sharded search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Build with 2 shards
            index = hnswlib.ShardedIndex(space='cosine', dim=self.dim)

            half = self.num_elements // 2
            index.build_shard(self.data[:half], tmpdir, 0, 0, M=16, ef_construction=100)
            index.build_shard(self.data[half:], tmpdir, 1, half, M=16, ef_construction=100)
            index.save_metadata(tmpdir, 2, self.num_elements, M=16, ef_construction=100)

            # Load and search
            index2 = hnswlib.ShardedIndex(space='cosine', dim=self.dim)
            index2.load_shards(tmpdir, self.data, 2)
            index2.set_ef(100)

            # Search with k=10
            k = 10
            labels, distances = index2.knn_query(self.data[:10], k=k)

            self.assertEqual(labels.shape, (10, k))
            self.assertEqual(distances.shape, (10, k))

            # Distances should be sorted (ascending for cosine/ip)
            for i in range(10):
                sorted_dists = np.sort(distances[i])
                np.testing.assert_array_almost_equal(distances[i], sorted_dists)

            # First result should be the query itself (distance ~0)
            for i in range(10):
                self.assertEqual(labels[i, 0], i)
                self.assertLess(distances[i, 0], 0.01)

    def test_sharded_l2_space(self):
        """Test sharded index with L2 distance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Non-normalized data for L2
            data_l2 = np.float32(np.random.random((self.num_elements, self.dim)))

            index = hnswlib.ShardedIndex(space='l2', dim=self.dim)

            shard_size = self.num_elements // 2
            index.build_shard(data_l2[:shard_size], tmpdir, 0, 0, M=16, ef_construction=100)
            index.build_shard(data_l2[shard_size:], tmpdir, 1, shard_size, M=16, ef_construction=100)
            index.save_metadata(tmpdir, 2, self.num_elements, M=16, ef_construction=100)

            # Load and search
            index2 = hnswlib.ShardedIndex(space='l2', dim=self.dim)
            index2.load_shards(tmpdir, data_l2, 2)
            index2.set_ef(100)

            labels, distances = index2.knn_query(data_l2, k=1)
            recall = np.mean(labels.flatten() == np.arange(self.num_elements))
            self.assertGreater(recall, 0.95)

    def test_sharded_batch_query(self):
        """Test batch queries across shards."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = hnswlib.ShardedIndex(space='cosine', dim=self.dim)

            # Build 2 shards
            half = self.num_elements // 2
            index.build_shard(self.data[:half], tmpdir, 0, 0, M=16, ef_construction=100)
            index.build_shard(self.data[half:], tmpdir, 1, half, M=16, ef_construction=100)
            index.save_metadata(tmpdir, 2, self.num_elements, M=16, ef_construction=100)

            index2 = hnswlib.ShardedIndex(space='cosine', dim=self.dim)
            index2.load_shards(tmpdir, self.data, 2)
            index2.set_ef(50)

            # Batch query with 1000 vectors
            batch_size = 1000
            labels, distances = index2.knn_query(self.data[:batch_size], k=5)

            self.assertEqual(labels.shape, (batch_size, 5))

            # Each vector should find itself first
            first_matches = labels[:, 0] == np.arange(batch_size)
            self.assertGreater(np.mean(first_matches), 0.95)

    def test_sharded_graph_size(self):
        """Test that sharded graphs are reasonably sized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = hnswlib.ShardedIndex(space='cosine', dim=self.dim)

            shard_size = self.num_elements // 2
            index.build_shard(self.data[:shard_size], tmpdir, 0, 0, M=8, ef_construction=100)

            graph_path = os.path.join(tmpdir, 'shard_0.graph')
            graph_size = os.path.getsize(graph_path)

            # ~68 bytes per element for M=8 (rough estimate)
            # Allow some overhead
            bytes_per_element = graph_size / shard_size
            self.assertLess(bytes_per_element, 200,
                           f"Graph too large: {bytes_per_element:.1f} bytes/element")
            self.assertGreater(bytes_per_element, 30,
                              f"Graph too small: {bytes_per_element:.1f} bytes/element")

    def test_repr(self):
        """Test string representation."""
        index = hnswlib.ShardedIndex(space='cosine', dim=self.dim)
        repr_str = repr(index)
        self.assertIn('ShardedIndex', repr_str)
        self.assertIn('cosine', repr_str)
        self.assertIn(str(self.dim), repr_str)


class ShardedIndexEdgeCasesTestCase(unittest.TestCase):
    """Edge case tests for sharded index."""

    def test_single_shard(self):
        """Test with just one shard (degenerates to regular index)."""
        dim = 64
        n_elements = 1000
        np.random.seed(123)
        data = np.float32(np.random.random((n_elements, dim)))
        data = data / np.linalg.norm(data, axis=1, keepdims=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            index = hnswlib.ShardedIndex(space='cosine', dim=dim)
            index.build_shard(data, tmpdir, 0, 0, M=16, ef_construction=100)
            index.save_metadata(tmpdir, 1, n_elements, M=16, ef_construction=100)

            index2 = hnswlib.ShardedIndex(space='cosine', dim=dim)
            index2.load_shards(tmpdir, data, 1)
            index2.set_ef(50)

            labels, _ = index2.knn_query(data, k=1)
            recall = np.mean(labels.flatten() == np.arange(n_elements))
            self.assertGreater(recall, 0.95)

    def test_uneven_shards(self):
        """Test with uneven shard sizes."""
        dim = 64
        np.random.seed(456)

        # Create uneven shards: 1000, 500, 2000
        shard_sizes = [1000, 500, 2000]
        total = sum(shard_sizes)
        data = np.float32(np.random.random((total, dim)))
        data = data / np.linalg.norm(data, axis=1, keepdims=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            index = hnswlib.ShardedIndex(space='cosine', dim=dim)

            offset = 0
            for shard_idx, size in enumerate(shard_sizes):
                shard_data = data[offset:offset + size]
                index.build_shard(shard_data, tmpdir, shard_idx, offset, M=16, ef_construction=100)
                offset += size

            index.save_metadata(tmpdir, len(shard_sizes), total, M=16, ef_construction=100)

            index2 = hnswlib.ShardedIndex(space='cosine', dim=dim)
            index2.load_shards(tmpdir, data, len(shard_sizes))
            index2.set_ef(50)

            # Test vectors from each shard
            test_indices = [0, 500, 1000, 1500, 2500, 3000]
            for idx in test_indices:
                labels, _ = index2.knn_query(data[idx:idx+1], k=1)
                self.assertEqual(labels[0, 0], idx, f"Failed to find vector {idx}")


if __name__ == '__main__':
    unittest.main()
