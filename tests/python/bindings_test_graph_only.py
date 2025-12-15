import os
import tempfile
import unittest

import numpy as np

import hnswlib


class GraphOnlyTestCase(unittest.TestCase):
    """Tests for graph-only / external vectors mode."""

    def setUp(self):
        self.dim = 128
        self.num_elements = 5000
        np.random.seed(42)
        self.data = np.float32(np.random.random((self.num_elements, self.dim)))
        # Normalize for cosine
        self.data = self.data / np.linalg.norm(self.data, axis=1, keepdims=True)

    def test_init_graph_only(self):
        """Test that graph-only index can be created and searched."""
        index = hnswlib.Index(space='cosine', dim=self.dim)
        index.init_index_graph_only(
            self.num_elements, self.data, M=16, ef_construction=100
        )
        index.add_items(self.data)

        self.assertEqual(index.get_current_count(), self.num_elements)

        # Search should work
        index.set_ef(50)
        labels, distances = index.knn_query(self.data[:10], k=1)

        # Each vector should find itself
        expected = np.arange(10).reshape(-1, 1)
        np.testing.assert_array_equal(labels, expected)

    def test_save_load_graph(self):
        """Test saving and loading graph-only index."""
        # Build index
        index = hnswlib.Index(space='cosine', dim=self.dim)
        index.init_index_graph_only(
            self.num_elements, self.data, M=16, ef_construction=100
        )
        index.add_items(self.data)
        index.set_ef(50)

        # Get results before save
        labels_before, distances_before = index.knn_query(self.data[:100], k=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = os.path.join(tmpdir, 'test.graph')

            # Save graph
            index.save_graph(graph_path)
            self.assertTrue(os.path.exists(graph_path))

            # Graph file should be much smaller than full index
            graph_size = os.path.getsize(graph_path)
            # Rough estimate: ~76 bytes per element for M=16
            expected_max = self.num_elements * 200  # generous upper bound
            self.assertLess(graph_size, expected_max)

            # Load into new index
            index2 = hnswlib.Index(space='cosine', dim=self.dim)
            index2.load_graph(graph_path)

            self.assertEqual(index2.get_current_count(), self.num_elements)

            # Set external vectors
            index2.set_external_vectors(self.data)
            index2.set_ef(50)

            # Results should match
            labels_after, distances_after = index2.knn_query(self.data[:100], k=5)
            np.testing.assert_array_equal(labels_before, labels_after)
            np.testing.assert_array_almost_equal(distances_before, distances_after)

    def test_graph_size_reduction(self):
        """Test that graph-only saves significant space compared to full index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Build standard index
            index_std = hnswlib.Index(space='cosine', dim=self.dim)
            index_std.init_index(self.num_elements, M=16, ef_construction=100)
            index_std.add_items(self.data)

            std_path = os.path.join(tmpdir, 'standard.bin')
            index_std.save_index(std_path)
            std_size = os.path.getsize(std_path)

            # Build graph-only index
            index_graph = hnswlib.Index(space='cosine', dim=self.dim)
            index_graph.init_index_graph_only(
                self.num_elements, self.data, M=16, ef_construction=100
            )
            index_graph.add_items(self.data)

            graph_path = os.path.join(tmpdir, 'graph.bin')
            index_graph.save_graph(graph_path)
            graph_size = os.path.getsize(graph_path)

            # Graph should be significantly smaller (at least 70% reduction)
            # For dim=128: ~588 bytes/element standard vs ~140 bytes/element graph
            reduction = 1 - (graph_size / std_size)
            self.assertGreater(reduction, 0.70)

    def test_recall_graph_only(self):
        """Test that graph-only mode achieves good recall."""
        # Build graph-only index
        index = hnswlib.Index(space='cosine', dim=self.dim)
        index.init_index_graph_only(
            self.num_elements, self.data, M=16, ef_construction=200
        )
        index.add_items(self.data)
        index.set_ef(100)

        # Query vectors for themselves
        labels, _ = index.knn_query(self.data, k=1)

        # Recall should be high (each vector finding itself)
        recall = np.mean(labels.flatten() == np.arange(self.num_elements))
        self.assertGreater(recall, 0.95)

    def test_l2_space(self):
        """Test graph-only mode works with l2 distance."""
        # Use non-normalized data for l2
        data_l2 = np.float32(np.random.random((self.num_elements, self.dim)))

        index = hnswlib.Index(space='l2', dim=self.dim)
        index.init_index_graph_only(
            self.num_elements, data_l2, M=16, ef_construction=200
        )
        index.add_items(data_l2)
        index.set_ef(100)

        # Query vectors for themselves
        labels, _ = index.knn_query(data_l2, k=1)
        recall = np.mean(labels.flatten() == np.arange(self.num_elements))
        self.assertGreater(recall, 0.95)

    def test_fp16_external_vectors(self):
        """Test that fp16 external vectors work with on-the-fly conversion."""
        # Build index with fp32
        index = hnswlib.Index(space='cosine', dim=self.dim)
        index.init_index_graph_only(
            self.num_elements, self.data, M=16, ef_construction=200
        )
        index.add_items(self.data)
        index.set_ef(100)

        # Convert to fp16 and set as external vectors
        data_fp16 = self.data.astype(np.float16)
        index.set_external_vectors_fp16(data_fp16.view(np.uint16))

        # Query all vectors for themselves
        labels, distances = index.knn_query(self.data, k=1)

        # High recall expected
        recall = np.mean(labels.flatten() == np.arange(self.num_elements))
        self.assertGreater(recall, 0.90)

    def test_fp16_save_load_graph(self):
        """Test save/load graph with fp16 vectors at search time."""
        # Build with fp32
        index = hnswlib.Index(space='cosine', dim=self.dim)
        index.init_index_graph_only(
            self.num_elements, self.data, M=16, ef_construction=200
        )
        index.add_items(self.data)
        index.set_ef(100)

        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = os.path.join(tmpdir, 'test.graph')
            index.save_graph(graph_path)

            # Load and use fp16 vectors
            index2 = hnswlib.Index(space='cosine', dim=self.dim)
            index2.load_graph(graph_path)

            # Set fp16 vectors
            data_fp16 = self.data.astype(np.float16)
            index2.set_external_vectors_fp16(data_fp16.view(np.uint16))
            index2.set_ef(100)

            # Query all vectors for themselves
            labels, _ = index2.knn_query(self.data, k=1)
            recall = np.mean(labels.flatten() == np.arange(self.num_elements))
            self.assertGreater(recall, 0.90)


class MmapStorageTestCase(unittest.TestCase):
    """Tests for mmap-backed storage mode."""

    def setUp(self):
        self.dim = 64
        self.num_elements = 1000
        np.random.seed(42)
        self.data = np.float32(np.random.random((self.num_elements, self.dim)))

    def test_init_mmap(self):
        """Test that mmap-backed index can be created and searched."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mmap_path = os.path.join(tmpdir, 'index.mmap')

            index = hnswlib.Index(space='l2', dim=self.dim)
            index.init_index_mmap(
                self.num_elements, mmap_path, M=16, ef_construction=100
            )
            index.add_items(self.data)

            self.assertEqual(index.get_current_count(), self.num_elements)

            # Mmap file should exist
            self.assertTrue(os.path.exists(mmap_path))

            # Search should work
            index.set_ef(50)
            labels, distances = index.knn_query(self.data[:10], k=1)
            expected = np.arange(10).reshape(-1, 1)
            np.testing.assert_array_equal(labels, expected)

    def test_mmap_save_load(self):
        """Test saving index built with mmap and loading normally."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mmap_path = os.path.join(tmpdir, 'scratch.mmap')
            index_path = os.path.join(tmpdir, 'index.bin')

            # Build with mmap
            index = hnswlib.Index(space='l2', dim=self.dim)
            index.init_index_mmap(
                self.num_elements, mmap_path, M=16, ef_construction=100
            )
            index.add_items(self.data)
            index.set_ef(50)

            labels_before, _ = index.knn_query(self.data[:100], k=5)

            # Save normally
            index.save_index(index_path)
            del index

            # Load normally (no mmap)
            index2 = hnswlib.Index(space='l2', dim=self.dim)
            index2.load_index(index_path)
            index2.set_ef(50)

            labels_after, _ = index2.knn_query(self.data[:100], k=5)

            np.testing.assert_array_equal(labels_before, labels_after)


if __name__ == '__main__':
    unittest.main()
