#!/usr/bin/env python3
"""
Test Product Quantization search with sharded HNSW index.

This test verifies:
1. PQ training and encoding work correctly
2. Sharded index loads with PQ codes
3. Search returns reasonable results
4. Recall is acceptable (should be ~90%+ with good PQ settings)
"""

import numpy as np
import tempfile
import os

import hnswlib


def normalize(vectors):
    """Normalize vectors for cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class SimplePQ:
    """Simple PQ implementation for testing (matches pq.py)."""

    def __init__(self, dim, n_subvectors, n_centroids=256):
        assert dim % n_subvectors == 0
        self.dim = dim
        self.n_subvectors = n_subvectors
        self.n_centroids = n_centroids
        self.subvector_dim = dim // n_subvectors
        self.codebooks = None

    def train(self, vectors, n_iter=10):
        """Train codebooks using k-means."""
        n_samples = vectors.shape[0]
        self.codebooks = np.zeros(
            (self.n_subvectors, self.n_centroids, self.subvector_dim),
            dtype=np.float32
        )

        for m in range(self.n_subvectors):
            start = m * self.subvector_dim
            end = start + self.subvector_dim
            subvectors = vectors[:, start:end].astype(np.float32)

            # K-means initialization
            indices = np.random.choice(n_samples, self.n_centroids, replace=False)
            centroids = subvectors[indices].copy()

            # K-means iterations
            for _ in range(n_iter):
                # Compute distances
                dists = np.sum((subvectors[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
                assignments = np.argmin(dists, axis=1)

                # Update centroids
                for k in range(self.n_centroids):
                    mask = assignments == k
                    if np.sum(mask) > 0:
                        centroids[k] = subvectors[mask].mean(axis=0)

            self.codebooks[m] = centroids

    def encode(self, vectors):
        """Encode vectors to PQ codes."""
        n_vectors = vectors.shape[0]
        codes = np.zeros((n_vectors, self.n_subvectors), dtype=np.uint8)

        for m in range(self.n_subvectors):
            start = m * self.subvector_dim
            end = start + self.subvector_dim
            subvectors = vectors[:, start:end].astype(np.float32)

            # Find nearest centroid
            dists = np.sum((subvectors[:, None, :] - self.codebooks[m][None, :, :]) ** 2, axis=2)
            codes[:, m] = np.argmin(dists, axis=1).astype(np.uint8)

        return codes

    def asymmetric_distance(self, query, codes):
        """Compute asymmetric distances from query to PQ codes."""
        # Precompute distance table
        table = np.zeros((self.n_subvectors, self.n_centroids), dtype=np.float32)
        for m in range(self.n_subvectors):
            start = m * self.subvector_dim
            end = start + self.subvector_dim
            query_sub = query[start:end]
            diff = self.codebooks[m] - query_sub
            table[m] = np.sum(diff ** 2, axis=1)

        # Lookup distances
        n_vectors = codes.shape[0]
        distances = np.zeros(n_vectors, dtype=np.float32)
        for m in range(self.n_subvectors):
            distances += table[m, codes[:, m]]

        return distances


def test_pq_basic():
    """Test basic PQ encode/decode."""
    print("=" * 60)
    print("Test: Basic PQ encode/decode")
    print("=" * 60)

    np.random.seed(42)

    dim = 64
    n_vectors = 1000
    n_subvectors = 8

    # Create random normalized vectors
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    vectors = normalize(vectors)

    # Train PQ
    pq = SimplePQ(dim, n_subvectors, n_centroids=256)
    pq.train(vectors, n_iter=10)

    # Encode
    codes = pq.encode(vectors)

    assert codes.shape == (n_vectors, n_subvectors)
    assert codes.dtype == np.uint8

    # Test asymmetric distance
    query = vectors[0]
    distances = pq.asymmetric_distance(query, codes)

    # Query should be closest to itself
    closest = np.argmin(distances)
    assert closest == 0, f"Expected query to be closest to itself, got idx {closest}"

    print(f"  Vectors: {n_vectors}, dim: {dim}")
    print(f"  Subvectors: {n_subvectors}, codebook size: {pq.codebooks.nbytes / 1024:.1f} KB")
    print(f"  Codes size: {codes.nbytes / 1024:.1f} KB")
    print(f"  Self-distance: {distances[0]:.6f}")
    print("  PASSED!")


def test_pq_recall():
    """Test PQ recall vs brute force."""
    print("\n" + "=" * 60)
    print("Test: PQ recall vs brute force")
    print("=" * 60)

    np.random.seed(42)

    dim = 128
    n_vectors = 5000
    n_queries = 100
    n_subvectors = 16  # 8 dims per subvector
    k = 10

    # Create random normalized vectors
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    vectors = normalize(vectors)

    # Train PQ
    print(f"  Training PQ on {n_vectors} vectors...")
    pq = SimplePQ(dim, n_subvectors, n_centroids=256)
    pq.train(vectors, n_iter=15)

    # Encode
    codes = pq.encode(vectors)

    # Test queries
    query_indices = np.random.choice(n_vectors, n_queries, replace=False)

    recalls = []
    for qi in query_indices:
        query = vectors[qi]

        # Brute force (exact)
        exact_dists = np.sum((vectors - query) ** 2, axis=1)
        exact_top_k = np.argsort(exact_dists)[:k]

        # PQ (approximate)
        pq_dists = pq.asymmetric_distance(query, codes)
        pq_top_k = np.argsort(pq_dists)[:k]

        # Recall
        recall = len(set(exact_top_k) & set(pq_top_k)) / k
        recalls.append(recall)

    mean_recall = np.mean(recalls)
    print(f"  Recall@{k}: {mean_recall * 100:.1f}%")

    # PQ recall depends heavily on data distribution - just check it's reasonable
    # For random vectors with 16 subvectors, recall is typically 30-50%
    assert mean_recall > 0.2, f"Recall too low: {mean_recall}"
    print("  PASSED!")


def test_single_shard_pq():
    """Test single-shard HNSW with PQ to isolate issues."""
    print("\n" + "=" * 60)
    print("Test: Single-shard HNSW with PQ search")
    print("=" * 60)

    np.random.seed(42)

    dim = 16
    n_vectors = 100
    n_subvectors = 4
    n_centroids = 256
    subvector_dim = dim // n_subvectors
    k = 10

    # Create random normalized vectors
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    vectors = normalize(vectors)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Build single shard
        print("  Building single shard...")
        index = hnswlib.ShardedIndex('cosine', dim)
        index.build_shard(vectors, tmpdir, 0, 0, M=16, ef_construction=200)
        index.save_metadata(tmpdir, 1, n_vectors, M=16, ef_construction=200)

        # Create random codebooks and encode
        print("  Creating codebooks and encoding...")
        codebooks = np.random.randn(n_subvectors, n_centroids, subvector_dim).astype(np.float32)
        pq_codes = np.zeros((n_vectors, n_subvectors), dtype=np.uint8)
        for i in range(n_vectors):
            for m in range(n_subvectors):
                start_d = m * subvector_dim
                end_d = start_d + subvector_dim
                subvec = vectors[i, start_d:end_d]
                dists = np.sum((codebooks[m] - subvec) ** 2, axis=1)
                pq_codes[i, m] = np.argmin(dists)

        # Load with PQ
        print("  Loading shard with PQ...")
        index2 = hnswlib.ShardedIndex('cosine', dim)
        index2.load_shards_pq(tmpdir, pq_codes, codebooks, 1, n_subvectors, n_centroids, subvector_dim)
        index2.set_ef(100)

        # Helper function
        def manual_pq_dist(q, code):
            dist = 0.0
            for m in range(n_subvectors):
                start = m * subvector_dim
                end = start + subvector_dim
                q_sub = q[start:end]
                centroid = codebooks[m, code[m]]
                dist += np.sum((q_sub - centroid) ** 2)
            return dist

        # Test search
        print("  Testing search...")
        query = vectors[0:1]
        labels, distances = index2.knn_query(query, k)

        self_dist = manual_pq_dist(vectors[0], pq_codes[0])
        print(f"  Query 0 self-dist: {self_dist:.6f}")
        print(f"  Returned labels: {labels[0]}")
        print(f"  Returned distances: {distances[0]}")

        # Verify all returned distances match manual computation
        all_match = True
        for i, (label, dist) in enumerate(zip(labels[0], distances[0])):
            manual = manual_pq_dist(vectors[0], pq_codes[label])
            if abs(dist - manual) > 0.001:
                print(f"  MISMATCH: label={label}, hnsw={dist:.6f}, manual={manual:.6f}")
                all_match = False
            else:
                print(f"  OK: label={label}, dist={dist:.6f}")

        assert all_match, "Distance computation mismatch!"
        print("  All distances match!")

        # Check if self is found (should be with random codebooks that are not too bad)
        assert 0 in labels[0], f"Query 0 should find itself, got {labels[0]}"
        pos = list(labels[0]).index(0)
        print(f"  Query 0 found itself at position {pos}")
        print("  PASSED!")


def test_sharded_pq_search():
    """Test sharded HNSW index with PQ codes."""
    print("\n" + "=" * 60)
    print("Test: Sharded HNSW with PQ search")
    print("=" * 60)

    np.random.seed(42)

    dim = 16  # Small dim for reliable PQ
    n_vectors = 500
    n_shards = 2
    shard_size = n_vectors // n_shards
    n_subvectors = 4
    n_centroids = 256
    subvector_dim = dim // n_subvectors
    k = 10

    # Create random normalized vectors
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    vectors = normalize(vectors)

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"  Building {n_shards} shards...")

        # Build shards
        index = hnswlib.ShardedIndex('cosine', dim)
        for shard_idx in range(n_shards):
            start = shard_idx * shard_size
            end = start + shard_size
            shard_vectors = vectors[start:end]
            index.build_shard(shard_vectors, tmpdir, shard_idx, start, M=16, ef_construction=200)

        index.save_metadata(tmpdir, n_shards, n_vectors, M=16, ef_construction=200)

        # Use random codebooks (simple but effective for testing)
        # In production, these would be trained with k-means
        print(f"  Creating codebooks ({n_subvectors} subvectors)...")
        codebooks = np.random.randn(n_subvectors, n_centroids, subvector_dim).astype(np.float32)

        # Encode: for each vector, find nearest centroid in each subspace
        print("  Encoding vectors to PQ codes...")
        pq_codes = np.zeros((n_vectors, n_subvectors), dtype=np.uint8)
        for i in range(n_vectors):
            for m in range(n_subvectors):
                start_d = m * subvector_dim
                end_d = start_d + subvector_dim
                subvec = vectors[i, start_d:end_d]
                dists = np.sum((codebooks[m] - subvec) ** 2, axis=1)
                pq_codes[i, m] = np.argmin(dists)

        # Load shards with PQ
        print("  Loading shards with PQ codes...")
        index2 = hnswlib.ShardedIndex('cosine', dim)
        index2.load_shards_pq(
            tmpdir,
            pq_codes,
            codebooks,
            n_shards,
            n_subvectors,
            n_centroids,
            subvector_dim
        )
        index2.set_ef(100)

        # Test search - focus on verifying the implementation works correctly
        print("  Testing search...")

        # Helper function for manual PQ distance
        def manual_pq_dist(q, code):
            dist = 0.0
            for m in range(n_subvectors):
                start = m * subvector_dim
                end = start + subvector_dim
                q_sub = q[start:end]
                centroid = codebooks[m, code[m]]
                dist += np.sum((q_sub - centroid) ** 2)
            return dist

        # Test 1: Query vector 0 (shard 0), should find itself
        query = vectors[0:1]
        labels, distances = index2.knn_query(query, k)

        self_pq_dist = manual_pq_dist(vectors[0], pq_codes[0])
        print(f"  Query 0 PQ dist to self: {self_pq_dist:.6f}")
        print(f"  Returned labels: {labels[0]}")
        print(f"  Returned distances: {distances[0]}")

        # Verify distances match manual computation (first 3)
        for i, (label, dist) in enumerate(zip(labels[0][:3], distances[0][:3])):
            manual_dist = manual_pq_dist(vectors[0], pq_codes[label])
            assert abs(dist - manual_dist) < 0.001, f"Distance mismatch for label {label}"
            print(f"    {i}: label={label}, hnsw_dist={dist:.6f}, manual_dist={manual_dist:.6f}")

        assert 0 in labels[0], f"Query 0 should find itself, got {labels[0]}"
        print(f"  Query 0 found itself at position {list(labels[0]).index(0)}")

        # Test 2: Query vector from second shard
        query_idx = shard_size + 10
        query = vectors[query_idx:query_idx+1]
        labels, distances = index2.knn_query(query, k)

        self_pq_dist = manual_pq_dist(vectors[query_idx], pq_codes[query_idx])
        print(f"  Query {query_idx} PQ dist to self: {self_pq_dist:.6f}")
        print(f"  Returned labels: {labels[0]}")
        print(f"  Returned distances: {distances[0]}")

        # Verify ALL returned distances match manual computation
        for i, (label, dist) in enumerate(zip(labels[0], distances[0])):
            manual_dist = manual_pq_dist(vectors[query_idx], pq_codes[label])
            assert abs(dist - manual_dist) < 0.001, f"Distance mismatch for label {label}: hnsw={dist}, manual={manual_dist}"
            print(f"    {i}: label={label}, hnsw_dist={dist:.6f}, manual_dist={manual_dist:.6f}")

        assert query_idx in labels[0], f"Query {query_idx} should find itself, got {labels[0]}"
        print(f"  Query {query_idx} (shard 2) results: top label is {labels[0][0]}")

        # Test 3: Verify distances are computed correctly
        query = vectors[0:1]
        labels, distances = index2.knn_query(query, k)
        manual_dist = manual_pq_dist(vectors[0], pq_codes[0])
        hnsw_dist = distances[0][list(labels[0]).index(0)] if 0 in labels[0] else None

        print(f"  Manual PQ dist to self: {manual_dist:.6f}")
        if hnsw_dist is not None:
            print(f"  HNSW PQ dist to self: {hnsw_dist:.6f}")
            # Allow some floating point tolerance
            assert abs(manual_dist - hnsw_dist) < 0.01, \
                f"Distance mismatch: manual={manual_dist}, hnsw={hnsw_dist}"
            print("  Distance computation matches!")

        print("  PASSED!")


def test_pq_with_mmap():
    """Test PQ search with mmap'd codes (simulates production usage)."""
    print("\n" + "=" * 60)
    print("Test: PQ search with mmap'd codes")
    print("=" * 60)

    np.random.seed(123)  # Different seed from other tests

    dim = 16
    n_vectors = 200
    n_shards = 2
    shard_size = n_vectors // n_shards
    n_subvectors = 4
    n_centroids = 256
    subvector_dim = dim // n_subvectors

    # Create random normalized vectors
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    vectors = normalize(vectors)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Build shards
        index = hnswlib.ShardedIndex('cosine', dim)
        for shard_idx in range(n_shards):
            start = shard_idx * shard_size
            end = start + shard_size
            index.build_shard(vectors[start:end], tmpdir, shard_idx, start, M=16, ef_construction=200)
        index.save_metadata(tmpdir, n_shards, n_vectors, M=16, ef_construction=200)

        # Create random codebooks and encode
        codebooks = np.random.randn(n_subvectors, n_centroids, subvector_dim).astype(np.float32)
        pq_codes = np.zeros((n_vectors, n_subvectors), dtype=np.uint8)
        for i in range(n_vectors):
            for m in range(n_subvectors):
                start_d = m * subvector_dim
                end_d = start_d + subvector_dim
                subvec = vectors[i, start_d:end_d]
                dists = np.sum((codebooks[m] - subvec) ** 2, axis=1)
                pq_codes[i, m] = np.argmin(dists)

        # Save to mmap file
        codes_path = os.path.join(tmpdir, 'pq_codes.mmap')
        codes_mmap = np.memmap(codes_path, dtype=np.uint8, mode='w+',
                               shape=(n_vectors, n_subvectors))
        codes_mmap[:] = pq_codes
        codes_mmap.flush()
        del codes_mmap

        # Save codebooks
        codebooks_path = os.path.join(tmpdir, 'codebooks.npz')
        np.savez(codebooks_path, codebooks=codebooks)

        # Reload from mmap (simulates production)
        print("  Loading PQ codes from mmap...")
        codes_mmap = np.memmap(codes_path, dtype=np.uint8, mode='r',
                               shape=(n_vectors, n_subvectors))
        cb_data = np.load(codebooks_path)
        loaded_codebooks = cb_data['codebooks']

        # Load index with mmap'd PQ codes
        index2 = hnswlib.ShardedIndex('cosine', dim)
        index2.load_shards_pq(
            tmpdir, codes_mmap, loaded_codebooks, n_shards,
            n_subvectors, n_centroids, subvector_dim
        )
        index2.set_ef(100)

        # Quick search test
        query = vectors[0:1]
        labels, distances = index2.knn_query(query, k=5)

        assert 0 in labels[0], f"Query should find itself, got {labels[0]}"
        print(f"  Query found itself at position {list(labels[0]).index(0)}")
        print("  PASSED!")


if __name__ == '__main__':
    test_pq_basic()
    test_pq_recall()
    test_single_shard_pq()
    test_sharded_pq_search()
    test_pq_with_mmap()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
