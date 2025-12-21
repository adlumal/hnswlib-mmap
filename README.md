# hnswlib-mmap

A fork of [hnswlib](https://github.com/nmslib/hnswlib) with memory-mapped storage and graph-only index support for large-scale vector search.

This fork adds five features for working with indexes that exceed available RAM:

1. **Graph-only mode**: Store only the graph structure (~95% smaller), with vectors provided externally
2. **FP16 external vectors**: Use half-precision vectors with on-the-fly conversion
3. **Sharded indexes**: Build indexes larger than RAM by splitting into independently-built shards
4. **Product Quantization (PQ)**: Compress vectors to ~8x smaller with ~94% recall for extreme scale
5. **IVF-HNSW**: Cluster-based shard routing to search only relevant shards, reducing I/O by 8-16x

These features enable building and searching HNSW indexes with hundreds of millions of vectors on memory-constrained systems.

---

*Based on hnswlib v0.8.0. See the [original repository](https://github.com/nmslib/hnswlib) for the full feature set and documentation.*

---

## Fork-specific features

### Graph-only mode

Standard hnswlib stores vectors alongside the graph structure. For large indexes, this dominates memory usage:

| Mode | Storage per element (M=8, dim=384) | 100M vectors |
|------|-----------------------------------|--------------|
| Standard | ~1,612 bytes | ~161 GB |
| Graph-only | ~76 bytes | ~8 GB |

Graph-only mode stores only the HNSW graph. Vectors are provided via a pointer to an external array (e.g., a memory-mapped numpy file).

```python
import hnswlib
import numpy as np

dim = 384
n = 1_000_000

# Your vectors (could be mmap'd from disk)
vectors = np.random.randn(n, dim).astype('float32')

# Build with external vectors
index = hnswlib.Index(space='cosine', dim=dim)
index.init_index_graph_only(n, vectors, M=8, ef_construction=200)
index.add_items(vectors)

# Save graph only (~24x smaller than save_index)
index.save_graph('index.graph')

# Later: load graph and point to vectors
index2 = hnswlib.Index(space='cosine', dim=dim)
index2.load_graph('index.graph')
index2.set_external_vectors(vectors)
labels, distances = index2.knn_query(query, k=10)
```

### FP16 external vectors

When your vectors are stored as float16 (e.g., to save disk space), you can use them directly without converting the entire array to float32:

```python
# Load fp16 vectors from disk (half the size of fp32)
vectors_fp16 = np.memmap('vectors.mmap', dtype=np.float16, mode='r',
                         shape=(n, dim))

# Load graph and set fp16 vectors
index = hnswlib.Index(space='cosine', dim=dim)
index.load_graph('index.graph')
index.set_external_vectors_fp16(vectors_fp16.view(np.uint16))

# Query with fp32 query vectors
query = np.random.randn(1, dim).astype('float32')
labels, distances = index.knn_query(query, k=10)
```

Vectors are converted to fp32 on-the-fly during distance calculations, using thread-local buffers. This adds minimal overhead while halving memory requirements.

### Sharded indexes

For truly massive datasets (hundreds of millions of vectors), even graph-only mode requires more RAM than available during construction. Sharded indexes solve this by building multiple smaller indexes independently:

| Approach | RAM during build |
|----------|------------------|
| Full index | Proportional to dataset size |
| Graph-only | Graph + vectors must fit |
| **Sharded** | **~15 GB per shard** |

```python
import hnswlib
import numpy as np

dim = 1536
n_total = 100_000_000
n_shards = 32
shard_size = (n_total + n_shards - 1) // n_shards  # ~10M per shard

# Build shards one at a time (only ~15GB RAM needed)
index = hnswlib.ShardedIndex(space='cosine', dim=dim)

for shard_idx in range(n_shards):
    start = shard_idx * shard_size
    end = min(start + shard_size, n_total)

    # Load only this shard's vectors into RAM
    shard_vectors = load_vectors(start, end).astype('float32')

    index.build_shard(shard_vectors, 'index_shards/', shard_idx, start,
                      M=8, ef_construction=200)
    del shard_vectors  # Free RAM before next shard

index.save_metadata('index_shards/', n_shards, n_total, M=8, ef_construction=200)
```

At search time, load all shards with mmap'd vectors:

```python
# Load vectors as mmap (no RAM for vectors!)
vectors_fp16 = np.memmap('vectors.mmap', dtype=np.float16, mode='r',
                          shape=(n_total, dim))

index = hnswlib.ShardedIndex(space='cosine', dim=dim)
index.load_shards_fp16('index_shards/', vectors_fp16.view(np.uint16), n_shards)
index.set_ef(50)

# Search returns global indices
labels, distances = index.knn_query(query, k=10)
```

The sharded search queries all shards and merges results. Trade-off: slightly higher latency than monolithic index, but enables building on normal hardware.

### Memory-mapped graph loading

For extremely memory-constrained environments, you can also memory-map the graph files themselves instead of loading them into RAM:

```python
# Load with mmap'd graphs - uses ~4GB instead of ~28GB for 64 shards
index = hnswlib.ShardedIndex(space='cosine', dim=dim)
index.load_shards_fp16('index_shards/', vectors_fp16.view(np.uint16), n_shards,
                       use_mmap_graphs=True)
```

Trade-off: Mmap'd graphs have higher search latency due to disk I/O during graph traversal, and actual memory usage depends on page cache behavior. Best used when:
- You have limited RAM but fast SSD storage
- You need to load more shards than RAM allows
- You're doing infrequent queries where latency is less critical

### Mmap-backed storage

For building indexes from scratch when even graph construction exceeds RAM, use mmap-backed storage:

```python
index = hnswlib.Index(space='cosine', dim=dim)
index.init_index_mmap(n, '/path/to/scratch.mmap', M=8, ef_construction=200)
index.add_items(vectors)
index.save_index('index.bin')  # or save_graph for graph-only
```

This creates a file-backed memory map instead of allocating heap memory.

### Product Quantization (PQ)

For the largest datasets where even FP16 vectors don't fit, Product Quantization compresses vectors to just M bytes each (where M is the number of subvectors). This enables searching datasets that would otherwise require terabytes of storage:

| Format | Storage (100M vectors, dim=384) | Recall@10 |
|--------|--------------------------------|-----------|
| FP32 | 153 GB | 100% |
| FP16 | 77 GB | ~100% |
| **PQ (M=192)** | **18 GB** | **~94%** |

PQ works by splitting each vector into M subvectors and quantizing each subspace independently using 256 centroids. At search time, distances are computed using precomputed lookup tables - just M table lookups instead of dim multiplications.

```python
import numpy as np
import hnswlib

dim = 384
n_total = 100_000_000
n_shards = 64
n_subvectors = 192  # 2 dimensions per subvector
n_centroids = 256

# Build shards first (same as before)
index = hnswlib.ShardedIndex(space='cosine', dim=dim)
# ... build shards ...

# Train PQ codebooks on a sample of your data (use k-means or similar)
# codebooks shape: (n_subvectors, n_centroids, subvector_dim)
codebooks = train_pq_codebooks(sample_vectors, n_subvectors, n_centroids)

# Encode all vectors to PQ codes
# pq_codes shape: (n_total, n_subvectors), dtype=uint8
pq_codes = encode_vectors_pq(all_vectors, codebooks)

# Save PQ data
np.savez('pq_codebooks.npz', codebooks=codebooks)
pq_mmap = np.memmap('pq_codes.mmap', dtype=np.uint8, mode='w+',
                    shape=(n_total, n_subvectors))
pq_mmap[:] = pq_codes
pq_mmap.flush()
```

At search time, load with PQ codes instead of vectors:

```python
# Load PQ data (mmap for minimal RAM)
pq_codes = np.memmap('pq_codes.mmap', dtype=np.uint8, mode='r',
                     shape=(n_total, n_subvectors))
codebooks = np.load('pq_codebooks.npz')['codebooks']

# Load shards with PQ
index = hnswlib.ShardedIndex(space='cosine', dim=dim)
index.load_shards_pq(
    'index_shards/',
    pq_codes,
    codebooks,
    n_shards=64,
    n_subvectors=192,
    n_centroids=256,
    subvector_dim=2  # dim / n_subvectors
)
index.set_ef(100)

# Search - query must be normalized fp32
query = normalize(query_vector.astype(np.float32))
labels, distances = index.knn_query(query, k=10)
```

**Hybrid loading** - when you have enough RAM for some shards but not all:

```python
# Load first 32 shards into RAM, mmap the remaining 32
index.load_shards_pq(
    'index_shards/',
    pq_codes,
    codebooks,
    n_shards=64,
    n_subvectors=192,
    n_centroids=256,
    subvector_dim=2,
    n_ram_shards=32  # First 32 in RAM, rest mmap'd
)
```

PQ is ideal when:
- Your dataset is too large for FP16 (hundreds of millions of vectors)
- You can accept ~94% recall instead of ~100%
- You have compute to train codebooks offline

### IVF-HNSW (cluster-based shard routing)

IVF-HNSW combines IVF (Inverted File Index) routing with HNSW search. Instead of searching all shards, queries are routed to only the most relevant clusters based on centroid similarity.

**IVF vs IVF-HNSW:**

| Approach | Cluster routing | Intra-cluster search | Use case |
|----------|----------------|---------------------|----------|
| Pure IVF | k-means centroids | Exhaustive scan | Small clusters |
| **IVF-HNSW** | k-means centroids | HNSW graph search | Large clusters |

Pure IVF does brute-force search within selected clusters. With large datasets split across 64 clusters, each cluster can have millions of vectors - exhaustive scan would be too slow. IVF-HNSW uses HNSW for O(log n) intra-cluster search instead.

**I/O reduction:**

| Search mode | Shards searched | Relative I/O |
|-------------|-----------------|--------------|
| Full search | 64 | 100% |
| IVF-HNSW (n_probe=8) | 8 | 12.5% |
| IVF-HNSW (n_probe=4) | 4 | 6.25% |

```python
import numpy as np
import hnswlib

# Load index (same as before)
index = hnswlib.ShardedIndex('cosine', dim=384)
index.load_shards_pq('index_dir/', pq_codes, codebooks, n_shards=64, ...)
index.set_ef(50)

# Load cluster centroids (one per shard, trained with k-means)
centroids = np.load('centroids.npy')  # shape: (64, 384)

# For each query, find nearest clusters via centroid comparison
def find_nearest_clusters(query, centroids, n_probe):
    query_norm = query / np.linalg.norm(query)
    similarities = query_norm @ centroids.T
    return np.argsort(-similarities)[:n_probe]

# Search only the relevant shards
query = get_query_vector()  # shape: (384,)
nearest_clusters = find_nearest_clusters(query, centroids, n_probe=4)

# IVF-HNSW search - only touches 4 shards instead of 64
labels, distances = index.knn_query_selective(
    query.reshape(1, -1),
    np.array(nearest_clusters, dtype=np.int64),
    k=10
)
```

**When to use IVF-HNSW:**
- Large clusters where exhaustive search would be slow (>100K vectors per cluster)
- Shards organized by semantic similarity (cluster-based, not sequential)
- Cold-start latency is a concern (fewer shards = less data to page in)
- You can accept slightly lower recall for much faster search

**Building cluster-based shards:**

1. Train k-means on a sample of vectors to get cluster centroids
2. Assign all vectors to their nearest centroid
3. Reorder vectors by cluster assignment
4. Build one HNSW shard per cluster

The key insight is that semantically similar vectors end up in the same shard, so searching only nearby clusters maintains high recall while dramatically reducing I/O.

---

## Original hnswlib features

### Highlights:
1) Lightweight, header-only, no dependencies other than C++ 11
2) Interfaces for C++, Python, external support for Java and R (https://github.com/jlmelville/rcpphnsw).
3) Has full support for incremental index construction and updating the elements (thanks to the contribution by Apoorv Sharma). Has support for element deletions 
(by marking them in index, later can be replaced with other elements). Python index is picklable.
4) Can work with custom user defined distances (C++).
5) Significantly less memory footprint and faster build time compared to current nmslib's implementation.

Description of the algorithm parameters can be found in [ALGO_PARAMS.md](ALGO_PARAMS.md).


### Python bindings

#### Supported distances:

| Distance         | parameter       | Equation                |
| -------------    |:---------------:| -----------------------:|
|Squared L2        |'l2'             | d = sum((Ai-Bi)^2)      |
|Inner product     |'ip'             | d = 1.0 - sum(Ai\*Bi)   |
|Cosine similarity |'cosine'         | d = 1.0 - sum(Ai\*Bi) / sqrt(sum(Ai\*Ai) * sum(Bi\*Bi))|

Note that inner product is not an actual metric. An element can be closer to some other element than to itself. That allows some speedup if you remove all elements that are not the closest to themselves from the index.

For other spaces use the nmslib library https://github.com/nmslib/nmslib. 

#### API description
* `hnswlib.Index(space, dim)` creates a non-initialized index an HNSW in space `space` with integer dimension `dim`.

`hnswlib.Index` methods:
* `init_index(max_elements, M = 16, ef_construction = 200, random_seed = 100, allow_replace_deleted = False)` initializes the index from with no elements. 
    * `max_elements` defines the maximum number of elements that can be stored in the structure(can be increased/shrunk).
    * `ef_construction` defines a construction time/accuracy trade-off (see [ALGO_PARAMS.md](ALGO_PARAMS.md)).
    * `M` defines tha maximum number of outgoing connections in the graph ([ALGO_PARAMS.md](ALGO_PARAMS.md)).
    * `allow_replace_deleted` enables replacing of deleted elements with new added ones.
    
* `add_items(data, ids, num_threads = -1, replace_deleted = False)` - inserts the `data`(numpy array of vectors, shape:`N*dim`) into the structure. 
    * `num_threads` sets the number of cpu threads to use (-1 means use default).
    * `ids` are optional N-size numpy array of integer labels for all elements in `data`. 
      - If index already has the elements with the same labels, their features will be updated. Note that update procedure is slower than insertion of a new element, but more memory- and query-efficient.
    * `replace_deleted` replaces deleted elements. Note it allows to save memory.
      - to use it `init_index` should be called with `allow_replace_deleted=True`
    * Thread-safe with other `add_items` calls, but not with `knn_query`.
    
* `mark_deleted(label)`  - marks the element as deleted, so it will be omitted from search results. Throws an exception if it is already deleted.

* `unmark_deleted(label)`  - unmarks the element as deleted, so it will be not be omitted from search results.

* `resize_index(new_size)` - changes the maximum capacity of the index. Not thread safe with `add_items` and `knn_query`.

* `set_ef(ef)` - sets the query time accuracy/speed trade-off, defined by the `ef` parameter (
[ALGO_PARAMS.md](ALGO_PARAMS.md)). Note that the parameter is currently not saved along with the index, so you need to set it manually after loading.

* `knn_query(data, k = 1, num_threads = -1, filter = None)` make a batch query for `k` closest elements for each element of the 
    * `data` (shape:`N*dim`). Returns a numpy array of (shape:`N*k`).
    * `num_threads` sets the number of cpu threads to use (-1 means use default).
    * `filter` filters elements by its labels, returns elements with allowed ids. Note that search with a filter works slow in python in multithreaded mode. It is recommended to set `num_threads=1`
    * Thread-safe with other `knn_query` calls, but not with `add_items`.
    
* `load_index(path_to_index, max_elements = 0, allow_replace_deleted = False)` loads the index from persistence to the uninitialized index.
    * `max_elements`(optional) resets the maximum number of elements in the structure.
    * `allow_replace_deleted` specifies whether the index being loaded has enabled replacing of deleted elements.
      
* `save_index(path_to_index)` saves the index from persistence.

**Fork-specific methods (mmap and graph-only):**

* `init_index_mmap(max_elements, mmap_path, M = 16, ef_construction = 200, random_seed = 100, allow_replace_deleted = False)` initializes the index using a memory-mapped file instead of heap allocation. Useful when building large indexes that exceed available RAM.

* `init_index_graph_only(max_elements, external_vectors, M = 16, ef_construction = 200, random_seed = 100)` initializes a graph-only index where vectors are stored externally. The `external_vectors` numpy array must remain valid for the lifetime of the index.

* `save_graph(path_to_graph)` saves only the graph structure (links + labels), not vectors. Results in much smaller files.

* `load_graph(path_to_graph)` loads a graph-only index. Must call `set_external_vectors()` afterwards before searching.

* `set_external_vectors(external_vectors)` sets the external vector source (fp32) for a graph-only index.

* `set_external_vectors_fp16(external_vectors)` sets the external vector source in fp16 format. Vectors are converted to fp32 on-the-fly during distance calculations. Pass `array.view(np.uint16)` to this method.

* `load_index_mmap(path_to_index, read_only = True)` loads an existing index using mmap instead of reading into RAM.

**ShardedIndex class** (`hnswlib.ShardedIndex`):

* `ShardedIndex(space, dim)` creates a sharded index for building/searching across multiple shards.

* `build_shard(vectors, output_dir, shard_idx, start_idx, M = 8, ef_construction = 200, num_threads = -1)` builds a single shard. Call once per shard with only that shard's vectors loaded.

* `save_metadata(output_dir, n_shards, total_elements, M, ef_construction)` saves global index metadata after all shards are built.

* `load_shards(output_dir, external_vectors, n_shards)` loads all shards for search with fp32 vectors.

* `load_shards_fp16(output_dir, external_vectors, n_shards, use_mmap_graphs=False)` loads all shards with fp16 vectors (pass `array.view(np.uint16)`). Set `use_mmap_graphs=True` to memory-map graph files instead of loading into RAM.

* `load_shards_pq(output_dir, pq_codes, codebooks, n_shards, n_subvectors, n_centroids, subvector_dim, use_mmap_graphs=False, n_ram_shards=0)` loads all shards with Product Quantization codes for compressed vector search.
    * `pq_codes`: numpy array of shape `(n_vectors, n_subvectors)`, dtype=uint8. Can be memory-mapped.
    * `codebooks`: numpy array of shape `(n_subvectors, n_centroids, subvector_dim)`, dtype=float32.
    * `n_subvectors`: number of subvectors (M). Must satisfy `dim = n_subvectors * subvector_dim`.
    * `n_centroids`: centroids per subspace (typically 256).
    * `subvector_dim`: dimension of each subspace (`dim / n_subvectors`).
    * `use_mmap_graphs`: if True, memory-map graph files instead of loading into RAM.
    * `n_ram_shards`: hybrid mode - load first N shards into RAM, mmap the rest (0=disabled). Useful when you have enough RAM for some shards but not all.

* `knn_query(data, k = 1, num_threads = -1)` searches across all shards. Returns global indices.

* `knn_query_selective(data, shard_ids, k = 1, num_threads = -1)` searches only the specified shards. Use for IVF-HNSW where you first find nearest cluster centroids, then search only those clusters.
    * `shard_ids`: numpy array of shard indices to search, dtype=int64.
    * Returns global indices (same as `knn_query`).
    * See the IVF-HNSW section above for usage example.

* `set_ef(ef)` sets the ef search parameter for all shards.

* `get_num_shards()` returns the number of loaded shards.

* `get_total_elements()` returns total elements across all shards.

* `set_num_threads(num_threads)` set the default number of cpu threads used during data insertion/querying.
  
* `get_items(ids, return_type = 'numpy')` - returns a numpy array (shape:`N*dim`) of vectors that have integer identifiers specified in `ids` numpy vector (shape:`N`) if `return_type` is `list` return list of lists. Note that for cosine similarity it currently returns **normalized** vectors.
  
* `get_ids_list()`  - returns a list of all elements' ids.

* `get_max_elements()` - returns the current capacity of the index

* `get_current_count()` - returns the current number of element stored in the index

Read-only properties of `hnswlib.Index` class:

* `space` - name of the space (can be one of "l2", "ip", or "cosine"). 

* `dim`   - dimensionality of the space. 

* `M` - parameter that defines the maximum number of outgoing connections in the graph. 

* `ef_construction` - parameter that controls speed/accuracy trade-off during the index construction. 

* `max_elements` - current capacity of the index. Equivalent to `p.get_max_elements()`. 

* `element_count` - number of items in the index. Equivalent to `p.get_current_count()`. 

Properties of `hnswlib.Index` that support reading and writing:

* `ef` - parameter controlling query time/accuracy trade-off.

* `num_threads` - default number of threads to use in `add_items` or `knn_query`. Note that calling `p.set_num_threads(3)` is equivalent to `p.num_threads=3`.

  
        
  
#### Python bindings examples
[See more examples here](examples/python/EXAMPLES.md):
* Creating index, inserting elements, searching, serialization/deserialization
* Filtering during the search with a boolean function
* Deleting the elements and reusing the memory of the deleted elements for newly added elements

An example of creating index, inserting elements, searching and pickle serialization:
```python
import hnswlib
import numpy as np
import pickle

dim = 128
num_elements = 10000

# Generating sample data
data = np.float32(np.random.random((num_elements, dim)))
ids = np.arange(num_elements)

# Declaring index
p = hnswlib.Index(space = 'l2', dim = dim) # possible options are l2, cosine or ip

# Initializing index - the maximum number of elements should be known beforehand
p.init_index(max_elements = num_elements, ef_construction = 200, M = 16)

# Element insertion (can be called several times):
p.add_items(data, ids)

# Controlling the recall by setting ef:
p.set_ef(50) # ef should always be > k

# Query dataset, k - number of the closest elements (returns 2 numpy arrays)
labels, distances = p.knn_query(data, k = 1)

# Index objects support pickling
# WARNING: serialization via pickle.dumps(p) or p.__getstate__() is NOT thread-safe with p.add_items method!
# Note: ef parameter is included in serialization; random number generator is initialized with random_seed on Index load
p_copy = pickle.loads(pickle.dumps(p)) # creates a copy of index p using pickle round-trip

### Index parameters are exposed as class properties:
print(f"Parameters passed to constructor:  space={p_copy.space}, dim={p_copy.dim}") 
print(f"Index construction: M={p_copy.M}, ef_construction={p_copy.ef_construction}")
print(f"Index size is {p_copy.element_count} and index capacity is {p_copy.max_elements}")
print(f"Search speed/quality trade-off parameter: ef={p_copy.ef}")
```

An example with updates after serialization/deserialization:
```python
import hnswlib
import numpy as np

dim = 16
num_elements = 10000

# Generating sample data
data = np.float32(np.random.random((num_elements, dim)))

# We split the data in two batches:
data1 = data[:num_elements // 2]
data2 = data[num_elements // 2:]

# Declaring index
p = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip

# Initializing index
# max_elements - the maximum number of elements (capacity). Will throw an exception if exceeded
# during insertion of an element.
# The capacity can be increased by saving/loading the index, see below.
#
# ef_construction - controls index search speed/build speed tradeoff
#
# M - is tightly connected with internal dimensionality of the data. Strongly affects memory consumption (~M)
# Higher M leads to higher accuracy/run_time at fixed ef/efConstruction

p.init_index(max_elements=num_elements//2, ef_construction=100, M=16)

# Controlling the recall by setting ef:
# higher ef leads to better accuracy, but slower search
p.set_ef(10)

# Set number of threads used during batch search/construction
# By default using all available cores
p.set_num_threads(4)

print("Adding first batch of %d elements" % (len(data1)))
p.add_items(data1)

# Query the elements for themselves and measure recall:
labels, distances = p.knn_query(data1, k=1)
print("Recall for the first batch:", np.mean(labels.reshape(-1) == np.arange(len(data1))), "\n")

# Serializing and deleting the index:
index_path='first_half.bin'
print("Saving index to '%s'" % index_path)
p.save_index("first_half.bin")
del p

# Re-initializing, loading the index
p = hnswlib.Index(space='l2', dim=dim)  # the space can be changed - keeps the data, alters the distance function.

print("\nLoading index from 'first_half.bin'\n")

# Increase the total capacity (max_elements), so that it will handle the new data
p.load_index("first_half.bin", max_elements = num_elements)

print("Adding the second batch of %d elements" % (len(data2)))
p.add_items(data2)

# Query the elements for themselves and measure recall:
labels, distances = p.knn_query(data, k=1)
print("Recall for two batches:", np.mean(labels.reshape(-1) == np.arange(len(data))), "\n")
```

#### C++ examples
[See examples here](examples/cpp/EXAMPLES.md):
* creating index, inserting elements, searching, serialization/deserialization
* filtering during the search with a boolean function
* deleting the elements and reusing the memory of the deleted elements for newly added elements
* multithreaded usage
* multivector search
* epsilon search


### Installation

This fork must be installed from source:

```bash
git clone https://github.com/adlumal/hnswlib-mmap.git
cd hnswlib-mmap
pip install .
```

Note: If you have the original hnswlib installed, this will replace it.


### For developers 
Contributions are highly welcome!

Please make pull requests against the `develop` branch.

When making changes please run tests (and please add a test to `tests/python` in case there is new functionality):
```bash
python -m unittest discover --start-directory tests/python --pattern "bindings_test*.py"
```


### Other implementations
* Non-metric space library (nmslib) - main library(python, C++), supports exotic distances: https://github.com/nmslib/nmslib
* Faiss library by facebook, uses own HNSW  implementation for coarse quantization (python, C++):
https://github.com/facebookresearch/faiss
* Code for the paper 
["Revisiting the Inverted Indices for Billion-Scale Approximate Nearest Neighbors"](https://arxiv.org/abs/1802.02422) 
(current state-of-the-art in compressed indexes, C++):
https://github.com/dbaranchuk/ivf-hnsw
* Amazon PECOS https://github.com/amzn/pecos 
* TOROS N2 (python, C++): https://github.com/kakao/n2 
* Online HNSW (C++): https://github.com/andrusha97/online-hnsw) 
* Go implementation: https://github.com/Bithack/go-hnsw
* Python implementation (as a part of the clustering code by by Matteo Dell'Amico): https://github.com/matteodellamico/flexible-clustering
* Julia implmentation https://github.com/JuliaNeighbors/HNSW.jl
* Java implementation: https://github.com/jelmerk/hnswlib
* Java bindings using Java Native Access: https://github.com/stepstone-tech/hnswlib-jna
* .Net implementation: https://github.com/curiosity-ai/hnsw-sharp
* CUDA implementation: https://github.com/js1010/cuhnsw
* Rust implementation https://github.com/rust-cv/hnsw
* Rust implementation for memory and thread safety purposes and There is  A Trait to enable the user to implement its own distances. It takes as data slices of types T satisfying T:Serialize+Clone+Send+Sync.: https://github.com/jean-pierreBoth/hnswlib-rs

### 200M SIFT test reproduction 
To download and extract the bigann dataset (from root directory):
```bash
python tests/cpp/download_bigann.py
```
To compile:
```bash
mkdir build
cd build
cmake ..
make all
```

To run the test on 200M SIFT subset:
```bash
./main
```

The size of the BigANN subset (in millions) is controlled by the variable **subset_size_millions** hardcoded in **sift_1b.cpp**.

### Updates test
To generate testing data (from root directory):
```bash
cd tests/cpp
python update_gen_data.py
```
To compile (from root directory):
```bash
mkdir build
cd build
cmake ..
make 
```
To run test **without** updates (from `build` directory)
```bash
./test_updates
```

To run test **with** updates (from `build` directory)
```bash
./test_updates update
```

### HNSW example demos

- Visual search engine for 1M amazon products (MXNet + HNSW): [website](https://thomasdelteil.github.io/VisualSearch_MXNet/), [code](https://github.com/ThomasDelteil/VisualSearch_MXNet), demo by [@ThomasDelteil](https://github.com/ThomasDelteil)

### References
HNSW paper:
```
@article{malkov2018efficient,
  title={Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs},
  author={Malkov, Yu A and Yashunin, Dmitry A},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={42},
  number={4},
  pages={824--836},
  year={2018},
  publisher={IEEE}
}
```

The update algorithm supported in this repository is to be published in "Dynamic Updates For HNSW, Hierarchical Navigable Small World Graphs" US Patent 15/929,802 by Apoorv Sharma, Abhishek Tayal and Yury Malkov.
