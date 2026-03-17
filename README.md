# dementiaDB

A vector database management system built from scratch in C++17. Supports HNSW indexing, metadata storage, persistence, and soft deletes — designed to be embedded as a library or run as a standalone server.

> Built to understand how vector databases work at a low level, not to replace production systems.

---

## Features

- **HNSW index**
- **Multiple distance metrics**:  L2, Cosine, Normalized Cosine, Inner Product
- **Metadata storage**: attach arbitrary JSON metadata to every vector
- **Persistence**: save and load collections to/from binary files
- **Soft deletes**: mark vectors as deleted without rebuilding the index
- **Bulk loading**: load vectors from HDF5 files + JSON metadata files
- **Parallel build**: OpenMP-accelerated index building

---

## Building

### Dependencies

```bash
sudo apt install cmake ninja-build g++
sudo apt install libhdf5-dev
sudo apt install libomp-dev
```

### Configure & Build

```bash
cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Options

| Option | Default | Description |
|---|---|---|
| `DEMENTIA_BUILD_EXAMPLES` | `ON` | Build example programs |
| `DEMENTIA_BUILD_BENCH` | `ON` | Build HDF5 benchmark |

---

## Usage

### As a library (FetchContent)

```cmake
include(FetchContent)
FetchContent_Declare(
    dementiaDB
    GIT_REPOSITORY https://github.com/KaindraDjoemena/dementiaDB.git
    GIT_TAG main
)
FetchContent_MakeAvailable(dementiaDB)
target_link_libraries(your_app PRIVATE dementia_engine)
```

> Requires HDF5 and OpenMP installed on the system.

### Basic example

```cpp
#include "dementiaDB.h"
using namespace demDB;

DementiaDB db;
db.createCollection("my_collection", INDEX_TYPE::HNSW);

// configure index
db.setIndexMetric("my_collection", METRIC::L2);
db.setIndexHNSWParam("my_collection", 16,  HNSW_PARAM::M);
db.setIndexHNSWParam("my_collection", 100, HNSW_PARAM::EF_CONSTRUCT);

// insert vectors
std::vector<float> vec(128, 0.5f);
nlohmann::json meta = { {"label", "example"} };
db.insert("my_collection", vec.data(), meta);

// search
std::vector<float> query(128, 0.4f);
auto results = db.search("my_collection", query, 10);
for (auto& r : results)
    std::cout << r.dump() << "\n";

// persist
db.save("my_db.dem");
db.open("my_db.dem");
```

### Bulk loading from HDF5

```cpp
db.bulkLoad("my_collection", "embeddings.hdf5", "vectors", "metadata.json");
```

The expected workflow:
1. A Python script embeds text and saves vectors to `.hdf5` and metadata to `.json`
2. `bulkLoad` ingests both into a collection in one call

---

## Benchmark

Benchmarks run against the [ANN Benchmarks](http://ann-benchmarks.com) HDF5 datasets. Download `sift-128-euclidean.hdf5` from there to reproduce.

### C++

```bash
./build/hdf5_bench <path_to_file.hdf5> <metric> <top_k>
# example:
./build/hdf5_bench data/sift-128-euclidean.hdf5 l2 10
```

Metrics: `l2`, `cos`, `ncos`, `ip`

### Python (hnswlib baseline)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r bench/requirements.txt

python3 bench/hdf5_bench.py <path_to_file.hdf5> <metric> <top_k>
```

### Results — SIFT-128 Euclidean (M=16, efConstruction=100, 1M vectors)

| ef | dementiaDB recall | dementiaDB QPS | dementiaDB latency | hnswlib recall | hnswlib QPS | hnswlib latency |
|:---:|:-----------------:|:--------------:|:------------------:|:--------------:|:-----------:|:---------------:|
| 10  | 0.0994 | 80,000 | 0.013 ms | 0.9119 | 7,193  | 0.139 ms |
| 50  | 0.4997 | 27,548 | 0.036 ms | 0.9119 | 29,504 | 0.034 ms |
| 100 | 0.9283 | 15,625 | 0.064 ms | 0.9119 | 29,298 | 0.034 ms |
| 200 | 0.9765 | 9,199  | 0.109 ms | 0.9706 | 15,155 | 0.066 ms |
| 400 | 0.9941 | 5,068  | 0.197 ms | 0.9923 | 9,115  | 0.110 ms |
| 800 | 0.9987 | 2,735  | 0.366 ms | 0.9983 | 5,690  | 0.176 ms |

**Build time:** dementiaDB ~78s · hnswlib ~29s

> dementiaDB runs ~2-3x slower than hnswlib at high recall targets. Unlike hnswlib's

> batched query mode, recall here scales correctly with ef as expected by the algorithm

---

## Architecture

```
src/
  dementiaDB.cpp/h   — top-level VDBMS, collection management
  index.cpp/h        — abstract index base class
  indexFlat.cpp/h    — brute-force flat index
  indexHnsw.cpp/h    — HNSW index implementation
  metadata.cpp/h     — JSON metadata store
utils/
  mathutils.h        — SIMD-optimized distance functions (L2, cosine, dot)
  timer.h            — scope timer utility
bench/
  hdf5_bench.cpp     — C++ benchmark
  hdf5_bench.py      — hnswlib Python baseline
examples/
  saving.cpp         — save a collection to disk
  opening.cpp        — load a collection from disk
```

---

## References

- [Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs](https://arxiv.org/pdf/1603.09320) — Malkov & Yashunin
- [ANN Benchmarks](http://ann-benchmarks.com) — benchmark datasets
- [hnswlib](https://github.com/nmslib/hnswlib) — used as baseline
