import hnswlib
import h5py
import numpy as np
import json
import time
import sys

def benchmark(hdf5_path, metric='l2', k=10, ef_construction=100, m=16):
    # Load HDF5
    with h5py.File(hdf5_path, 'r') as f:
        train = f['train'][:]
        test  = f['test'][:]
        gt    = f['neighbors'][:]

    # Build index
    dim          = train.shape[1]
    num_elements = train.shape[0]

    metric_map  = {'l2': 'l2', 'cos': 'cosine', 'ip': 'ip'}
    hnsw_metric = metric_map.get(metric, 'l2')

    index = hnswlib.Index(space=hnsw_metric, dim=dim)
    index.init_index(max_elements=num_elements, ef_construction=ef_construction, M=m)

    start      = time.time()
    index.add_items(train)
    build_time = (time.time() - start) * 1000

    # Search benchmark
    results = []
    for ef in [10, 50, 100, 200, 400, 800]:
        index.set_ef(ef)

        start       = time.time()
        labels, _   = index.knn_query(test, k=k)
        search_time = (time.time() - start) * 1000

        hits = 0
        for i, neighbors in enumerate(labels):
            gt_set = set(gt[i][:k])
            hits  += len(set(neighbors) & gt_set)

        recall = hits / (len(test) * k)
        qps    = len(test) / (search_time / 1000)

        results.append({
            'ef':         ef,
            'recall':     round(recall, 4),
            'qps':        int(qps),
            'latency_ms': round(search_time / len(test), 4)
        })

    output = {
        'implementation':  'hnswlib',
        'dataset':         hdf5_path.split('/')[-1].rsplit('.', 1)[0],
        'metric':          metric,
        'M':               m,
        'ef_construction': ef_construction,
        'build_time_ms':   int(build_time),
        'results':         results
    }

    print(json.dumps(output, indent=2))

if __name__ == '__main__':
    benchmark(
        sys.argv[1],
        metric          = sys.argv[2]        if len(sys.argv) > 2 else 'l2',
        k               = int(sys.argv[3])   if len(sys.argv) > 3 else 10,
        m               = int(sys.argv[4])   if len(sys.argv) > 4 else 16,
        ef_construction = int(sys.argv[5])   if len(sys.argv) > 5 else 100
    )