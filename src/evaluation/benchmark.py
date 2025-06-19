import time
import json
from src.modeling.knn import custom_knn_with_kernel
from src.data_ingestion.loader import testdata_knn

def run_knn_benchmark(N: int, D: int, A, X, K: int):
    """Simple timing for exact k-NN."""
    start = time.perf_counter()
    idx, vals = custom_knn_with_kernel(N, D, A, X, K)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"Exact k-NN: {elapsed:.2f} ms")
    return {"time_ms": elapsed, "indices": idx.tolist()}

def save_results(results: dict, fname: str):
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
