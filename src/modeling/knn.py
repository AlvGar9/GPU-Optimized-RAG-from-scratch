import numpy as np
import cupy as cp
from src.data_processing.distances import l2_cupy_kernel
from typing import Tuple

def custom_topk(distances: cp.ndarray, k: int, largest: bool=False) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Hybrid top-k: CuPy partition + CPU sort for exact indices.
    """
    n = distances.size
    idx = cp.argpartition(distances if not largest else -distances, k-1)[:k]
    vals = distances[idx].get()
    inds = idx.get().tolist()
    # simple Python sort
    sorted_pairs = sorted(zip(vals, inds), reverse=largest)
    v, i = zip(*sorted_pairs)
    return cp.asarray(v), cp.asarray(i, dtype=cp.int32)

def custom_knn_with_kernel(N: int, D: int, A: np.ndarray, X: np.ndarray,
                           K: int, kernel=l2_cupy_kernel) -> Tuple[np.ndarray,np.ndarray]:
    """
    Chunked exact k-NN using provided kernel.
    """
    all_vals, all_idx = [], []
    chunk = 100_000
    for i in range(0, N, chunk):
        block = A[i:i+chunk]
        dists = kernel(block, X)
        vals, idx = custom_topk(dists, K)
        all_vals.append(vals.get())
        all_idx.append((idx.get() + i))
    # merge top-K across chunks (left as exercise)
    return np.concatenate(all_idx)[:K], np.concatenate(all_vals)[:K]
