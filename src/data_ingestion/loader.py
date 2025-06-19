import numpy as np
import json
from pathlib import Path
from typing import Tuple, Optional, Union

def read_data(file_path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Read data from .npy or text file.
    """
    if not file_path:
        return None
    file_path = Path(file_path)
    if file_path.suffix == ".npy":
        return np.load(file_path)
    else:
        return np.loadtxt(file_path)

def testdata_knn(test_file: str = "") -> Tuple[int,int,np.ndarray,np.ndarray,int]:
    """
    Returns N, D, A matrix, X vector, and K for k-NN.
    """
    if not test_file:
        N, D, K = 1000, 100, 10
        A = np.random.randn(N, D)
        X = np.random.randn(D)
        return N, D, A, X, K
    data = json.load(open(test_file))
    A = np.loadtxt(data["a_file"])
    X = np.loadtxt(data["x_file"])
    return data["n"], data["d"], A, X, data["k"]

def testdata_ann(test_file: str = "") -> Tuple[int,int,np.ndarray,np.ndarray,int]:
    """Same signature as knn; for ANN."""
    return testdata_knn(test_file)
