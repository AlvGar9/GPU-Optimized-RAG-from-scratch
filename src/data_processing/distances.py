from typing import Tuple
import numpy as np
import cupy as cp
import torch
import triton
import triton.language as tl

# -- NumPy baseline --
def l2_numpy(x: np.ndarray, y: np.ndarray) -> float:
    return np.linalg.norm(x - y)

def dot_numpy(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.dot(x, y))

def manhattan_numpy(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum(np.abs(x - y)))

def cosine_numpy(x: np.ndarray, y: np.ndarray) -> float:
    denom = np.linalg.norm(x) * np.linalg.norm(y) + 1e-8
    return 1.0 - np.dot(x, y) / denom

# -- PyTorch built-ins --
def l2_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.norm(x - y)

def cosine_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 1 - torch.nn.functional.cosine_similarity(x, y, dim=0)

# -- CuPy custom kernel example --
_l2_kernel = cp.RawKernel(r'''
extern "C" __global__
void l2_optimized(const float* A, const float* X, float* out, int D) {
  int tid = threadIdx.x, row = blockIdx.x;
  float sum=0; int offset=row*D;
  for(int i=tid;i<D;i+=blockDim.x) {
    float d=A[offset+i]-X[i];
    sum+=d*d;
  }
  __shared__ float cache[1024];
  cache[tid]=sum; __syncthreads();
  if(tid==0){ float tot=0; for(int i=0;i<blockDim.x;i++) tot+=cache[i]; out[row]=sqrtf(tot);}
}
''','l2_optimized')

def l2_cupy_kernel(A: np.ndarray, X: np.ndarray) -> cp.ndarray:
    A_gpu = cp.asarray(A, dtype=cp.float32)
    X_gpu = cp.asarray(X, dtype=cp.float32)
    out = cp.zeros((A.shape[0],), dtype=cp.float32)
    threads = min(1024, A.shape[1])
    _l2_kernel((A.shape[0],),(threads,),(A_gpu, X_gpu, out, A.shape[1]), shared_mem=threads*4)
    return out

# -- Triton wrapper example --
@triton.jit
def _triton_l2(X, Y, partials, D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offs < D
    x = tl.load(X + offs, mask=m, other=0.0)
    y = tl.load(Y + offs, mask=m, other=0.0)
    diff = x - y
    out = tl.sum(diff * diff, axis=0)
    tl.store(partials + pid, out)

def l2_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    D = x.numel()
    B = 1024
    blocks = (D + B - 1)//B
    partials = torch.zeros((blocks,), device=x.device)
    _triton_l2[(blocks,)](x, y, partials, D=D, BLOCK_SIZE=B, num_warps=4)
    return torch.sqrt(partials.sum())
