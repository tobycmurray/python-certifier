import numpy as np
from typing import List

from arithmetic import Q
from linear_algebra import Vector, Matrix


def relu_vec(v: Vector) -> Vector:
    return [x if x > 0 else Q(0) for x in v]

def forward(network: List[Matrix], x: Vector) -> Vector:
    """
    Forward pass with ReLU after each non-final layer; no biases
    """
    v = x[:]
    L = len(network)
    for idx, W in enumerate(network):
        v = mv_product(transpose(W), v)
        if idx < L - 1:
            v = relu_vec(v)
    return v

def _to_numpy32_matrix(W: Matrix) -> np.ndarray:
    """Convert a Q-matrix to np.float32 with shape (m, n)."""
    m = len(W)
    n = len(W[0]) if m else 0
    A = np.empty((m, n), dtype=np.float32)
    for i in range(m):
        row = W[i]
        for j in range(n):
            A[i, j] = np.float32(float(row[j]))  # exact decimal -> float -> f32
    return A

def forward_numpy_float32(network: List[Matrix], x: Vector) -> List[float]:
    """
    Float32 forward using NumPy:
      v <- (W^T) v, ReLU after each non-final layer, no biases.
    Returns Python floats (each representable as float32).
    """
    # convert once
    nets_np = [_to_numpy32_matrix(W) for W in network]
    v = np.array([np.float32(float(q)) for q in x], dtype=np.float32)
    L = len(nets_np)
    for ell, W in enumerate(nets_np):
        z = W.T @ v  # (n,) = (n,m) @ (m,)
        if ell < L - 1:
            # ReLU in float32
            v = np.maximum(z, np.float32(0.0), dtype=np.float32)
        else:
            v = z.astype(np.float32, copy=False)
    # return as Python floats (still float32 values)
    return [float(t) for t in v.tolist()]
