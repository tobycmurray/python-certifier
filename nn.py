import numpy as np
from typing import List

from arithmetic import Q
from linear_algebra import Vector, Matrix, mv_product


def relu_vec(v: Vector) -> Vector:
    return [x if x > 0 else Q(0) for x in v]

def forward(network: List[Matrix], x: Vector) -> Vector:
    """
    Forward pass with ReLU after each non-final layer; no biases
    """
    v = x[:]
    L = len(network)
    for idx, W in enumerate(network):
        v = mv_product(W, v)
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
        z = W @ v  # (n,) = (n,m) @ (m,)
        if ell < L - 1:
            # ReLU in float32
            v = np.maximum(z, np.float32(0.0), dtype=np.float32)
        else:
            v = z.astype(np.float32, copy=False)
    # return as Python floats (still float32 values)
    return [float(t) for t in v.tolist()]

def _to_numpy16_matrix(W: Matrix) -> np.ndarray:
    """Convert a Q-matrix to np.float16 with shape (m, n)."""
    m = len(W)
    n = len(W[0]) if m else 0
    A = np.empty((m, n), dtype=np.float16)
    for i in range(m):
        row = W[i]
        for j in range(n):
            A[i, j] = np.float16(float(row[j]))  # exact decimal -> float -> f16
    return A

def forward_numpy_float16(network: List[Matrix], x: Vector) -> List[float]:
    """
    Float16 forward using NumPy:
      v <- (W^T) v, ReLU after each non-final layer, no biases.
    Returns Python floats (each representable as float16).
    """
    # convert once
    nets_np = [_to_numpy16_matrix(W) for W in network]
    v = np.array([np.float16(float(q)) for q in x], dtype=np.float16)
    L = len(nets_np)
    for ell, W in enumerate(nets_np):
        z = W @ v  # (n,) = (n,m) @ (m,)
        if ell < L - 1:
            # ReLU in float16
            v = np.maximum(z, np.float16(0.0), dtype=np.float16)
        else:
            v = z.astype(np.float16, copy=False)
    # return as Python floats (still float16 values)
    return [float(t) for t in v.tolist()]

def forward_layerwise_exact(network: List[Matrix], x: Vector) -> List[Vector]:
    """
    Exact forward pass returning activations at each layer.
    Returns list [z_0, z_1, ..., z_{L-1}] where z_ℓ is the activation after layer ℓ.
    """
    activations = []
    v = x[:]
    L = len(network)
    for idx, W in enumerate(network):
        v = mv_product(W, v)
        if idx < L - 1:
            v = relu_vec(v)
        activations.append(v)
    return activations

def forward_layerwise_float32(network: List[Matrix], x: Vector) -> List[np.ndarray]:
    """
    Float32 forward pass returning activations at each layer.
    Returns list [z_0, z_1, ..., z_{L-1}] where z_ℓ is the activation after layer ℓ.
    Each activation is a numpy array of float32.
    """
    nets_np = [_to_numpy32_matrix(W) for W in network]
    v = np.array([np.float32(float(q)) for q in x], dtype=np.float32)
    activations = []
    L = len(nets_np)
    for ell, W in enumerate(nets_np):
        z = W @ v  # (n,) = (n,m) @ (m,)
        if ell < L - 1:
            # ReLU in float32
            v = np.maximum(z, np.float32(0.0), dtype=np.float32)
        else:
            v = z.astype(np.float32, copy=False)
        activations.append(v.copy())  # Store activation after layer ell
    return activations

def forward_layerwise_float16(network: List[Matrix], x: Vector) -> List[np.ndarray]:
    """
    Float16 forward pass returning activations at each layer.
    Returns list [z_0, z_1, ..., z_{L-1}] where z_ℓ is the activation after layer ℓ.
    Each activation is a numpy array of float16.
    """
    nets_np = [_to_numpy16_matrix(W) for W in network]
    v = np.array([np.float16(float(q)) for q in x], dtype=np.float16)
    activations = []
    L = len(nets_np)
    for ell, W in enumerate(nets_np):
        z = W @ v  # (n,) = (n,m) @ (m,)
        if ell < L - 1:
            # ReLU in float16
            v = np.maximum(z, np.float16(0.0), dtype=np.float16)
        else:
            v = z.astype(np.float16, copy=False)
        activations.append(v.copy())  # Store activation after layer ell
    return activations

def forward_layerwise_float64(network: List[Matrix], x: Vector) -> List[np.ndarray]:
    """
    Float64 forward pass returning activations at each layer (high precision baseline).
    Returns list [z_0, z_1, ..., z_{L-1}] where z_ℓ is the activation after layer ℓ.
    Each activation is a numpy array of float64.
    """
    # Convert to float64 matrices
    nets_np = []
    for W in network:
        m = len(W)
        n = len(W[0]) if m else 0
        A = np.empty((m, n), dtype=np.float64)
        for i in range(m):
            row = W[i]
            for j in range(n):
                A[i, j] = float(row[j])
        nets_np.append(A)

    v = np.array([float(q) for q in x], dtype=np.float64)
    activations = []
    L = len(nets_np)
    for ell, W in enumerate(nets_np):
        z = W @ v
        if ell < L - 1:
            v = np.maximum(z, 0.0)
        else:
            v = z
        activations.append(v.copy())
    return activations

def compute_deviations(exact_activations: List[Vector],
                      fp_activations: List[np.ndarray]) -> List[float]:
    """
    Compute L2 norm of deviation at each layer.

    Args:
        exact_activations: List of exact activations (Q vectors)
        fp_activations: List of FP activations (numpy arrays)

    Returns:
        List of L2 norms [||ẑ_0 - z_0||_2, ..., ||ẑ_{L-1} - z_{L-1}||_2]
    """
    deviations = []
    for z_exact, z_fp in zip(exact_activations, fp_activations):
        # Convert exact to float64 for comparison
        z_exact_np = np.array([float(q) for q in z_exact], dtype=np.float64)
        z_fp_np = z_fp.astype(np.float64)

        # Compute L2 norm of difference
        diff = z_fp_np - z_exact_np
        l2_norm = np.linalg.norm(diff, ord=2)
        deviations.append(float(l2_norm))

    return deviations


def convert_network_to_numpy64(network: List[Matrix]) -> List[np.ndarray]:
    """
    Convert Q network to numpy float64 matrices (one-time conversion).

    This is much more efficient than converting on every forward pass.
    Use this once at startup, then use forward_layerwise_float64_optimized
    for all subsequent forward passes.

    Args:
        network: List of Q weight matrices

    Returns:
        List of numpy float64 weight matrices
    """
    weights_np = []
    for W in network:
        m = len(W)
        n = len(W[0]) if m > 0 else 0
        W_np = np.empty((m, n), dtype=np.float64)
        for i in range(m):
            for j in range(n):
                W_np[i, j] = float(W[i][j])
        weights_np.append(W_np)
    return weights_np


def forward_layerwise_float64_optimized(weights_np: List[np.ndarray],
                                        x: Vector) -> List[np.ndarray]:
    """
    Optimized float64 forward pass using pre-converted weight matrices.

    This is ~750× faster than forward_layerwise_float64() because it avoids
    converting Q matrices to numpy on every forward pass.

    Usage:
        # One-time conversion (do once at startup):
        weights_np = convert_network_to_numpy64(network)

        # Fast forward passes (do for each input):
        activations = forward_layerwise_float64_optimized(weights_np, x)

    Args:
        weights_np: Pre-converted numpy float64 weight matrices
        x: Input vector (Q or numpy)

    Returns:
        List [z_0, z_1, ..., z_{L-1}] where z_ℓ is the activation after layer ℓ.
        Each activation is a numpy array of float64.
    """
    # Convert input to numpy if needed
    if len(x) > 0 and isinstance(x[0], Q):
        v = np.array([float(q) for q in x], dtype=np.float64)
    else:
        v = np.asarray(x, dtype=np.float64)

    activations = []
    L = len(weights_np)

    for ell, W in enumerate(weights_np):
        z = W @ v
        if ell < L - 1:
            v = np.maximum(z, 0.0)
        else:
            v = z
        activations.append(v.copy())

    return activations


def convert_network_to_numpy32(network: List[Matrix]) -> List[np.ndarray]:
    """
    Convert Q network to numpy float32 matrices (one-time conversion).

    This is much more efficient than converting on every forward pass.
    Use this once at startup, then use forward_layerwise_float32_optimized
    for all subsequent forward passes.

    Args:
        network: List of Q weight matrices

    Returns:
        List of numpy float32 weight matrices
    """
    weights_np = []
    for W in network:
        m = len(W)
        n = len(W[0]) if m > 0 else 0
        W_np = np.empty((m, n), dtype=np.float32)
        for i in range(m):
            for j in range(n):
                W_np[i, j] = np.float32(float(W[i][j]))
        weights_np.append(W_np)
    return weights_np


def forward_layerwise_float32_optimized(weights_np: List[np.ndarray],
                                        x: Vector) -> List[np.ndarray]:
    """
    Optimized float32 forward pass using pre-converted weight matrices.

    This is much faster than forward_layerwise_float32() because it avoids
    converting Q matrices to numpy on every forward pass.

    Usage:
        # One-time conversion (do once at startup):
        weights_np = convert_network_to_numpy32(network)

        # Fast forward passes (do for each input):
        activations = forward_layerwise_float32_optimized(weights_np, x)

    Args:
        weights_np: Pre-converted numpy float32 weight matrices
        x: Input vector (Q or numpy)

    Returns:
        List [z_0, z_1, ..., z_{L-1}] where z_ℓ is the activation after layer ℓ.
        Each activation is a numpy array of float32.
    """
    # Convert input to numpy float32
    if len(x) > 0 and isinstance(x[0], Q):
        v = np.array([np.float32(float(q)) for q in x], dtype=np.float32)
    else:
        v = np.asarray(x, dtype=np.float32)

    activations = []
    L = len(weights_np)

    for ell, W in enumerate(weights_np):
        z = W @ v
        if ell < L - 1:
            v = np.maximum(z, np.float32(0.0), dtype=np.float32)
        else:
            v = z.astype(np.float32, copy=False)
        activations.append(v.copy())

    return activations


def convert_network_to_numpy16(network: List[Matrix]) -> List[np.ndarray]:
    """
    Convert Q network to numpy float16 matrices (one-time conversion).

    This is much more efficient than converting on every forward pass.
    Use this once at startup, then use forward_layerwise_float16_optimized
    for all subsequent forward passes.

    Args:
        network: List of Q weight matrices

    Returns:
        List of numpy float16 weight matrices
    """
    weights_np = []
    for W in network:
        m = len(W)
        n = len(W[0]) if m > 0 else 0
        W_np = np.empty((m, n), dtype=np.float16)
        for i in range(m):
            for j in range(n):
                W_np[i, j] = np.float16(float(W[i][j]))
        weights_np.append(W_np)
    return weights_np


def forward_layerwise_float16_optimized(weights_np: List[np.ndarray],
                                        x: Vector) -> List[np.ndarray]:
    """
    Optimized float16 forward pass using pre-converted weight matrices.

    This is much faster than forward_layerwise_float16() because it avoids
    converting Q matrices to numpy on every forward pass.

    Usage:
        # One-time conversion (do once at startup):
        weights_np = convert_network_to_numpy16(network)

        # Fast forward passes (do for each input):
        activations = forward_layerwise_float16_optimized(weights_np, x)

    Args:
        weights_np: Pre-converted numpy float16 weight matrices
        x: Input vector (Q or numpy)

    Returns:
        List [z_0, z_1, ..., z_{L-1}] where z_ℓ is the activation after layer ℓ.
        Each activation is a numpy array of float16.
    """
    # Convert input to numpy float16
    if len(x) > 0 and isinstance(x[0], Q):
        v = np.array([np.float16(float(q)) for q in x], dtype=np.float16)
    else:
        v = np.asarray(x, dtype=np.float16)

    activations = []
    L = len(weights_np)

    for ell, W in enumerate(weights_np):
        z = W @ v
        if ell < L - 1:
            v = np.maximum(z, np.float16(0.0), dtype=np.float16)
        else:
            v = z.astype(np.float16, copy=False)
        activations.append(v.copy())

    return activations


def measure_center_diff_norm(
    weights_np64: List[np.ndarray],
    weights_np_target: List[np.ndarray],
    x: Vector,
    hidden_layers: int,
) -> float:
    """Measure ||z^hi_{H-1}(x) - z^fp_{H-1}(x)||_2 without storing intermediate activations.

    Runs the fp64 and target-format forward passes simultaneously through the
    hidden layers only (stopping before the output layer), keeping only the
    current layer's activation in each format.  No intermediate arrays are
    allocated or returned.

    Args:
        weights_np64:      Pre-converted fp64 weight matrices (length >= hidden_layers).
        weights_np_target: Pre-converted target-format weight matrices (same length).
                           Must have the same length as weights_np64.
                           Pass weights_np64 again when target IS fp64 (returns 0.0).
        x:                 Input vector (Q or numpy-compatible).
        hidden_layers:     Number of hidden layers H = L-1 to process.

    Returns:
        ||z^hi_{H-1} - z^fp_{H-1}||_2 as a Python float.
    """
    if weights_np_target is weights_np64:
        return 0.0

    # Convert input
    if len(x) > 0 and isinstance(x[0], Q):
        v_hi = np.array([float(q) for q in x], dtype=np.float64)
    else:
        v_hi = np.asarray(x, dtype=np.float64)

    target_dtype = weights_np_target[0].dtype
    v_fp = v_hi.astype(target_dtype)

    # Run through hidden layers only — no intermediate allocation
    zero_hi = np.float64(0.0)
    zero_fp = target_dtype.type(0)

    for ell in range(hidden_layers):
        v_hi = np.maximum(weights_np64[ell] @ v_hi, zero_hi)
        v_fp = np.maximum(weights_np_target[ell] @ v_fp, zero_fp)

    return float(np.linalg.norm(v_fp.astype(np.float64) - v_hi, ord=2))
