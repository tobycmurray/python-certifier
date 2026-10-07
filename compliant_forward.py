"""Compliant numpy forward pass: IEEE-754 gradual-underflow execution matching the
certifier's rounding model, at a tunable float format.

Unlike the Keras/TensorFlow backend (which flushes subnormals to zero at
float32/float64), numpy preserves subnormals and rounds to nearest per op, so
this backend executes the deployed model under exactly the semantics the theory
(and the Rocq mechanisation) assume. Verified by the conformance probe
(tools/conformance_probe.py): no flush-to-zero / denormals-are-zero, and per-op
round-to-nearest, at float16/float32/float64.

Matvec accumulation:
  - float32/float64: np.einsum(..., optimize=False) -- numpy's own per-op
    sum-of-products, NOT a BLAS call (the same pattern the fp64 Gram iteration in
    linear_algebra.py relies on). Accumulates in the target format (per-op).
  - float16: numpy reductions promote fp16 accumulation to fp32 (wide), so we use
    an explicit pairwise-fp16 tree that casts to fp16 after every add, restoring
    genuine per-op fp16 (needed so the low-precision counterexamples reflect true
    fp16 rounding).

Interface mirrors keras_forward: prepare() builds the per-format arrays once,
forward_activations()/forward_logits() run an instance.
"""
from typing import List, Optional, Tuple
import numpy as np

from linear_algebra import Matrix
from arithmetic import Q  # noqa: F401  (documents the exact-rational input type)

_DT = {"float16": np.float16, "float32": np.float32, "float64": np.float64}


def _matvec(W: np.ndarray, x: np.ndarray, dt) -> np.ndarray:
    """fl(W x) with per-op round-to-nearest accumulation in format dt (no BLAS)."""
    if dt is np.float16:
        Wh = W.astype(np.float16)
        p = (Wh * x.astype(np.float16)[None, :]).astype(np.float16)
        while p.shape[1] > 1:
            if p.shape[1] % 2:
                p = np.concatenate([p, np.zeros((p.shape[0], 1), np.float16)], axis=1)
            p = (p[:, 0::2] + p[:, 1::2]).astype(np.float16)  # per-op fp16
        return p[:, 0]
    # float32/float64: numpy's own per-op sum-of-products, NOT a BLAS gemm.
    return np.einsum('ij,j->i', W.astype(dt), x.astype(dt), optimize=False)


PreparedModel = Tuple[List[np.ndarray], Optional[List[np.ndarray]], object]


def prepare(net: List[Matrix], biases: Optional[List], policy: str) -> PreparedModel:
    """Convert the certifier's exact-rational net (net[l] is (out,in)) and biases
    to numpy arrays at `policy`. Conversion is lossless when the exact values are
    representable in `policy` (float32 weights into float32/float64); at float16
    the cast is the (correct) deployment rounding of the weights."""
    if policy not in _DT:
        raise ValueError(f"unsupported policy {policy!r} (float16/float32/float64)")
    dt = _DT[policy]
    Ws = [np.array([[float(v) for v in row] for row in W], dtype=dt) for W in net]
    bs = None if biases is None else [np.array([float(v) for v in b], dtype=dt) for b in biases]
    return (Ws, bs, dt)


def forward_pre_and_activations(model: PreparedModel, x) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """(pre-activations, activations) per layer. The pre-activation of layer l is
    the linear part fl(fl(W_l z) + b_l) BEFORE the activation (ReLU hidden,
    identity output; so the last pre-activation is the logit vector). x is any
    iterable of numbers (exact rationals or floats); it is cast to the model's
    format."""
    Ws, bs, dt = model
    z = np.array([float(v) for v in x], dtype=dt)
    L = len(Ws)
    pres = []
    outs = []
    for l in range(L):
        a = _matvec(Ws[l], z, dt)
        if bs is not None:
            a = (a + bs[l]).astype(dt)          # per-op bias add
        pres.append(a)
        z = np.maximum(a, dt(0.0)) if l < L - 1 else a   # ReLU / identity
        outs.append(z)
    return pres, outs


def forward_activations(model: PreparedModel, x) -> List[np.ndarray]:
    """Per-layer activations (ReLU hidden, identity output). x is any iterable of
    numbers (exact rationals or floats); it is cast to the model's format."""
    return forward_pre_and_activations(model, x)[1]


def forward_logits(model: PreparedModel, x) -> np.ndarray:
    """Final-layer logits (the deployed model's y for input x)."""
    return forward_activations(model, x)[-1]
