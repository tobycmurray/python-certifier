"""Run the actual Keras model to obtain the measured (hybrid-meas) forward-pass
activations, instead of a manual numpy re-simulation.

This is what makes hybrid-meas sound by construction: the measured deviation is
taken from the *same* execution that produces the deployed logits, so there is no
numpy-vs-Keras conservatism assumption. It also handles biases natively (build
the model with bias terms when --biases is present), which the numpy path did
not -- the bug that let the biased model be (wrongly) certified.

We build a *flat* Dense model (Input(flat_dim) -> Dense x L). The deployed model
is Input(H,W,C) -> Flatten -> Dense x L; Flatten is a pure reshape, so on the
already-flattened certifier input the two compute bit-identical Dense ops. We
verify this by checking that the rebuilt model reproduces the recorded y1 exactly
(see verify_reproduces_logits).
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from typing import List, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

from linear_algebra import Matrix, Vector
from arithmetic import Q

# Policy -> numpy dtype for the executed weights/compute.
_POLICY_DTYPE = {"float16": np.float16, "float32": np.float32, "float64": np.float64}


def _kernel(W: Matrix, npdt) -> np.ndarray:
    """Keras Dense kernel (in, out) from a certifier weight matrix (out, in)."""
    out_dim = len(W)
    in_dim = len(W[0])
    K = np.empty((in_dim, out_dim), dtype=npdt)
    for i in range(out_dim):
        row = W[i]
        for j in range(in_dim):
            K[j, i] = npdt(float(row[j]))
    return K


def _bias(b, npdt) -> np.ndarray:
    return np.array([npdt(float(v)) for v in b], dtype=npdt)


def build_activation_model(net: List[Matrix],
                           biases: Optional[List[Vector]],
                           policy: str) -> Model:
    """A multi-output Keras model giving the per-layer activations of `net`.

    net[l] is (out, in); biases (if given) is a list of per-layer bias vectors.
    `policy` is 'float32' (target execution) or 'float64' (high-precision ref).
    Weights/biases are set from the certifier's exact values, so the executed
    model is exactly the model being certified.
    """
    if policy not in _POLICY_DTYPE:
        raise ValueError(f"unsupported policy {policy!r} (expected float32/float64)")
    npdt = _POLICY_DTYPE[policy]
    mixed_precision.set_global_policy(policy)

    L = len(net)
    in_dim = len(net[0][0])
    use_bias = biases is not None

    inp = Input((in_dim,))
    z = inp
    dense_layers = []
    for l in range(L):
        out_dim = len(net[l])
        activation = "relu" if l < L - 1 else None   # identity on the output layer
        layer = Dense(out_dim, use_bias=use_bias, activation=activation)
        z = layer(z)
        dense_layers.append(layer)
    model = Model(inp, [layer.output for layer in dense_layers])

    for l, layer in enumerate(dense_layers):
        K = _kernel(net[l], npdt)
        if use_bias:
            layer.set_weights([K, _bias(biases[l], npdt)])
        else:
            layer.set_weights([K])
    return model


def forward_activations(model: Model, x: Vector, policy: str) -> List[np.ndarray]:
    """Run `model` on flat input x; return per-layer activations as numpy arrays."""
    npdt = _POLICY_DTYPE[policy]
    xv = np.asarray([float(v) for v in x], dtype=npdt)[None, :]
    outs = model(xv, training=False)
    if not isinstance(outs, (list, tuple)):
        outs = [outs]
    return [np.asarray(o)[0] for o in outs]


def verify_reproduces_logits(model_fp: Model, x: Vector, y1, policy: str,
                             atol: float = 0.0) -> bool:
    """Check the rebuilt target-format model reproduces the recorded logits y1.

    With atol=0 this asserts bit-for-bit equality, establishing that the rebuilt
    model is the same execution that generated y1.
    """
    out = forward_activations(model_fp, x, policy)[-1].astype(np.float64)
    y1a = np.asarray(y1, dtype=np.float64)
    return bool(np.array_equal(out, y1a)) if atol == 0.0 else bool(
        np.max(np.abs(out - y1a)) <= atol)
