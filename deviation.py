"""Layer-by-layer deviation computation for floating-point neural networks.

This module implements the deviation recursion from the LaTeX writeup (Lemma: One-step
floating-point deviation) and the Coq formalization (deviation.v).

The deviation at layer ℓ is bounded by:
    ||d_ℓ||_2 ≤ α_ℓ · ||d_{ℓ-1}||_2 + β_ℓ

where:
    - α_ℓ = ||W_ℓ||_2 + κ_ℓ · |||W_ℓ|||_2   (propagation factor)
    - β_ℓ = κ_ℓ · |||W_ℓ|||_2 · r_{ℓ-1} + (1+u)·a_dot(n_ℓ)·√m_ℓ   (new deviation)
    - κ_ℓ = γ_n + u·(1+γ_n)   (combined relative + absolute error)
    - γ_n = (n·u)/(1-n·u)     (relative error accumulation)

This enables forward induction through the network:
    - Layer 0: D_{-1} = 0 → compute D_0
    - Layer ℓ: D_{ℓ-1} known → compute D_ℓ
"""

from dataclasses import dataclass
from typing import List
from arithmetic import Q, float_to_q
from formats import FloatFormat, gamma_n, a_dot
from linear_algebra import Matrix, dims


@dataclass(frozen=True)
class LayerDeviationParams:
    """Deviation recursion parameters for a single layer.

    These parameters define the linear recursion ||d_ℓ||_2 ≤ α_ℓ · ||d_{ℓ-1}||_2 + β_ℓ
    """
    layer_idx: int    # Layer index ℓ
    alpha: Q          # Propagation factor: α_ℓ = ||W_ℓ||_2 + κ_ℓ·|||W_ℓ|||_2
    beta: Q           # New deviation budget: β_ℓ = κ_ℓ·|||W_ℓ|||_2·r_{ℓ-1} + noise
    gamma: Q          # γ_n = (n·u)/(1-n·u)
    kappa: Q          # κ = γ + u·(1+γ)


def compute_layer_deviation_params(
    layer_idx: int,
    op2_norm: Q,           # ||W_ℓ||_2
    op2_abs_norm: Q,       # |||W_ℓ|||_2
    r_prev: Q,             # r_{ℓ-1} (exact radius bound on input to this layer)
    layer_width: int,      # n_ℓ (input dimension)
    output_dim: int,       # m_ℓ (output dimension)
    sqrt_m: Q,             # √m_ℓ
    fmt: FloatFormat,
) -> LayerDeviationParams:
    """Compute deviation recursion parameters for a single layer.

    Implements the per-layer computation from the deviation recursion in the LaTeX writeup.

    Args:
        layer_idx: Layer index ℓ
        op2_norm: ||W_ℓ||_2 (operator 2-norm of weight matrix)
        op2_abs_norm: |||W_ℓ|||_2 (operator 2-norm of element-wise absolute values)
        r_prev: r_{ℓ-1} (exact radius bound on z_{ℓ-1})
        layer_width: n_ℓ (number of inputs to this layer)
        output_dim: m_ℓ (number of outputs from this layer)
        sqrt_m: √m_ℓ (square root of output dimension, precomputed)
        fmt: Floating-point format (contains u, denorm_min)

    Returns:
        LayerDeviationParams containing α_ℓ, β_ℓ, γ_n, κ

    Raises:
        ValueError: If n·u ≥ 1 (model is invalid for this format)
    """
    u = float_to_q(fmt.u)                  # Unit roundoff
    amul = float_to_q(fmt.denorm_min) / 2  # a_mul = denorm_min / 2

    # Compute margin terms
    gamma = gamma_n(layer_width, u)
    kappa = gamma + u * (Q(1) + gamma)

    # α_ℓ = ||W_ℓ||_2 + κ_ℓ · |||W_ℓ|||_2
    alpha = op2_norm + kappa * op2_abs_norm

    # β_ℓ = κ_ℓ · |||W_ℓ|||_2 · r_{ℓ-1} + (1+u)·a_dot(n_ℓ)·√m_ℓ
    beta_data = kappa * op2_abs_norm * r_prev
    beta_noise = (Q(1) + u) * a_dot(layer_width, u, amul) * sqrt_m
    beta = beta_data + beta_noise

    return LayerDeviationParams(
        layer_idx=layer_idx,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        kappa=kappa
    )


def compute_deviation_bound(
    D_prev: Q,
    params: LayerDeviationParams,
) -> Q:
    """Apply deviation recursion to compute D_ℓ from D_{ℓ-1}.

    Implements the linear recursion:
        D_ℓ = α_ℓ · D_{ℓ-1} + β_ℓ

    This is the sound upper bound on ||d_ℓ||_2 = ||ẑ_ℓ - z_ℓ||_2.

    Args:
        D_prev: D_{ℓ-1} (proven deviation bound from previous layer)
        params: Parameters for layer ℓ (α_ℓ, β_ℓ)

    Returns:
        D_ℓ (deviation bound for this layer)
    """
    return params.alpha * D_prev + params.beta


@dataclass(frozen=True)
class DeviationReport:
    """Report from deviation computation through all hidden layers.

    Contains the per-layer parameters and the final cumulative deviation D_{L-2}.
    """
    layer_params: List[LayerDeviationParams]  # Parameters for each hidden layer
    deviation_bounds: List[Q]                  # [D_0, D_1, ..., D_{L-2}]
    DLm1: Q                                     # D_{L-2} (final hidden layer deviation)
