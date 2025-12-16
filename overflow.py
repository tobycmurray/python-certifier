from dataclasses import dataclass
from typing import List, Optional, Tuple
from arithmetic import Q, float_to_q
from formats import FloatFormat, gamma_n, a_dot_fwd

@dataclass(frozen=True)
class OverflowLayerStats:
    """Statistics for overflow checking on a single layer.

    All quantities use the corrected theory from LaTeX Theorem 3.1 + Coq proof,
    which accounts for roundoff error accumulation in floating-point dot products.

    Uses tighter bounds matching Coq formalization:
    - S_layer uses max row L2 norm (not spectral norm of |W|)
    - M_layer uses max absolute entry (not max row sum)
    """
    layer_idx: int
    layer_width: int       # n_l (number of inputs to this layer)
    max_row_l2: Q         # max_r ‖W_r‖_2 (for S_layer)
    max_abs_entry: Q      # max_{r,k} |W_{r,k}| (for M_layer)
    S_layer: Q            # max_r ‖W_r‖_2 · (r_{l-1} + D_{l-1})
    M_layer: Q            # max_{r,k} |W_{r,k}| · (r_{l-1} + D_{l-1})
    gamma_n: Q            # Relative error accumulation: γ_n = (n·u)/(1-n·u)
    a_dot_n: Q            # Absolute error for subnormals: a_dot_fwd(n) = (1+γ_{n-1})·n·a_mul
    S_with_margin: Q      # S_layer · (1 + γ_n) + a_dot(n) [what actually needs checking]
    slack_2: Q            # F_max - S_with_margin (>0 means no overflow)
    slack_inf: Q          # F_max - M_layer (>0 means no overflow)
    ok: bool

@dataclass(frozen=True)
class OverflowReport:
    """Report from overflow certification.

    Verifies that no overflow can occur during floating-point evaluation
    of the network on any input in the certified ball.
    """
    layers: List[OverflowLayerStats]
    ok: bool
    first_failure: Optional[Tuple[int, str]]  # (layer_idx, "S_layer"|"M_layer")


def check_overflow_single_layer(
    layer_idx: int,
    max_row_l2: Q,           # max_r ‖W_r‖_2 (max row L2 norm)
    max_abs_entry: Q,        # max_{r,k} |W_{r,k}| (max absolute entry)
    fp_activation_bound: Q,  # r_{ℓ-1} + D_{ℓ-1} (bound on ‖ẑ_{ℓ-1}‖_2)
    layer_width: int,        # n_ℓ (input dimension)
    fmt: FloatFormat,
    bias_norm: Q = Q(0),     # ‖b_ℓ‖_∞ (default 0 if no bias)
) -> OverflowLayerStats:
    """Check overflow for a single layer using FP activation bound.

    This implements the deviation-aware overflow check from the updated LaTeX writeup
    (Theorem 3.1) and Coq formalization (layerwise_simple_with_deviation_implies_finite).

    For layer ℓ, verifies:
        1. S_layer · (1 + γ_n) + a_dot(n) + ‖b_ℓ‖_∞ < F_max
        2. M_layer < F_max

    where (using tighter bounds matching Coq):
        - S_layer = (max_r ‖W_r‖_2) · (r_{ℓ-1} + D_{ℓ-1})
        - M_layer = (max_{r,k} |W_{r,k}|) · (r_{ℓ-1} + D_{ℓ-1})
        - fp_activation_bound = r_{ℓ-1} + D_{ℓ-1} from triangle inequality

    Args:
        layer_idx: Layer index ℓ
        max_row_l2: max_r ‖W_r‖_2 (max row L2 norm)
        max_abs_entry: max_{r,k} |W_{r,k}| (max absolute entry in matrix)
        fp_activation_bound: r_{ℓ-1} + D_{ℓ-1} (bounds ‖ẑ_{ℓ-1}‖_2)
        layer_width: n_ℓ (input dimension)
        fmt: Floating-point format
        bias_norm: ‖b_ℓ‖_∞ (infinity norm of bias, 0 if no bias)

    Returns:
        OverflowLayerStats with pass/fail and detailed statistics

    Raises:
        ValueError: If n·u ≥ 1 (model is invalid for this format)
    """
    Fmax_q = float_to_q(fmt.Fmax)
    u_q = float_to_q(fmt.u)
    a_mul_q = float_to_q(fmt.denorm_min) / 2

    # Compute bounds using FP activation bound (not just exact radius!)
    # Uses tighter bounds matching Coq: max row norms instead of spectral/matrix norms
    S_layer = max_row_l2 * fp_activation_bound       # (max_r ‖W_r‖_2) · (r + D)
    M_layer = max_abs_entry * fp_activation_bound    # (max |W_{r,k}|) · (r + D)

    # Compute margin terms
    gamma = gamma_n(layer_width, u_q)
    a_dot_n = a_dot_fwd(layer_width, u_q, a_mul_q)

    # Apply margin to S_layer check
    S_with_margin = S_layer * (1 + gamma) + a_dot_n + bias_norm

    # Compute slack (positive = passes)
    s2 = Fmax_q - S_with_margin
    si = Fmax_q - M_layer

    ok = (s2 > 0) and (si > 0)

    return OverflowLayerStats(
        layer_idx=layer_idx,
        layer_width=layer_width,
        max_row_l2=max_row_l2,
        max_abs_entry=max_abs_entry,
        S_layer=S_layer,
        M_layer=M_layer,
        gamma_n=gamma,
        a_dot_n=a_dot_n,
        S_with_margin=S_with_margin,
        slack_2=s2,
        slack_inf=si,
        ok=ok
    )
