from dataclasses import dataclass
from typing import List, Optional, Tuple
from arithmetic import Q, float_to_q
from formats import FloatFormat, gamma_n, a_dot

@dataclass(frozen=True)
class OverflowLayerStats:
    """Statistics for overflow checking on a single layer.

    All quantities use the corrected theory from LaTeX Theorem 3.1 + Coq proof,
    which accounts for roundoff error accumulation in floating-point dot products.
    """
    layer_idx: int
    layer_width: int       # n_l (number of inputs to this layer)
    norm2_abs: Q          # ‖|W_l|‖_2
    norminf: Q            # ‖W_l‖_∞
    S_layer: Q            # ‖|W_l|‖_2 · r_{l-1} (exact bound)
    M_layer: Q            # ‖W_l‖_∞ · r_{l-1}
    gamma_n: Q            # Relative error accumulation: γ_n = (n·u)/(1-n·u)
    a_dot_n: Q            # Absolute error for subnormals: (1+γ_{n-1})·n·a_mul
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
    op2_abs: Q,              # ‖|W_ℓ|‖_2
    inf_norm: Q,             # ‖W_ℓ‖_∞
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

    where:
        - S_layer = ‖|W_ℓ|‖_2 · (r_{ℓ-1} + D_{ℓ-1})  [uses FP activation bound!]
        - M_layer = ‖W_ℓ‖_∞ · (r_{ℓ-1} + D_{ℓ-1})
        - fp_activation_bound = r_{ℓ-1} + D_{ℓ-1} from triangle inequality

    Args:
        layer_idx: Layer index ℓ
        op2_abs: ‖|W_ℓ|‖_2
        inf_norm: ‖W_ℓ‖_∞
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
    S_layer = op2_abs * fp_activation_bound      # ‖|W|‖_2 · (r_{ℓ-1} + D_{ℓ-1})
    M_layer = inf_norm * fp_activation_bound     # ‖W‖_∞ · (r_{ℓ-1} + D_{ℓ-1})

    # Compute margin terms
    gamma = gamma_n(layer_width, u_q)
    a_dot_n = a_dot(layer_width, u_q, a_mul_q)

    # Apply margin to S_layer check
    S_with_margin = S_layer * (1 + gamma) + a_dot_n + bias_norm

    # Compute slack (positive = passes)
    s2 = Fmax_q - S_with_margin
    si = Fmax_q - M_layer

    ok = (s2 > 0) and (si > 0)

    return OverflowLayerStats(
        layer_idx=layer_idx,
        layer_width=layer_width,
        norm2_abs=op2_abs,
        norminf=inf_norm,
        S_layer=S_layer,
        M_layer=M_layer,
        gamma_n=gamma,
        a_dot_n=a_dot_n,
        S_with_margin=S_with_margin,
        slack_2=s2,
        slack_inf=si,
        ok=ok
    )


def certify_no_overflow_normwise(
    op2_abs: List[Q],        # [‖|W_0|‖_2, ..., ‖|W_{L-1}|‖_2]
    inf_norm: List[Q],       # [‖W_0‖_∞, ..., ‖W_{L-1}‖_∞]
    radii_prev: List[Q],     # [r_0, ..., r_{L-1}] where r_{l-1} bounds ‖z_{l-1}‖_2
    layer_widths: List[int], # [n_0, ..., n_{L-1}] where n_l is input dim of layer l
    fmt: FloatFormat,
    bias_norms: Optional[List[Q]] = None,  # [‖b_0‖_∞, ..., ‖b_{L-1}‖_∞] (if network has biases)
) -> OverflowReport:
    """Certify absence of overflow using layerwise worst-case checks.

    This implements the corrected Theorem 3.1 from the LaTeX writeup,
    backed by the Coq proof in overflow.v (corollary layerwise_simple_implies_finite).

    For each layer l, verifies:
        1. S_layer(l) · (1 + γ_{n_l}) + a_dot(n_l) + ‖b_l‖_∞ < F_max
        2. M_layer(l) < F_max

    where:
        - S_layer(l) = ‖|W_l|‖_2 · r_{l-1}  (worst-case sum of absolute products)
        - M_layer(l) = ‖W_l‖_∞ · r_{l-1}    (worst-case single product)
        - γ_{n_l} = (n_l·u) / (1 - n_l·u)  (relative error accumulation)
        - a_dot(n_l) = (1+γ_{n_l-1})·n_l·a_mul  (absolute error for subnormals)
        - ‖b_l‖_∞ = max_i |b_l[i]|  (max absolute bias, 0 if no biases)

    The margin terms (1+γ_n) and a_dot(n) account for roundoff error accumulation
    in floating-point dot products. For typical float32 networks:
        - γ_n ≈ 0.012% for n ≤ 1000
        - a_dot(n) ≈ 10^(-42) (negligible)

    Args:
        op2_abs: Operator 2-norm of |W_l| for each layer
        inf_norm: Infinity norm (max row sum) of W_l for each layer
        radii_prev: Radius bound r_{l-1} on ‖z_{l-1}‖_2 for each layer
        layer_widths: Input dimension n_l for each layer
        fmt: Floating-point format (contains F_max, u, denorm_min)
        bias_norms: Optional infinity norms of bias vectors (None if no biases)

    Returns:
        OverflowReport with per-layer statistics and overall pass/fail

    Raises:
        ValueError: If input lists have mismatched lengths or if n·u >= 1 for any layer
    """
    L = len(op2_abs)
    if not (len(inf_norm) == L and len(radii_prev) == L and len(layer_widths) == L):
        raise ValueError(
            f"Input lists must have same length. Got: "
            f"op2_abs={len(op2_abs)}, inf_norm={len(inf_norm)}, "
            f"radii_prev={len(radii_prev)}, layer_widths={len(layer_widths)}"
        )

    if bias_norms is not None and len(bias_norms) != L:
        raise ValueError(f"bias_norms length {len(bias_norms)} != {L}")

    Fmax_q = float_to_q(fmt.Fmax)
    u_q = float_to_q(fmt.u)
    a_mul_q = float_to_q(fmt.denorm_min) / 2

    layers: List[OverflowLayerStats] = []

    for l in range(L):
        a2 = op2_abs[l]
        ainf = inf_norm[l]
        rprev = radii_prev[l]
        n = layer_widths[l]
        b_norm = bias_norms[l] if bias_norms is not None else Q(0)

        # Compute bounds
        S_layer = a2 * rprev      # ‖|W|‖_2 · r_{l-1}
        M_layer = ainf * rprev    # ‖W‖_∞ · r_{l-1}

        # Compute margin terms
        gamma = gamma_n(n, u_q)
        a_dot_n = a_dot(n, u_q, a_mul_q)

        # Apply margin to S_layer check
        S_with_margin = S_layer * (1 + gamma) + a_dot_n + b_norm

        # Compute slack (positive = passes)
        s2 = Fmax_q - S_with_margin
        si = Fmax_q - M_layer

        ok = (s2 > 0) and (si > 0)

        stats = OverflowLayerStats(
            layer_idx=l,
            layer_width=n,
            norm2_abs=a2,
            norminf=ainf,
            S_layer=S_layer,
            M_layer=M_layer,
            gamma_n=gamma,
            a_dot_n=a_dot_n,
            S_with_margin=S_with_margin,
            slack_2=s2,
            slack_inf=si,
            ok=ok
        )
        layers.append(stats)

        if not ok:
            failure_type = "S_layer" if s2 <= 0 else "M_layer"
            return OverflowReport(layers, False, (l, failure_type))

    return OverflowReport(layers, True, None)


def certify_no_overflow_with_deviations(
    op2_abs: List[Q],
    inf_norm: List[Q],
    radii_prev: List[Q],          # [r_0, ..., r_{L-1}]
    deviation_bounds: List[Q],    # [D_{-1}, D_0, ..., D_{L-2}] (length L)
    layer_widths: List[int],
    fmt: FloatFormat,
    bias_norms: Optional[List[Q]] = None,
) -> OverflowReport:
    """Certify absence of overflow using deviation-aware bounds (SOUND VERSION).

    This is the corrected overflow checker that uses triangle inequality to bound
    FP activations: ‖ẑ_{ℓ-1}‖_2 ≤ ‖z_{ℓ-1}‖_2 + ‖d_{ℓ-1}‖_2 ≤ r_{ℓ-1} + D_{ℓ-1}

    For layer ℓ, uses fp_bound = r_{ℓ-1} + D_{ℓ-1} where:
        - r_{ℓ-1} bounds the exact activation ‖z_{ℓ-1}‖_2 (from radii.v)
        - D_{ℓ-1} bounds the deviation ‖ẑ_{ℓ-1} - z_{ℓ-1}‖_2 (from deviation recursion)

    Args:
        op2_abs: [‖|W_0|‖_2, ..., ‖|W_{L-1}|‖_2]
        inf_norm: [‖W_0‖_∞, ..., ‖W_{L-1}‖_∞]
        radii_prev: [r_0, ..., r_{L-1}] (exact radius bounds)
        deviation_bounds: [D_{-1}, D_0, ..., D_{L-2}] where:
            - deviation_bounds[0] = D_{-1} = 0 for layer 0
            - deviation_bounds[ℓ] = D_{ℓ-1} for layer ℓ > 0
        layer_widths: [n_0, ..., n_{L-1}]
        fmt: Floating-point format
        bias_norms: Optional [‖b_0‖_∞, ..., ‖b_{L-1}‖_∞]

    Returns:
        OverflowReport with per-layer statistics and overall pass/fail

    Raises:
        ValueError: If input lists have mismatched lengths or n·u ≥ 1
    """
    L = len(op2_abs)
    if not (len(inf_norm) == L and len(radii_prev) == L and
            len(deviation_bounds) == L and len(layer_widths) == L):
        raise ValueError(
            f"Input lists must have same length. Got: "
            f"op2_abs={len(op2_abs)}, inf_norm={len(inf_norm)}, "
            f"radii_prev={len(radii_prev)}, deviation_bounds={len(deviation_bounds)}, "
            f"layer_widths={len(layer_widths)}"
        )

    if bias_norms is not None and len(bias_norms) != L:
        raise ValueError(f"bias_norms length {len(bias_norms)} != {L}")

    layers: List[OverflowLayerStats] = []

    for l in range(L):
        rprev = radii_prev[l]
        D_prev = deviation_bounds[l]
        b_norm = bias_norms[l] if bias_norms is not None else Q(0)

        # Use triangle inequality bound: ‖ẑ_{ℓ-1}‖_2 ≤ r_{ℓ-1} + D_{ℓ-1}
        fp_bound = rprev + D_prev

        # Check overflow for this layer
        stats = check_overflow_single_layer(
            layer_idx=l,
            op2_abs=op2_abs[l],
            inf_norm=inf_norm[l],
            fp_activation_bound=fp_bound,
            layer_width=layer_widths[l],
            fmt=fmt,
            bias_norm=b_norm
        )

        layers.append(stats)

        if not stats.ok:
            failure_type = "S_layer" if stats.slack_2 <= 0 else "M_layer"
            return OverflowReport(layers, False, (l, failure_type))

    return OverflowReport(layers, True, None)
