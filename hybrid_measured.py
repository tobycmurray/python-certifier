"""Helper functions for the hybrid-only and hybrid-measured certification modes.

Shared by BOTH hybrid modes (the hybrid centre deviation D^hybrid(x,0)):
  - compute_D_hi_all_layers     : theoretical fp64 deviation bound at each hidden layer
  - compute_D_hybrid_center     : D^hybrid(x,0) = measured_diff + D^hi_{L-2}
  - compute_measured_center_diff: ||z_fp - z_hi|| from the Keras forward passes

HYBRID-MEASURED ONLY (the optional, marginal appendix mode -- measured E_ball
radii; see writeup). Kept because its incremental TCB is small (~6 pure-rational
functions, no new dependency) and its soundness is the Coq theorem
mode_B_hybrid_robust_nlayer:
  - compute_z_hi_norms, compute_cumulative_lipschitz,
    compute_r_meas[_all_layers], compute_D_meas_with_input
  - build_measured_comp_inputs  : high-level entry assembling the above
"""

from typing import List, Dict, Tuple, Optional

import numpy as np

from arithmetic import Q, float_to_q, sqrt_upper_bound, round_up
from linear_algebra import Matrix, Vector, dims, l2_norm_upper_bound_vec
from formats import FloatFormat, get_float_format
from deviation import compute_layer_deviation_params, compute_deviation_bound


def compute_D_hi_all_layers(
    network: List[Matrix],
    op2_norms: List[Q],
    op2_abs_norms: List[Q],
    x: Vector,
    sqrt_m_dict: Dict[int, Q],
    fmt_hi: FloatFormat,
    num_layers: int,
    bias_l2_norms: Optional[List[Q]] = None,
) -> List[Q]:
    """Compute theoretical fp64 deviation bound at the first *num_layers* layers.

    Uses the standard deviation recursion with fp64 format parameters.
    Since fp64 has tiny unit roundoff (u ≈ 1.1e-16) the results are negligible
    but non-zero, and are needed for the soundness of D^hybrid.
    Bias-aware exactly as robust_certifier.radii / compute_D_meas_with_input:
    the radii gain ||b_ℓ||_2 and each β_ℓ gains u·||b_ℓ||_2 via bias_l2_norm.

    Args:
        network:      Weight matrices (length >= num_layers).
        op2_norms:    [||W_0||_2, ..., ||W_{L-1}||_2]
        op2_abs_norms:[|||W_0|||_2, ..., |||W_{L-1}|||_2]
        x:            Input vector (centre point, epsilon = 0).
        sqrt_m_dict:  Precomputed sqrt(m) values.
        fmt_hi:       High-precision format (float64).
        num_layers:   How many layers to recurse through (pass H = L-1 to stop
                      at the final hidden layer and avoid the output layer).
        bias_l2_norms:[||b_0||_2, ..., ||b_{L-1}||_2] (None without biases).

    Returns:
        [D^hi_0, ..., D^hi_{num_layers-1}]
    """
    # Radii at center (epsilon = 0): r_ell = ||W_{ell-1}|| * r_{ell-1} + ||b_{ell-1}||_2
    r0 = l2_norm_upper_bound_vec(x)
    r_list = [r0]
    r = r0
    for ell in range(1, num_layers):
        r = op2_norms[ell - 1] * r
        if bias_l2_norms is not None and bias_l2_norms[ell - 1] is not None:
            r = r + bias_l2_norms[ell - 1]
        r_list.append(r)

    D_hi: List[Q] = []
    D_prev = Q(0)

    for ell in range(num_layers):
        m_ell, n_ell = dims(network[ell])
        sqrt_m = sqrt_m_dict.get(m_ell, sqrt_upper_bound(Q(m_ell)))

        params = compute_layer_deviation_params(
            layer_idx=ell,
            op2_norm=op2_norms[ell],
            op2_abs_norm=op2_abs_norms[ell],
            r_prev=r_list[ell],
            layer_width=n_ell,
            output_dim=m_ell,
            sqrt_m=sqrt_m,
            fmt=fmt_hi,
            bias_l2_norm=bias_l2_norms[ell] if bias_l2_norms is not None else Q(0),
        )
        D_ell = compute_deviation_bound(D_prev, params)
        D_hi.append(D_ell)
        D_prev = D_ell

    return D_hi


def compute_D_hybrid_center(measured_center_diff: Q, D_hi_final: Q) -> Q:
    """D^hybrid(x,0) = ||ẑ_{L-2}(x) - ẑ^hi_{L-2}(x)||_2 + D^hi_{L-2}(x,0).

    Args:
        measured_center_diff: ||z^fp_{L-2}(x) - z^hi_{L-2}(x)||_2 (from
                              compute_measured_center_diff on the Keras
                              fp64/target activations).
        D_hi_final:           D^hi_{L-2}(x,0) from compute_D_hi_all_layers[-1].

    Returns:
        D^hybrid(x,0) — a sound upper bound on ||z^fp_{L-2}(x) - z^exact_{L-2}(x)||_2.
    """
    return measured_center_diff + D_hi_final


# ---------------------------------------------------------------------------
# Measured-radii machinery for the hybrid-measured E_ball bound.
#
# These reproduce the measured-radii deviation analysis (LaTeX Lemma 11.2,
# Coq theorem mode_B_hybrid_robust_nlayer in coq/hybrid_certification.v): the
# ball radius at each layer is bounded using the *measured* fp64 activation norm
# plus Lipschitz growth over the ball, rather than the theoretical spectral-norm
# product, giving a much tighter E_ball term (especially deep in the network).
# ---------------------------------------------------------------------------

def _vec_to_q(z: np.ndarray) -> Vector:
    """Exact rational copy of a float vector (float_to_q is lossless)."""
    return [float_to_q(float(v)) for v in z]


def compute_z_hi_norms(z_hi: List[np.ndarray]) -> List[Q]:
    """L2 norms of the fp64 activations at each layer, as sound rational upper bounds.

    ||ẑ^hi_ℓ(x)||_2 for ℓ = 0 .. L-1, computed in exact rational arithmetic
    (exact sum of squares + sqrt_upper_bound), not via np.linalg.norm.
    """
    return [l2_norm_upper_bound_vec(_vec_to_q(z)) for z in z_hi]


def compute_cumulative_lipschitz(op2_norms: List[Q]) -> List[Q]:
    """Cumulative Lipschitz constants Lip_ℓ = ∏_{k=0}^{ℓ-1} ||W_k||_2.

    Returns a list of length L+1: [Lip_0=1, Lip_1, ..., Lip_L].
    """
    Lip = [Q(1)]  # Lip_0 = 1 (identity at input)
    cumulative = Q(1)
    for ell in range(len(op2_norms)):
        cumulative = cumulative * op2_norms[ell]
        Lip.append(cumulative)
    return Lip


def compute_measured_center_diff(z_fp: List[np.ndarray], z_hi: List[np.ndarray]) -> Q:
    """||ẑ_{L-2}(x) - ẑ^hi_{L-2}(x)||_2 at the final hidden layer (index L-2).

    Exact rational arithmetic: both activation vectors are converted losslessly
    (float_to_q), subtracted in Q, and the norm is l2_norm_upper_bound_vec
    (exact sum of squares + sqrt_upper_bound) -- a sound upper bound.
    """
    L = len(z_fp)
    if L < 2:
        return Q(0)
    z_fp_final = _vec_to_q(z_fp[L - 2])
    z_hi_final = _vec_to_q(z_hi[L - 2])
    assert len(z_fp_final) == len(z_hi_final)
    diff = [a - b for a, b in zip(z_fp_final, z_hi_final)]
    return l2_norm_upper_bound_vec(diff)


def compute_r_meas(z_hi_norm: Q, Lip_ell: Q, epsilon: Q, D_hi_ell: Q) -> Q:
    """Measured radius at one layer: r^meas_ℓ = ||ẑ^hi_ℓ(x)|| + Lip_ℓ·ε + D^hi_ℓ(x,0)."""
    return z_hi_norm + Lip_ell * epsilon + D_hi_ell


def compute_r_meas_all_layers(
    z_hi_norms: List[Q],
    Lip_cumulative: List[Q],
    epsilon: Q,
    D_hi: List[Q],
) -> List[Q]:
    """Measured radii [r^meas_0, ..., r^meas_{L-1}], each bounding ||z_ℓ(x')|| over the ball.

    For z_ℓ (output of layer ℓ) the Lipschitz constant from the input is
    Lip_cumulative[ℓ+1] = ∏_{k=0}^{ℓ} ||W_k||_2.
    """
    r_meas = []
    for ell in range(len(z_hi_norms)):
        r_meas.append(compute_r_meas(z_hi_norms[ell], Lip_cumulative[ell + 1], epsilon, D_hi[ell]))
    return r_meas


def compute_D_meas_with_input(
    network: List[Matrix],
    op2_norms: List[Q],
    op2_abs_norms: List[Q],
    input_radius: Q,
    r_meas: List[Q],
    sqrt_m_dict: Dict[int, Q],
    fmt: FloatFormat,
    num_hidden_layers: Optional[int] = None,
    bias_l2_norms: Optional[List[Q]] = None,
) -> Tuple[Q, List[Q]]:
    """D^meas over the ε-ball, using the deviation recursion driven by measured radii.

    D_ℓ^meas = α_ℓ · D_{ℓ-1}^meas + β_ℓ(r_{ℓ-1}^meas), where layer ℓ uses the input
    radius r_{ℓ-1}: for ℓ=0 that is input_radius = ||x||+ε, otherwise r_meas[ℓ-1].
    The β_ℓ also includes the u·||b_ℓ|| bias term (matching the Coq
    compute_beta_with_type), via bias_l2_norms.

    Returns (D^meas_{H-1}, [D^meas_0, ..., D^meas_{H-1}]) where H = num_hidden_layers
    (default: L-1, i.e. through the hidden layers only).
    """
    L = len(network)
    H = num_hidden_layers if num_hidden_layers is not None else L - 1
    D_prev = Q(0)
    D_all: List[Q] = []
    for ell in range(H):
        m_ell, n_ell = dims(network[ell])
        sqrt_m = sqrt_m_dict.get(m_ell, sqrt_upper_bound(Q(m_ell)))
        r_prev = input_radius if ell == 0 else r_meas[ell - 1]
        params = compute_layer_deviation_params(
            layer_idx=ell,
            op2_norm=op2_norms[ell],
            op2_abs_norm=op2_abs_norms[ell],
            r_prev=r_prev,
            layer_width=n_ell,
            output_dim=m_ell,
            sqrt_m=sqrt_m,
            fmt=fmt,
            bias_l2_norm=bias_l2_norms[ell] if bias_l2_norms is not None else Q(0),
        )
        D_ell = compute_deviation_bound(D_prev, params)
        D_all.append(D_ell)
        D_prev = D_ell
    return D_prev, D_all


def build_measured_comp_inputs(
    network: List[Matrix],
    x: Vector,
    epsilon: Q,
    op2_norms: List[Q],
    op2_abs_norms: List[Q],
    z_hi: List[np.ndarray],
    z_fp: List[np.ndarray],
    sqrt_m_dict: Dict[int, Q],
    fmt: FloatFormat,
    H: int,
    bias_l2_norms: Optional[List[Q]] = None,
) -> Tuple[Q, Q, Q, Q]:
    """Compute the four scalars the certifier needs to build the measured comp_ctr/comp_ball.

    Returns (D_hybrid_center, r_Lm1_center, D_meas_ball, r_Lm1_ball) where:
      - D_hybrid_center = ||ẑ - ẑ^hi||_{L-2} + D^hi_{L-2}      (E_ctr deviation)
      - r_Lm1_center    = r^meas_{L-2} at ε=0                  (E_ctr final-layer radius)
      - D_meas_ball     = D^meas_{H-1} over the ε-ball         (E_ball deviation)
      - r_Lm1_ball      = r^meas_{L-2} over the ε-ball         (E_ball final-layer radius)

    With these, E_ctr = α_L·D_hybrid_center + β_L(r_Lm1_center) and
    E_ball = α_L·D_meas_ball + β_L(r_Lm1_ball), matching compute_final_pair_params.
    """
    fmt_hi = get_float_format("float64")
    z_hi_norms = compute_z_hi_norms(z_hi)
    Lip_cumulative = compute_cumulative_lipschitz(op2_norms)
    # D^hi over all layers (fp64; negligible but non-zero, needed for soundness of r^meas).
    D_hi = compute_D_hi_all_layers(
        network, op2_norms, op2_abs_norms, x, sqrt_m_dict, fmt_hi, len(network),
        bias_l2_norms=bias_l2_norms,
    )
    measured_center_diff = compute_measured_center_diff(z_fp, z_hi)

    r_meas_center = compute_r_meas_all_layers(z_hi_norms, Lip_cumulative, Q(0), D_hi)
    r_meas_ball = compute_r_meas_all_layers(z_hi_norms, Lip_cumulative, epsilon, D_hi)

    D_hi_final = D_hi[H - 1] if H > 0 else Q(0)
    D_hybrid_center = compute_D_hybrid_center(measured_center_diff, D_hi_final)

    input_radius = l2_norm_upper_bound_vec(x) + epsilon
    D_meas_ball, _ = compute_D_meas_with_input(
        network, op2_norms, op2_abs_norms, input_radius, r_meas_ball, sqrt_m_dict, fmt,
        bias_l2_norms=bias_l2_norms,
    )

    r_Lm1_center = r_meas_center[H - 1] if H > 0 else r_meas_center[-1]
    r_Lm1_ball = r_meas_ball[H - 1] if H > 0 else r_meas_ball[-1]
    # Round the four measured scalars UP before they feed the per-class-pair final
    # step. The deviation term D_hi makes r_meas an exact rational with a ~233k-bit
    # denominator (float64 gamma_n compounded over layers, unrounded); multiplying
    # it across all class pairs costs ~0.8s/call. round_up (the certifier's 16-dp
    # "Dafny-accurate" rounding) collapses it to a ~53-bit rational. Sound by
    # monotonicity: round_up(v) >= v, and the per-pair RHS E = alpha*D + beta(r) is
    # monotone increasing in D and r, so the verdict can only become MORE conservative
    # (never wrongly certify). ~75x faster per call.
    return (round_up(D_hybrid_center), round_up(r_Lm1_center),
            round_up(D_meas_ball), round_up(r_Lm1_ball))
