"""Helper functions for the hybrid-only certification mode.

Provides:
  - compute_D_hi_all_layers   : theoretical fp64 deviation bound at each hidden layer
  - compute_D_hybrid_center   : D^hybrid(x,0) = measured_diff + D^hi_{L-2}
"""

from typing import List, Dict

from arithmetic import Q, float_to_q, sqrt_upper_bound
from linear_algebra import Matrix, Vector, dims, l2_norm_upper_bound_vec
from formats import FloatFormat
from deviation import compute_layer_deviation_params, compute_deviation_bound


def compute_D_hi_all_layers(
    network: List[Matrix],
    op2_norms: List[Q],
    op2_abs_norms: List[Q],
    x: Vector,
    sqrt_m_dict: Dict[int, Q],
    fmt_hi: FloatFormat,
    num_layers: int,
) -> List[Q]:
    """Compute theoretical fp64 deviation bound at the first *num_layers* layers.

    Uses the standard deviation recursion with fp64 format parameters.
    Since fp64 has tiny unit roundoff (u ≈ 1.1e-16) the results are negligible
    but non-zero, and are needed for the soundness of D^hybrid.

    Args:
        network:      Weight matrices (length >= num_layers).
        op2_norms:    [||W_0||_2, ..., ||W_{L-1}||_2]
        op2_abs_norms:[|||W_0|||_2, ..., |||W_{L-1}|||_2]
        x:            Input vector (centre point, epsilon = 0).
        sqrt_m_dict:  Precomputed sqrt(m) values.
        fmt_hi:       High-precision format (float64).
        num_layers:   How many layers to recurse through (pass H = L-1 to stop
                      at the final hidden layer and avoid the output layer).

    Returns:
        [D^hi_0, ..., D^hi_{num_layers-1}]
    """
    # Radii at center (epsilon = 0): r_ell = prod_{k<ell} ||W_k|| * ||x||
    r0 = l2_norm_upper_bound_vec(x)
    r_list = [r0]
    r = r0
    for ell in range(1, num_layers):
        r = op2_norms[ell - 1] * r
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
        )
        D_ell = compute_deviation_bound(D_prev, params)
        D_hi.append(D_ell)
        D_prev = D_ell

    return D_hi


def compute_D_hybrid_center(measured_center_diff: Q, D_hi_final: Q) -> Q:
    """D^hybrid(x,0) = ||ẑ_{L-2}(x) - ẑ^hi_{L-2}(x)||_2 + D^hi_{L-2}(x,0).

    Args:
        measured_center_diff: ||z^fp_{L-2}(x) - z^hi_{L-2}(x)||_2 (from
                              measure_center_diff_norm in nn.py).
        D_hi_final:           D^hi_{L-2}(x,0) from compute_D_hi_all_layers[-1].

    Returns:
        D^hybrid(x,0) — a sound upper bound on ||z^fp_{L-2}(x) - z^exact_{L-2}(x)||_2.
    """
    return measured_center_diff + D_hi_final
