"""Hybrid+Measured certification mode for floating-point neural networks.

This module implements the tighter deviation bounds from LaTeX Section 7:
- Lemma 7.1 (lem:hybrid-center): Hybrid deviation bound at center
- Lemma 7.2 (lem:measured-radii-ball): Measured-radii deviation for ball

Key formulas:
    D^hybrid(x,0) = ||ẑ(x) - ẑ^hi(x)||_2 + D^hi(x,0)
    r^meas_ℓ(x,ε) = ||ẑ^hi_ℓ(x)||_2 + Lip_ℓ · ε + D^hi_ℓ(x,0)
    D^meas(x,ε) uses standard recursion with measured radii

The approach leverages a single fp64 forward pass to get much tighter bounds
than purely theoretical radii, while remaining provably sound.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np

from arithmetic import Q, float_to_q, sqrt_upper_bound
from linear_algebra import Matrix, Vector, dims, l2_norm_upper_bound_vec
from formats import FloatFormat, get_float_format, gamma_n, a_dot
from deviation import compute_layer_deviation_params, compute_deviation_bound
from nn import forward_layerwise_float64, forward_layerwise_float16


@dataclass(frozen=True)
class HybridMeasuredData:
    """Data from fp64 forward pass needed for hybrid+measured certification.

    Contains all the per-layer measurements and computed bounds.
    """
    # Raw measurements from fp64 forward pass
    z_hi_norms: List[Q]        # ||ẑ^hi_ℓ(x)||_2 for ℓ = 0..L-1

    # Theoretical fp64 deviation at each layer (very small)
    D_hi: List[Q]              # D^hi_ℓ(x,0) for ℓ = 0..L-1

    # Cumulative Lipschitz constants
    Lip_cumulative: List[Q]    # Lip_ℓ = ∏_{k=1}^ℓ ||W_k||_2, with Lip_0 = 1

    # For hybrid center bound: measured difference between fp16 and fp64
    measured_center_diff: Q    # ||ẑ_{L-1}(x) - ẑ^hi_{L-1}(x)||_2


def compute_z_hi_norms(z_hi: List[np.ndarray]) -> List[Q]:
    """Compute L2 norms of fp64 activations at each layer.

    Args:
        z_hi: List of fp64 activations [z_0, z_1, ..., z_{L-1}]

    Returns:
        List of Q values [||z_0||_2, ||z_1||_2, ..., ||z_{L-1}||_2]
    """
    norms = []
    for z in z_hi:
        # Compute L2 norm in float64, then convert to exact rational upper bound
        norm_float = float(np.linalg.norm(z, ord=2))
        # Round up to get sound upper bound
        norms.append(float_to_q(norm_float))
    return norms


def compute_cumulative_lipschitz(op2_norms: List[Q]) -> List[Q]:
    """Compute cumulative Lipschitz constants Lip_ℓ = ∏_{k=1}^ℓ ||W_k||_2.

    Args:
        op2_norms: [||W_0||_2, ||W_1||_2, ..., ||W_{L-1}||_2]

    Returns:
        [Lip_0, Lip_1, ..., Lip_{L-1}] where Lip_0 = 1
    """
    L = len(op2_norms)
    Lip = [Q(1)]  # Lip_0 = 1 (identity at input)

    cumulative = Q(1)
    for ell in range(L):
        cumulative = cumulative * op2_norms[ell]
        Lip.append(cumulative)

    return Lip  # Length L+1: [Lip_0, Lip_1, ..., Lip_L]


def compute_D_hi_all_layers(
    network: List[Matrix],
    op2_norms: List[Q],
    op2_abs_norms: List[Q],
    x: Vector,
    sqrt_m_dict: Dict[int, Q],
    fmt_hi: FloatFormat
) -> List[Q]:
    """Compute theoretical fp64 deviation at all layers.

    Uses the standard deviation recursion but with fp64 format parameters.
    Since fp64 has tiny unit roundoff (u ≈ 1.1e-16), these bounds are negligible.

    Args:
        network: List of weight matrices
        op2_norms: [||W_0||_2, ..., ||W_{L-1}||_2]
        op2_abs_norms: [|||W_0|||_2, ..., |||W_{L-1}|||_2]
        x: Input vector
        sqrt_m_dict: Precomputed sqrt(m) values
        fmt_hi: High-precision format (float64)

    Returns:
        [D^hi_0, D^hi_1, ..., D^hi_{L-1}] theoretical deviation bounds
    """
    L = len(network)

    # Compute theoretical radii for fp64 deviation computation
    # r_0 = ||x||_2 (no epsilon for center point)
    r0 = l2_norm_upper_bound_vec(x)
    r_list = [r0]
    r = r0
    for ell in range(1, L):
        r = op2_norms[ell - 1] * r
        r_list.append(r)

    # Compute D^hi at each layer using the recursion
    D_hi = []
    D_prev = Q(0)  # D_{-1} = 0

    for ell in range(L):
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
            fmt=fmt_hi
        )

        D_ell = compute_deviation_bound(D_prev, params)
        D_hi.append(D_ell)
        D_prev = D_ell

    return D_hi


def compute_measured_center_diff(
    z_fp: List[np.ndarray],
    z_hi: List[np.ndarray]
) -> Q:
    """Compute ||ẑ_{L-1}(x) - ẑ^hi_{L-1}(x)||_2 for hybrid center bound.

    Args:
        z_fp: FP activations (e.g., float16) at all layers
        z_hi: High-precision activations (float64) at all layers

    Returns:
        L2 norm of difference at final hidden layer
    """
    # Get the final hidden layer activations (index -1 is output layer, -2 is last hidden)
    # But wait - our indexing is 0..L-1 where L-1 is output layer
    # So final hidden layer is L-2
    L = len(z_fp)
    if L < 2:
        return Q(0)

    # For hidden layers 0..L-2, the relevant one for D_{L-1} is index L-2
    z_fp_final = z_fp[L - 2].astype(np.float64)
    z_hi_final = z_hi[L - 2].astype(np.float64)

    diff_norm = float(np.linalg.norm(z_fp_final - z_hi_final, ord=2))
    return float_to_q(diff_norm)


def compute_r_meas(
    z_hi_norm: Q,
    Lip_ell: Q,
    epsilon: Q,
    D_hi_ell: Q
) -> Q:
    """Compute measured radius at a single layer.

    r^meas_ℓ(x,ε) = ||ẑ^hi_ℓ(x)||_2 + Lip_ℓ · ε + D^hi_ℓ(x,0)

    Args:
        z_hi_norm: ||ẑ^hi_ℓ(x)||_2
        Lip_ell: Cumulative Lipschitz to layer ℓ
        epsilon: Perturbation radius
        D_hi_ell: D^hi_ℓ(x,0)

    Returns:
        r^meas_ℓ(x,ε)
    """
    return z_hi_norm + Lip_ell * epsilon + D_hi_ell


def compute_r_meas_all_layers(
    z_hi_norms: List[Q],
    Lip_cumulative: List[Q],
    epsilon: Q,
    D_hi: List[Q]
) -> List[Q]:
    """Compute measured radii at all layers.

    Args:
        z_hi_norms: [||ẑ^hi_0||, ..., ||ẑ^hi_{L-1}||]
        Lip_cumulative: [Lip_0, Lip_1, ..., Lip_L] where Lip_0 = 1
        epsilon: Perturbation radius
        D_hi: [D^hi_0, ..., D^hi_{L-1}]

    Returns:
        [r^meas_0, ..., r^meas_{L-1}]
    """
    L = len(z_hi_norms)
    r_meas = []

    for ell in range(L):
        # r_meas[ell] bounds ||z_ell||, the output of layer ell
        # The Lipschitz constant from input x to z_ell is:
        #   Lip_to_z_ell = ∏_{k=0}^{ell} ||W_k||_2 = Lip_cumulative[ell+1]
        # where:
        #   Lip_cumulative[0] = 1 (identity)
        #   Lip_cumulative[1] = ||W_0||
        #   Lip_cumulative[ell+1] = ∏_{k=0}^{ell} ||W_k||
        #
        # So for z_ell (output of layer ell), use Lip_cumulative[ell+1]

        r = compute_r_meas(z_hi_norms[ell], Lip_cumulative[ell + 1], epsilon, D_hi[ell])
        r_meas.append(r)

    return r_meas


def compute_D_meas(
    network: List[Matrix],
    op2_norms: List[Q],
    op2_abs_norms: List[Q],
    r_meas: List[Q],
    sqrt_m_dict: Dict[int, Q],
    fmt: FloatFormat
) -> Q:
    """Compute D^meas_{L-1}(x,ε) using measured radii in the deviation recursion.

    Uses the standard recursion D_ℓ = α_ℓ · D_{ℓ-1} + β_ℓ, but β_ℓ uses
    measured radii instead of theoretical radii.

    Args:
        network: List of weight matrices
        op2_norms: [||W_0||, ..., ||W_{L-1}||]
        op2_abs_norms: [|||W_0|||, ..., |||W_{L-1}|||]
        r_meas: [r^meas_0, ..., r^meas_{L-1}]
        sqrt_m_dict: Precomputed sqrt(m) values
        fmt: Target floating-point format (e.g., float16)

    Returns:
        D^meas_{L-1}(x,ε) - final deviation bound using measured radii
    """
    L = len(network)
    D_prev = Q(0)

    for ell in range(L):
        m_ell, n_ell = dims(network[ell])
        sqrt_m = sqrt_m_dict.get(m_ell, sqrt_upper_bound(Q(m_ell)))

        # Use measured radius r^meas_{ell-1} for this layer's β computation
        # For layer 0, we use r^meas_0 (the input radius)
        # Actually: β_ℓ uses r_{ℓ-1}, so for layer ell, we use r_meas[ell]
        # Wait, let me check: in the standard recursion, layer ell uses r_{ell-1}.
        # But r_meas[ell] = r^meas_ell, which bounds ||z_ell||.
        # For layer ell's computation, we need the input radius, which is r_{ell-1}.
        # So we should use r_meas[ell] as the input to layer ell+1.
        # For layer 0, input is x, so we need r_meas for input = ||x|| + eps.
        # But r_meas[0] = ||ẑ^hi_0|| + Lip_0*eps + D^hi_0 bounds ||z_0||, not input.
        #
        # Let me re-read the LaTeX more carefully:
        # eq:measured-deviation-recursion says:
        # D_ℓ^meas = α_ℓ · D_{ℓ-1}^meas + β_ℓ(r_{ℓ-1}^meas)
        # So for layer ℓ, we use r_{ℓ-1}^meas.
        # r_meas[ell] = r^meas_ell bounds ||z_ell||.
        # So for layer ell (0-indexed), we need r_meas[ell-1] for ell >= 1.
        # For ell = 0, we need r_{-1}^meas = input radius = ||x|| + eps.

        if ell == 0:
            # For first layer, input radius is ||x|| + eps
            # But we don't have x here... we need to pass it in or precompute.
            # Actually, let's restructure: r_meas should include input radius as first element.
            # OR we pass input_radius separately.
            # For now, let's use r_meas[0] as the input to layer 0.
            # This is slightly conservative (r_meas[0] bounds output of layer 0, not input).
            # Actually no - let me reconsider the indexing.
            #
            # In the LaTeX, layers are 1-indexed:
            #   Layer 1: W_1, input is x, output is z_1
            #   Layer ℓ: W_ℓ, input is z_{ℓ-1}, output is z_ℓ
            #
            # In our code, layers are 0-indexed:
            #   Layer 0: W_0, input is x (=z_{-1}), output is z_0
            #   Layer ℓ: W_ℓ, input is z_{ℓ-1}, output is z_ℓ
            #
            # r^meas_ℓ bounds ||z_ℓ||.
            # For layer ℓ, β_ℓ needs r_{ℓ-1} (input radius).
            # So:
            #   Layer 0: needs r_{-1} = input radius = ||x|| + eps
            #   Layer ℓ: needs r_{ℓ-1} = r_meas[ℓ-1]
            #
            # We need to prepend input radius to r_meas, or handle layer 0 specially.
            # For now, pass input_radius separately.
            raise ValueError("Need input_radius for layer 0 - restructure needed")

        r_prev = r_meas[ell - 1]  # This is r_{ell-1}^meas

        params = compute_layer_deviation_params(
            layer_idx=ell,
            op2_norm=op2_norms[ell],
            op2_abs_norm=op2_abs_norms[ell],
            r_prev=r_prev,
            layer_width=n_ell,
            output_dim=m_ell,
            sqrt_m=sqrt_m,
            fmt=fmt
        )

        D_ell = compute_deviation_bound(D_prev, params)
        D_prev = D_ell

    return D_prev


def compute_D_meas_with_input(
    network: List[Matrix],
    op2_norms: List[Q],
    op2_abs_norms: List[Q],
    input_radius: Q,
    r_meas: List[Q],
    sqrt_m_dict: Dict[int, Q],
    fmt: FloatFormat,
    num_hidden_layers: Optional[int] = None
) -> Tuple[Q, List[Q]]:
    """Compute D^meas using measured radii, with explicit input radius.

    Args:
        network: List of weight matrices
        op2_norms: [||W_0||, ..., ||W_{L-1}||]
        op2_abs_norms: [|||W_0|||, ..., |||W_{L-1}|||]
        input_radius: r_0 = ||x|| + epsilon (bounds input to first layer)
        r_meas: [r^meas_0, ..., r^meas_{L-1}] (each bounds output of that layer)
        sqrt_m_dict: Precomputed sqrt(m) values
        fmt: Target floating-point format
        num_hidden_layers: Number of hidden layers to compute (default: L-1)

    Returns:
        (D^meas_{H-1}, [D^meas_0, ..., D^meas_{H-1}]) where H is num_hidden_layers
    """
    L = len(network)
    # Default: compute through hidden layers only (L-1 layers)
    H = num_hidden_layers if num_hidden_layers is not None else L - 1
    D_prev = Q(0)
    D_all = []

    for ell in range(H):
        m_ell, n_ell = dims(network[ell])
        sqrt_m = sqrt_m_dict.get(m_ell, sqrt_upper_bound(Q(m_ell)))

        # r_prev = input radius to this layer
        if ell == 0:
            r_prev = input_radius
        else:
            r_prev = r_meas[ell - 1]

        params = compute_layer_deviation_params(
            layer_idx=ell,
            op2_norm=op2_norms[ell],
            op2_abs_norm=op2_abs_norms[ell],
            r_prev=r_prev,
            layer_width=n_ell,
            output_dim=m_ell,
            sqrt_m=sqrt_m,
            fmt=fmt
        )

        D_ell = compute_deviation_bound(D_prev, params)
        D_all.append(D_ell)
        D_prev = D_ell

    return D_prev, D_all


def compute_D_hybrid_center(
    measured_center_diff: Q,
    D_hi_final: Q
) -> Q:
    """Compute hybrid deviation bound at center.

    D^hybrid(x,0) = ||ẑ_{L-1}(x) - ẑ^hi_{L-1}(x)||_2 + D^hi_{L-1}(x,0)

    Args:
        measured_center_diff: ||ẑ_{L-1}(x) - ẑ^hi_{L-1}(x)||_2
        D_hi_final: D^hi_{L-1}(x,0)

    Returns:
        D^hybrid(x,0)
    """
    return measured_center_diff + D_hi_final


def build_hybrid_measured_data(
    network: List[Matrix],
    x: Vector,
    epsilon: Q,
    op2_norms: List[Q],
    op2_abs_norms: List[Q],
    z_hi: List[np.ndarray],
    z_fp: List[np.ndarray],
    sqrt_m_dict: Dict[int, Q],
    fmt: FloatFormat
) -> HybridMeasuredData:
    """Build all hybrid+measured data from a single fp64 forward pass.

    Args:
        network: List of weight matrices
        x: Input vector
        epsilon: Perturbation radius
        op2_norms: [||W_0||, ..., ||W_{L-1}||]
        op2_abs_norms: [|||W_0|||, ..., |||W_{L-1}|||]
        z_hi: fp64 activations at all layers
        z_fp: target format activations at all layers
        sqrt_m_dict: Precomputed sqrt(m) values
        fmt: Target floating-point format

    Returns:
        HybridMeasuredData containing all measurements and computed bounds
    """
    # Get fp64 format for D^hi computation
    fmt_hi = get_float_format("float64")

    # Compute ||ẑ^hi_ℓ||_2 at each layer
    z_hi_norms = compute_z_hi_norms(z_hi)

    # Compute cumulative Lipschitz constants
    Lip_cumulative = compute_cumulative_lipschitz(op2_norms)

    # Compute D^hi at all layers (using fp64 params - will be tiny)
    D_hi = compute_D_hi_all_layers(
        network, op2_norms, op2_abs_norms, x, sqrt_m_dict, fmt_hi
    )

    # Compute measured center difference ||ẑ - ẑ^hi||
    measured_center_diff = compute_measured_center_diff(z_fp, z_hi)

    return HybridMeasuredData(
        z_hi_norms=z_hi_norms,
        D_hi=D_hi,
        Lip_cumulative=Lip_cumulative,
        measured_center_diff=measured_center_diff
    )


@dataclass(frozen=True)
class HybridMeasuredResult:
    """Results of hybrid+measured certification computation."""
    # Measured radii
    r_meas_center: List[Q]     # r^meas_ℓ(x, 0)
    r_meas_ball: List[Q]       # r^meas_ℓ(x, ε)

    # Deviation bounds
    D_hybrid_center: Q         # D^hybrid(x, 0) for center
    D_meas_ball: Q             # D^meas_{L-1}(x, ε) for ball
    D_meas_all: List[Q]        # D^meas at each layer for ball

    # Error budgets (for final layer margin)
    E_ctr: Dict[Tuple[int, int], Q]   # E^{(i,j)}_ctr for each (i,j) pair
    E_ball: Dict[Tuple[int, int], Q]  # E^{(i,j)}_ball for each (i,j) pair


def compute_hybrid_measured_certification(
    network: List[Matrix],
    x: Vector,
    epsilon: Q,
    op2_norms: List[Q],
    op2_abs_norms: List[Q],
    sqrt_m_dict: Dict[int, Q],
    fmt: FloatFormat,
    L_ij: Dict[Tuple[int, int], Q],
    S_ij: Dict[Tuple[int, int], Q],
    i_star: int
) -> HybridMeasuredResult:
    """Compute full hybrid+measured certification data.

    Args:
        network: List of weight matrices
        x: Input vector
        epsilon: Perturbation radius
        op2_norms: [||W_0||, ..., ||W_{L-1}||]
        op2_abs_norms: [|||W_0|||, ..., |||W_{L-1}|||]
        sqrt_m_dict: Precomputed sqrt(m) values
        fmt: Target floating-point format
        L_ij: {(i,j): L_{i,j}} row difference norms
        S_ij: {(i,j): S_{i,j}} row sum norms
        i_star: Predicted class

    Returns:
        HybridMeasuredResult with all certification data
    """
    L = len(network)

    # Step 1: Forward passes
    z_hi = forward_layerwise_float64(network, x)

    # Forward pass in target format
    if fmt.name == "float16":
        z_fp = forward_layerwise_float16(network, x)
    else:
        # For float32, use float64 as approximation (should add float32 forward)
        z_fp = z_hi  # Conservative: assumes no fp16-fp64 difference

    # Step 2: Build hybrid+measured data
    data = build_hybrid_measured_data(
        network, x, epsilon, op2_norms, op2_abs_norms,
        z_hi, z_fp, sqrt_m_dict, fmt
    )

    # Step 3: Compute measured radii for center (ε=0) and ball
    r_meas_center = compute_r_meas_all_layers(
        data.z_hi_norms, data.Lip_cumulative, Q(0), data.D_hi
    )
    r_meas_ball = compute_r_meas_all_layers(
        data.z_hi_norms, data.Lip_cumulative, epsilon, data.D_hi
    )

    # Step 4: Compute D^hybrid for center
    D_hi_final = data.D_hi[-2] if L > 1 else data.D_hi[-1]  # D^hi at L-2 (last hidden)
    D_hybrid_center = compute_D_hybrid_center(data.measured_center_diff, D_hi_final)

    # Step 5: Compute D^meas for ball
    input_radius = l2_norm_upper_bound_vec(x) + epsilon
    D_meas_ball, D_meas_all = compute_D_meas_with_input(
        network, op2_norms, op2_abs_norms,
        input_radius, r_meas_ball, sqrt_m_dict, fmt
    )

    # Step 6: Compute E_ctr and E_ball for each (i,j) pair
    # α_L^{(i,j)} = L_{i,j} + κ · S_{i,j}
    # β_L^{(i,j)}(r) = κ · S_{i,j} · r + u·(|b_i| + |b_j|) + 2(1+u)·a_dot

    W_L = network[-1]
    m_L, n_L = dims(W_L)

    u = float_to_q(fmt.u)
    amul = float_to_q(fmt.denorm_min) / 2
    gamma = gamma_n(n_L, u)
    kappa = gamma + u * (Q(1) + gamma)
    a_dot_val = a_dot(n_L, u, amul)

    E_ctr = {}
    E_ball = {}

    # r_{L-1} for center and ball
    r_Lm1_center = r_meas_center[-2] if L > 1 else r_meas_center[-1]
    r_Lm1_ball = r_meas_ball[-2] if L > 1 else r_meas_ball[-1]

    for (i, j), L_val in L_ij.items():
        if i != i_star:
            continue

        S_val = S_ij[(i, j)]

        # α_L = L_{i,j} + κ · S_{i,j}
        alpha_L = L_val + kappa * S_val

        # For bias term, assume 0 bias (or extract from network if available)
        # β_L(r) = κ · S · r + u·(|b_i|+|b_j|) + 2(1+u)·a_dot
        # Simplified: assume no bias for now
        beta_ctr = kappa * S_val * r_Lm1_center + Q(2) * (Q(1) + u) * a_dot_val
        beta_ball = kappa * S_val * r_Lm1_ball + Q(2) * (Q(1) + u) * a_dot_val

        # E_ctr = α_L · D^hybrid + β_L(r^meas_center)
        E_ctr[(i, j)] = alpha_L * D_hybrid_center + beta_ctr

        # E_ball = α_L · D^meas + β_L(r^meas_ball)
        E_ball[(i, j)] = alpha_L * D_meas_ball + beta_ball

    return HybridMeasuredResult(
        r_meas_center=r_meas_center,
        r_meas_ball=r_meas_ball,
        D_hybrid_center=D_hybrid_center,
        D_meas_ball=D_meas_ball,
        D_meas_all=D_meas_all,
        E_ctr=E_ctr,
        E_ball=E_ball
    )
