# robust_certifier.py
# Unverified Python implementation of the verified certifier described in:
# "A Formally Verified Robustness Certifier for Neural Networks" (CAV 2025).
# - Follows Gram iteration (Fig. 6) and SqrtUpperBound (Fig. 7).
# - Arithmetic uses exact rationals (fractions.Fraction).
#
# NOTE: This mirrors the algorithmic structure; it is not a formally verified artifact.

from __future__ import annotations
from typing import List, Tuple, Dict

import sys

from parsing import load_network_from_file, ParseError, load_vector_from_file
from arithmetic import Q, qstr, sqrt_upper_bound, round_up
from linear_algebra import Matrix, Vector, mtm, is_zero_matrix, frobenius_norm_upper_bound, matrix_div_scalar, \
    truncate_with_error, l2_norm_upper_bound_vec, layer_opnorm_upper_bound, layer_infinity_norm, abs_matrix, \
    dims, vecqstr
from overflow import certify_no_overflow_normwise, OverflowReport
from formats import get_float_format, FloatFormat
from nn import forward_numpy_float32, forward

# ---- Mode B components builder ---------------------------------------------
from dataclasses import dataclass

@dataclass(frozen=True)
class ModeBComponents:
    # Everything you need to plug into Theorem 4 for a fixed (x, ε)
    r_prev: List[Q]                                 # [r0, ..., r_{L-1}]
    alphas: List[Q]                                 # hidden layers 0..L-2
    betas: List[Q]                                  # hidden layers 0..L-2
    DLm1: Q                                         # hidden stack degradation
    final_pairs: Dict[Tuple[int,int], Tuple[Q, Q]]  # (i,j) -> (alpha_L_ij, beta_L_ij)

def build_modeb_components(
    network: List[Matrix],
    op2_norms: List[Q],         # [‖W_0‖₂, ..., ‖W_{L-1}‖₂]
    op2_abs_norms: List[Q],     # [‖|W_0|‖₂, ..., ‖|W_{L-1}|‖₂]
    x: Vector,
    epsilon: Q,
    fmt: FloatFormat,
) -> ModeBComponents:
    """
    Compute all Mode B ingredients for the given (x, ε):
      - r_prev(x,ε)
      - hidden {alpha_l, beta_l}
      - D_{L-1}(x,ε)
      - final layer pair params {alpha_L^(i,j), beta_L^(i,j)}
    """
    # 1) radii r_{-1..} but we return per-layer inputs [r0..r_{L-1}]
    r_prev = radii(op2_norms, x, epsilon)              # length L

    # 2) hidden recursion (uses r_prev[:-1])
    alphas, betas, _kappas = compute_linear_recursion_hidden(
        op2_norms=op2_norms,
        op2_abs_norms=op2_abs_norms,
        radii_prev=r_prev[:-1],     # r_0..r_{L-2}
        network=network,
        fmt=fmt,
    )

    # 3) D_{L-1}(x,ε)
    DLm1 = hidden_stack_degradation(alphas, betas)

    # 4) final-layer pair params (needs r_{L-1})
    final_pairs = compute_final_pair_params(
        W_last=network[-1],
        r_last_minus1=r_prev[-1],
        fmt=fmt,
    )

    return ModeBComponents(
        r_prev=r_prev,
        alphas=alphas,
        betas=betas,
        DLm1=DLm1,
        final_pairs=final_pairs,
    )

@dataclass(frozen=True)
class ModeBPairResult:
    j: int
    margin_center: Q       # y[i*] - y[j] as Q
    rhs_bound: Q           # ε·L_real[i*,j] + E_ctr(i*,j) + E_ball(i*,j)
    rhs_real: Q            # ε·L_real[i*,j]
    float_conservatism: Q  # E_ctr(i*,j) + E_ball(i*,j)
    ok_real: bool
    ok: bool

@dataclass(frozen=True)
class ModeBReport:
    ok: bool
    xstar: int
    pairs: List[ModeBPairResult]
    first_failure: Tuple[int, Q, Q] | None  # (j, lhs, rhs)

def E_for_pair(components: ModeBComponents, i: int, j: int) -> Q:
    """E^{(i,j)} = α_L^(i,j) * D_{L-1} + β_L^(i,j)."""
    alpha_L_ij, beta_L_ij = components.final_pairs[(i, j)]
    return alpha_L_ij * components.DLm1 + beta_L_ij

def certify_mode_b_theorem4(
    y_f32: List[float],                       # center logits (NumPy float32 forward is fine)
    epsilon: Q,                               
    L_real: List[List[Q]],                    # real-arithmetic margin Lipschitz matrix
    comp_ctr: ModeBComponents,                # built with ε = 0
    comp_ball: ModeBComponents,               # built with ε (target)
) -> ModeBReport:
    """
    Mode B certification per Theorem 4:
      For i* = argmax y, check ∀ j≠i*:
          (y[i*] - y[j])  >  ε·L_real[i*,j] + E_ctr(i*,j) + E_ball(i*,j).
    """
    if not y_f32:
        return ModeBReport(ok=False, xstar=-1, pairs=[], first_failure=None)

    # top class at the center (float32)
    xstar = max(range(len(y_f32)), key=lambda k: y_f32[k])
    # exact Q versions of logits for precise comparisons
    yQ = [Q(str(v)) for v in y_f32]

    results: List[ModeBPairResult] = []
    first_fail: Tuple[int, Q, Q] | None = None
    ncls = len(y_f32)

    for j in range(ncls):
        if j == xstar:
            continue

        # left-hand side: center margin
        lhs = yQ[xstar] - yQ[j]

        # error budgets and Lipschitz term
        E_ctr  = E_for_pair(comp_ctr,  xstar, j)
        E_ball = E_for_pair(comp_ball, xstar, j)

        # round this up so we can print out the results without having to cast to float
        float_conservatism = round_up(E_ctr + E_ball)

        rhs_real = epsilon * L_real[xstar][j]
        rhs = rhs_real + float_conservatism
        ok_real = lhs > rhs_real
        ok = lhs > rhs
        results.append(ModeBPairResult(j=j, margin_center=lhs, rhs_bound=rhs, rhs_real=rhs_real, float_conservatism=float_conservatism, ok_real=ok_real, ok=ok))
        if (not ok) and (first_fail is None):
            first_fail = (j, lhs, rhs)

    return ModeBReport(ok=all(r.ok for r in results), xstar=xstar, pairs=results, first_failure=first_fail)

def _q(v: float) -> Q:
    return Q(str(v))

def _gamma(n: int, u: Q) -> Q:
    # γ_n = (n u) / (1 - n u)
    nu = u * n
    return nu / (Q(1) - nu)

def _adot(n: int, amul: Q, u: Q) -> Q:
    # adot(n) = (1 + γ_{n-1}) * n * amul   (with γ_{n-1} defined for n-1 >= 1; else take γ_0 = 0)
    gamma_nm1 = _gamma(n-1, u) if n > 1 else Q(0)
    return (Q(1) + gamma_nm1) * n * amul



def hidden_stack_degradation(alphas: List[Q], betas: List[Q]) -> Q:
    """
    D_{L-1}(x, ε) = sum_{s=1}^{L-1} beta_s * (alpha_{s+1} ... alpha_{L-1})
    Here 'alphas' and 'betas' are indexed 0..L-2 (hidden layers).
    """
    Lh = len(alphas)  # number of hidden layers
    if Lh == 0:
        return Q(0)
    # right_products[t] = prod_{u=t+1}^{L-1} alpha_u
    right_products = [Q(1)] * Lh
    right_products[-1] = Q(1)
    for t in range(len(alphas) - 2, -1, -1):
        right_products[t] = right_products[t + 1] * alphas[t + 1]    
    return sum(betas[s] * right_products[s] for s in range(Lh))

def compute_linear_recursion_hidden(
    op2_norms: List[Q],        # [‖W_1‖₂, …, ‖W_{L-1}‖₂]
    op2_abs_norms: List[Q],    # [‖|W_1|‖₂, …, ‖|W_{L-1}|‖₂]
    radii_prev: List[Q],       # [r_0, …, r_{L-2}]   (length L-1; one per hidden layer input)
    network: List[Matrix],     # to read m_ℓ, n_ℓ
    fmt: FloatFormat,          # to get u, denorm_min
) -> Tuple[List[Q], List[Q], List[Q]]:
    """
    Returns (alphas, betas, kappas) for layers ℓ = 1..L-1 (hidden stack).
    All values are Q and follow the Mode B linear recursion definitions.
    """
    L = len(network)
    if L == 0:
        return [], [], []
    if not (len(op2_norms) == L and len(op2_abs_norms) == L and len(radii_prev) == L-1):
        raise ValueError("length mismatch between norms/radii/network")

    u = _q(fmt.u)                  # unit roundoff as Q
    amul = _q(fmt.denorm_min) / 2  # amul = denorm_min / 2

    alphas: List[Q] = []
    betas:  List[Q] = []
    kappas: List[Q] = []

    # hidden layers are indices 0..L-2
    for ell in range(L-1):
        W = network[ell]
        m_ell, n_ell = dims(W)     # m rows, n cols
        kappa = _gamma(n_ell, u) + u * (Q(1) + _gamma(n_ell, u))
        kappas.append(kappa)

        alpha = op2_norms[ell] + kappa * op2_abs_norms[ell]
        # βℓ = κ_nℓ |||Wℓ|||_2 r_{ℓ-1} + (1+u) adot(nℓ) sqrt(mℓ)
        beta_noise = (Q(1) + u) * _adot(n_ell, amul, u) * sqrt_upper_bound(Q(m_ell))
        beta = kappa * op2_abs_norms[ell] * radii_prev[ell] + beta_noise

        alphas.append(alpha)
        betas.append(beta)

    return alphas, betas, kappas


def compute_final_pair_params(
    W_last: Matrix,
    r_last_minus1: Q,
    fmt: FloatFormat,
) -> Dict[Tuple[int,int], Tuple[Q, Q]]:
    """
    For identity final activation (logits) with no bias:
      α_L^(i,j) = L_{i,j} + κ_{n_L} S_{i,j}
      β_L^(i,j) = κ_{n_L} S_{i,j} r_{L-1} + 2(1+u) adot(n_L)

    Returns a dict mapping (i,j) with i!=j to (alpha_L_ij, beta_L_ij).
    """
    m, n = dims(W_last)
    u = _q(fmt.u)
    amul = _q(fmt.denorm_min) / 2
    kappa = _gamma(n, u) + u * (Q(1) + _gamma(n, u))
    ad = _adot(n, amul, u)

    # helper to get rows
    def row(idx: int) -> List[Q]:
        return W_last[idx]

    out: Dict[Tuple[int,int], Tuple[Q,Q]] = {}
    for i in range(m):
        Wi = row(i)
        absWi = [abs(x) for x in Wi]
        for j in range(m):
            if i == j:
                continue
            Wj = row(j)
            # L_{i,j} = || Wi - Wj ||_2
            diff = [Wj[k] - Wi[k] for k in range(n)]
            Lij = l2_norm_upper_bound_vec(diff)
            # S_{i,j} = || |Wi| + |Wj| ||_2
            absWj = [abs(x) for x in Wj]
            sumabs = [absWi[k] + absWj[k] for k in range(n)]
            Sij = l2_norm_upper_bound_vec(sumabs)

            # be conservative and do not take advantage of the "Variant" remark atm
            alpha_ij = Lij + (1 + kappa) * Sij
            beta_ij  = kappa * Sij * r_last_minus1 + Q(2) * (Q(1) + u) * ad
            out[(i, j)] = (alpha_ij, beta_ij)

    return out

def radii(op2_norms: List[Q], x: Vector, epsilon: Q) -> List[Q]:
    L = len(op2_norms)
    if L == 0:
        return []

    r0 = l2_norm_upper_bound_vec(x) + epsilon
    rs = [r0]

    r = r0
    # propagate only until r_{L-1}
    for ell in range(1, L):
        # r_l = L_phi(||W_l|| r_{l-1} + bias_l)
        # no bias term, plus Lipschitz constant of ReLU is 1
        r = op2_norms[ell - 1] * r
        rs.append(r)
        
    assert len(rs) == L
    return rs

def margin_lipschitz_bounds(network: List[Matrix], op2_norms: List[Q]) -> List[List[Q]]:
    """
    Compute L[i][j] margin bounds per the paper:
    - product of operator-norm upper bounds for layers 1..n-1
    - times l2 norm of (last_layer[j] - last_layer[i])
    """
    assert len(network) >= 1
    *hidden, last = network
    *hidden_norms, _ = op2_norms
    # product of op-norm bounds for hidden layers
    prod = Q(1)
    for i, norm in enumerate(hidden_norms):
        prod *= norm
    # compute per-pair margins using last layer's row differences
    rows = last  # last is matrix, rows = classes
    num_classes = len(rows)
    L = [[Q(0) for _ in range(num_classes)] for _ in range(num_classes)]
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                continue
            diff = [rows[j][k] - rows[i][k] for k in range(len(rows[0]))]
            r = l2_norm_upper_bound_vec(diff)
            L[i][j] = prod * r
    return L

def certify(v_prime: Vector, epsilon: Q, L: List[List[Q]]) -> bool:
    """
    Implements the certification rule (Fig. 4):
    - x = argmax v'
    - for all i != x: require v'[x] - v'[i] > epsilon * L[i][x]
    """
    # argmax index
    x = max(range(len(v_prime)), key=lambda k: v_prime[k])
    vx = v_prime[x]
    for i in range(len(v_prime)):
        if i == x:
            continue
        if epsilon * L[i][x] >= vx - v_prime[i]:
            return False
    return True

def main():
    if len(sys.argv) != 5:
        print("Usage: main <neural_network_input.txt> <GRAM_ITERATIONS> <input_x.txt> <epsilon>")
        sys.exit(1)

    network_file = sys.argv[1]
    input_file = sys.argv[3]
    try:
        gram_iters = int(sys.argv[2])
    except ValueError:
        print("Error: <GRAM_ITERATIONS> must be an integer.")
        sys.exit(1)

    try:
        epsilon = Q(sys.argv[4])
    except ValueError:
        print("Error: <epsilon> must be a float.")
        sys.exit(1)
        
    try:
        net = load_network_from_file(network_file, validate=True)
    except ParseError as e:
        print(f"Error parsing network file: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: file not found: {network_file}")
        sys.exit(1)

    print(f"Loaded network with {len(net)} layers; running {gram_iters} Gram iterations per layer...")
    for i, W in enumerate(net):
        print(f"  Layer {i} has dims: {dims(W)}")

    try:
        x = load_vector_from_file(input_file)
    except ParseError as e:
        print(f"Error parsing input file: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: file not found: {input_file}")
        sys.exit(1)

    # check input compatibility with the neural network
    first_cols = len(net[0][0])
    if len(x) != first_cols:
        print(f"Error: input length {len(x)} does not match first layer column count {first_cols}.")
        sys.exit(1)        
    print(f"Loaded input with length {len(x)}")
    
    inf_norms = []
    op2_norms = []
    op2_abs_norms = []

    for i, W in enumerate(net):
        print(f"Computing norms for layer {i}, ...")
        print(f"  infinity norm...")
        inf_norm = layer_infinity_norm(W)
        print(f"  operator norm...") 
        op2_norm = layer_opnorm_upper_bound(W, gram_iters)
        print(f"  operator norm of abs...")
        op2_abs_norm = layer_opnorm_upper_bound(abs_matrix(W), gram_iters)

        inf_norms.append(inf_norm)
        op2_norms.append(op2_norm)
        op2_abs_norms.append(op2_abs_norm)

    print("Computing radii...")
    rs = radii(op2_norms, x, epsilon)
    print("\nRadii:")
    for i, r in enumerate(rs):
        print(f"{i}: {qstr(r)}")

    fmt32 = get_float_format("float32")

    print("Certifying absence of overflow...")
    report = certify_no_overflow_normwise(op2_abs_norms, inf_norms, rs, fmt32)
    if not report.ok:
        print("Overflow certification failed: ")
        print(report)
        sys.exit(1)

    print("Simulating neural network forward pass...")
    y_f32 = forward_numpy_float32(net, x)

    print("Simulating real-valued neural network forward pass...")
    y_real = forward(net, x)

    print("f32  logits: ", y_f32)
    print("real logits: ", vecqstr(y_real))

    print("Computing Mode B ball components...")
    comp_ball = build_modeb_components(net, op2_norms, op2_abs_norms, x, epsilon, fmt32)
    print("Computing Mode B centre components...")
    comp_ctr  = build_modeb_components(net, op2_norms, op2_abs_norms, x, Q(0),   fmt32)

    print("Computing margin Lipschitz bounds...")
    L = margin_lipschitz_bounds(net, op2_norms)
    
    print("\nRunning Mode B (Theorem 4) certification...")
    modeb = certify_mode_b_theorem4(
        y_f32=y_f32,
        epsilon=epsilon,
        L_real=L,
        comp_ctr=comp_ctr,
        comp_ball=comp_ball,
    )
    print(f"Mode B certification check done, got {len(modeb.pairs)} pairs")
    for r in modeb.pairs:
        print(f"  j={r.j}: margin={qstr(r.margin_center)}  "
              f"bound={qstr(r.rhs_bound)}  ok_real={bool(r.ok_real)}   -> {'PASS' if r.ok else 'FAIL'}")

    if modeb.ok:
        print("Mode B: PASS")
    else:
        j, lhs, rhs = modeb.first_failure or (-1, Q(0), Q(0))
        print(f"Mode B: FAIL at j={j}: margin={qstr(lhs)} ≤ bound={qstr(rhs)}")

if __name__ == "__main__":
    main()

