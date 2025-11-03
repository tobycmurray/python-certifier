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
import os
import json
import time

from parsing import load_network_from_file, ParseError, load_vector_from_file, load_vector_from_npy_file
from arithmetic import Q, qstr, sqrt_upper_bound, round_up
from linear_algebra import Matrix, Vector, mtm, is_zero_matrix, frobenius_norm_upper_bound, matrix_div_scalar, \
    truncate_with_error, l2_norm_upper_bound_vec, layer_opnorm_upper_bound, layer_infinity_norm, abs_matrix, \
    dims, vecqstr
from overflow import certify_no_overflow_normwise, OverflowReport
from formats import get_float_format, FloatFormat
from nn import forward_numpy_float32, forward
from norms import compute_norms, load_norms, save_norms, hash_file_contents, QEncoder
from margin_lipschitz import margin_lipschitz_bounds, check_margin_lipschitz_bounds

def _q(v: float) -> Q:
    return Q(format(v, '.150f'))

def check_rounding_preconditions(network: List[Matrix], fmt: FloatFormat) -> None:
    """
    Ensures nu = n*u < 1 for every layer (matvec length constraint needed for gamma_n).
    Raises ValueError with an explanatory message if violated.
    """
    u = _q(fmt.u)
    for ell, W in enumerate(network):
        m, n = dims(W)
        nu = u * n
        if nu >= Q(1):
            raise ValueError(
                f"[Precondition] Layer {ell}: n*u = {qstr(nu)} ≥ 1 (n={n}, u={qstr(u)}). "
                "The rounding-error model with γ_n is invalid. "
                "Consider a wider-precision format or reducing layer width."
            )

# ---- Mode B components builder ---------------------------------------------
from dataclasses import dataclass

@dataclass(frozen=True)
class ModeBComponents:
    r_prev: List[Q]                                 # [r0, ..., r_{L-1}]
    alphas: List[Q]                                 # hidden layers 0..L-2
    betas: List[Q]                                  # hidden layers 0..L-2
    DLm1: Q                                         # hidden stack degradation
    final_pairs: Dict[Tuple[int,int], Tuple[Q, Q]]  # (i,j) -> (alpha_L_ij, beta_L_ij)

def build_modeb_components(
    network: List[Matrix],
    sqrt_m_ells: Dict[int,Q],
    op2_norms: List[Q],         # [‖W_0‖₂, ..., ‖W_{L-1}‖₂]
    op2_abs_norms: List[Q],     # [‖|W_0|‖₂, ..., ‖|W_{L-1}|‖₂]
    x: Vector,
    epsilon: Q,
    fmt: FloatFormat,
    L: Dict[Tuple[int,int],Q],
    S: Dict[Tuple[int,int],Q]
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
        sqrt_m_ells=sqrt_m_ells,
        fmt=fmt,
    )

    # 3) D_{L-1}(x,ε)
    DLm1 = hidden_stack_degradation(alphas, betas)

    # 4) final-layer pair params (needs r_{L-1})
    final_pairs = compute_final_pair_params(
        W_last=network[-1],
        r_last_minus1=r_prev[-1],
        fmt=fmt,
        L=L,
        S=S
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
    ok_real: bool
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
    yQ = [_q(v) for v in y_f32]

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

    return ModeBReport(ok=all(r.ok for r in results), ok_real=all(r.ok_real for r in results), xstar=xstar, pairs=results, first_failure=first_fail)

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
    sqrt_m_ells: Dict[int,Q],
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
        beta_noise = (Q(1) + u) * _adot(n_ell, amul, u) * sqrt_m_ells[m_ell]
        beta = kappa * op2_abs_norms[ell] * radii_prev[ell] + beta_noise

        alphas.append(alpha)
        betas.append(beta)

    return alphas, betas, kappas


def compute_L_and_S(W_last: Matrix):
    # helper to get rows
    def row(idx: int) -> List[Q]:
        return W_last[idx]

    m, n = dims(W_last)
    L = {}
    S = {}
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
            L[(i,j)] = Lij
            # S_{i,j} = || |Wi| + |Wj| ||_2
            absWj = [abs(x) for x in Wj]
            sumabs = [absWi[k] + absWj[k] for k in range(n)]
            Sij = l2_norm_upper_bound_vec(sumabs)
            S[(i,j)] = Sij
    return (L,S)

def compute_final_pair_params(
    W_last: Matrix,
    r_last_minus1: Q,
    fmt: FloatFormat,
    L: Dict[Tuple[int,int],Q],
    S: Dict[Tuple[int,int],Q],
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

    out: Dict[Tuple[int,int], Tuple[Q,Q]] = {}
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            Lij = L[(i,j)]
            Sij = S[(i,j)]

            # using (1 + kappa) instead of just kappa here makes a significant difference
            # on the MNIST (CAV 2025) model robustness goes from 93.25% (with 1 + kappa)
            # to 95.50% (with kappa), with a ceiling of 95.74% (the CAV 2025 certifier)
            #
            # for Fashion MNIST (CAV 2025), we certify 82.16% robust (with kappa), with
            # a ceiling of 83.65% (the CAV 2025 certifier)
            alpha_ij = Lij + kappa * Sij
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

def compute_sqrt_m_ells(network: List[Matrix]):
    res = {}
    L = len(network)
    for ell in range(L-1):
        W = network[ell]
        m_ell, n_ell = dims(W)     # m rows, n cols
        if m_ell not in res:
            res[m_ell] = sqrt_upper_bound(Q(m_ell))
    return res

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
    if len(sys.argv) != 7:
        print(f"Usage: {sys.argv[0]} format <neural_network_input.txt> <GRAM_ITERATIONS> --cex <cex_file.json> <dafny-ref-json-file>")
        print(f"Usage: {sys.argv[0]} format <neural_network_input.txt> <GRAM_ITERATIONS> <input_x_file> <epsilon> <dafny-ref-json-file>")
        sys.exit(1)

    float_format = sys.argv[1]
    sys.argv = sys.argv[1:]
    network_file = sys.argv[1]
    try:
        gram_iters = int(sys.argv[2])
    except ValueError:
        print("Error: <GRAM_ITERATIONS> must be an integer.")
        sys.exit(1)

    cex_file, input_file = None, None
    if sys.argv[3] == "--cex":
        cex_file = sys.argv[4]
    else:
        input_file = sys.argv[3]
        try:
            epsilon = Q(sys.argv[4])
        except ValueError:
            print("Error: <epsilon> must be a float.")
            sys.exit(1)

    dafny_json_file = sys.argv[5]

    # load network, get norms
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

    hsh = hash_file_contents(network_file)
    norms_file = hsh+f".{gram_iters}.norms.json" # self-authenticating file name
    try:
        norms = load_norms(hsh, gram_iters, norms_file)
    except Exception as e:
        print(f"Failed to load pre-computed norms. Got error: {e}")
        norms = compute_norms(net, gram_iters)
        save_norms(hsh, gram_iters, norms, norms_file)

    inf_norms, op2_norms, op2_abs_norms = norms.inf_norms, norms.op2_norms, norms.op2_abs_norms

    print("Computing margin Lipschitz bounds...")
    L_real = margin_lipschitz_bounds(net, op2_norms)

    check_margin_lipschitz_bounds(L_real, gram_iters, dafny_json_file)
    print("Computed margin Lipschitz bounds match Dafny reference numbers exactly.")

    # floating-point format: for now, float32 only
    fmt = get_float_format(float_format)

    check_rounding_preconditions(net, fmt)

    # build a list of (x,epsilon,y_f32) triples to certify
    to_certify = []
    if cex_file is not None:
        with open(cex_file, "r") as f:
            cexs = json.load(f)
        for cex in cexs:
            # ignore the absolute paths in the JSON file and assume inputs are in the same directory as the cex file
            d = os.path.dirname(cex_file)
            x1_file = cex["x1_file"]
            x1_file = os.path.basename(x1_file)
            x1_file = os.path.join(d,x1_file)
            x = load_vector_from_npy_file(x1_file)
            epsilon = Q(cex["max_eps"])
            # whether we simulate it or load the answer produced by the original model
            # seems to produce identical robustness numbers for the MNIST CAV 2025 model
            # and for confirming the non-robustness of the MNIST DeepFlool float32 counter-examples.
            if "y1" not in cex:
                print("Simulating neural network forward pass...")
                y_f32 = forward_numpy_float32(net, x)
            else:
                y_f32 = cex["y1"]
            to_certify.append((x,epsilon,y_f32))

    if input_file is not None:
        if input_file.endswith(".npy"):
            x = load_vector_from_npy_file(input_file)
        else:
            x = load_vector_from_file(input_file)
        print("Simulating neural network forward pass...")
        y_f32 = forward_numpy_float32(net, x)
        to_certify.append((x,epsilon,y_f32))


    sqrt_m_ells = compute_sqrt_m_ells(net)
    W_last = net[-1]
    L,S = compute_L_and_S(W_last)

    print(f"Got {len(to_certify)} instances to certify...")
    results = []

    times={}
    times["radii"] = 0
    times["overflow_check"] = 0
    times["components"] = 0
    times["certification"] = 0

    for i, (x,epsilon,y_f32) in enumerate(to_certify):
        if i % 100 == 0:
            print(f"Certifying {i} of {len(to_certify)}")
        # check input compatibility with the neural network
        first_cols = len(net[0][0])
        if len(x) != first_cols:
            print(f"Error: input length {len(x)} does not match first layer column count {first_cols}.")
            sys.exit(1)

        time_start = time.perf_counter()
        rs = radii(op2_norms, x, epsilon)
        time_radius = time.perf_counter()
        times["radii"] += time_radius - time_start

        time_overflow_check_start = time.perf_counter()
        report = certify_no_overflow_normwise(op2_abs_norms, inf_norms, rs, fmt)
        time_overflow_check_end = time.perf_counter()

        if not report.ok:
            print("Overflow certification failed: ")
            print(report)
            sys.exit(1)

        times["overflow_check"] += (time_overflow_check_end - time_overflow_check_start)

        time_certification_start = time.perf_counter()
        comp_ball = build_modeb_components(net, sqrt_m_ells, op2_norms, op2_abs_norms, x, epsilon, fmt, L, S)
        comp_ctr  = build_modeb_components(net, sqrt_m_ells, op2_norms, op2_abs_norms, x, Q(0),   fmt,  L, S)
        time_components_end = time.perf_counter()
        modeb = certify_mode_b_theorem4(
            y_f32=y_f32,
            epsilon=epsilon,
            L_real=L_real,
            comp_ctr=comp_ctr,
            comp_ball=comp_ball,
        )
        time_certification_end = time.perf_counter()

        times["components"] += (time_components_end - time_certification_start)
        times["certification"] += (time_certification_end - time_components_end)

        results.append(modeb)

    results_ok = [r for r in results if r.ok]
    results_fail = [r for r in results if not r.ok]
    results_ok_real = [r for r in results if r.ok_real]

    print(f"Of {len(results)} instances we attempted to certify: ")
    print(f"Certified {len(results_ok)} instances as robust")
    print(f"Failed to certify {len(results_fail)} instances as robust")
    print(f"Real certifier would have certified {len(results_ok_real)} instances as robust")

    N = len(to_certify)
    print(f"\nAverage Times (miliseconds, over {N} runs): ")
    print(f"  Radii computation: {times['radii'] * 1000 / N}")
    print(f"  Overflow check:    {times['overflow_check'] * 1000 / N}")
    print(f"  Components   :     {times['components'] * 1000 / N}")
    print(f"  Certification:     {times['certification'] * 1000 / N}")
    print(f"  Overall:           {(times['radii'] + times['overflow_check'] + times['components'] + times['certification']) * 1000 / N}")

if __name__ == "__main__":
    main()
