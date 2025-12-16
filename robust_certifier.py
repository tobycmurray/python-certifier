from __future__ import annotations
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import sys, os, json, time, math

from parsing import load_network_from_file, ParseError, load_vector_from_npy_file
from arithmetic import Q, qstr, sqrt_upper_bound, round_up, float_to_q
from linear_algebra import Matrix, Vector, l2_norm_upper_bound_vec, dims
from overflow import check_overflow_single_layer
from deviation import compute_layer_deviation_params, compute_deviation_bound
from formats import get_float_format, FloatFormat, gamma_n, a_dot
from norms import compute_norms, load_norms, save_norms, hash_file_contents
from margin_lipschitz import margin_lipschitz_bounds, check_margin_lipschitz_bounds

@dataclass(frozen=True)
class FloatStats:
    mean: Optional[float]
    minimum: Optional[float]
    maximum: Optional[float]

def compute_stats(vals: List[float]) -> FloatStats:
    if not vals:
        return FloatStats(None, None, None)
    fsum = math.fsum(vals)
    return FloatStats(fsum/len(vals), min(vals), max(vals))

def check_rounding_preconditions(network: List[Matrix], fmt: FloatFormat) -> None:
    """
     Ensures nu = n*u < 1 for every layer (matvec length constraint needed for gamma_n).
    Raises ValueError with an explanatory message if violated.
    """
    u = float_to_q(fmt.u)
    for ell, W in enumerate(network):
        m, n = dims(W)
        nu = u * n
        if nu >= Q(1):
            raise ValueError(
                f"[Precondition] Layer {ell}: n*u = {qstr(nu)} ≥ 1 (n={n}, u={qstr(u)}). "
                "The rounding-error model with γ_n is invalid. "
                "Consider a wider-precision format or reducing layer width."
            )

@dataclass(frozen=True)
class ModeBComponents:
    r_prev: List[Q]                                 # [r0, ..., r_{L-1}]
    alphas: List[Q]                                 # hidden layers 0..L-2
    betas: List[Q]                                  # hidden layers 0..L-2
    gammas: List[Q]
    kappas: List[Q]
    DLm1: Q                                         # hidden stack degradation
    right_products: List[Q]     # ∏_{t=ℓ+1}^{L-1} α_t for ℓ=0..L-2
    contribs: List[Q]           # β_ℓ * right_products[ℓ]
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
    r_prev = radii(op2_norms, x, epsilon)

    alphas, betas, gammas, kappas = compute_linear_recursion_hidden(
        op2_norms, op2_abs_norms, r_prev[:-1], network, sqrt_m_ells, fmt
    )

    DLm1, right_products = hidden_stack_degradation_with_products(alphas, betas)

    contribs = []
    for ell in range(len(alphas)):
        contribs.append(betas[ell] * right_products[ell])

    final_pairs = compute_final_pair_params(network[-1], r_prev[-1], fmt, L, S)

    return ModeBComponents(
        r_prev=r_prev, alphas=alphas, betas=betas,
        gammas=gammas, kappas=kappas, DLm1=DLm1, right_products=right_products,
        contribs=contribs, final_pairs=final_pairs
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
    max_lhs: Q

def E_for_pair(components: ModeBComponents, i: int, j: int) -> Q:
    """E^{(i,j)} = α_L^(i,j) * D_{L-1} + β_L^(i,j)."""
    alpha_L_ij, beta_L_ij = components.final_pairs[(i, j)]
    return alpha_L_ij * components.DLm1 + beta_L_ij

def certify_mode_b_theorem4(
    y_f32: List[float],
    epsilon: Q,
    L_real: List[List[Q]],
    comp_ctr: ModeBComponents,
    comp_ball: ModeBComponents,
) -> ModeBReport:
    if not y_f32:
        return ModeBReport(ok=False, ok_real=False, xstar=-1, pairs=[], first_failure=None, max_lhs=Q(0))

    xstar = max(range(len(y_f32)), key=lambda k: y_f32[k])
    yQ = [float_to_q(v) for v in y_f32]

    results: List[ModeBPairResult] = []
    first_fail: Tuple[int, Q, Q] | None = None
    max_lhs = None

    for j in range(len(y_f32)):
        if j == xstar: continue
        lhs = yQ[xstar] - yQ[j]
        max_lhs = lhs if max_lhs is None or lhs > max_lhs else max_lhs

        E_ctr  = E_for_pair(comp_ctr,  xstar, j)
        E_ball = E_for_pair(comp_ball, xstar, j)
        float_cons = round_up(E_ctr + E_ball)

        rhs_real = epsilon * L_real[xstar][j]
        rhs = rhs_real + float_cons
        ok_real = lhs > rhs_real
        ok = lhs > rhs
        results.append(
            ModeBPairResult(j=j, margin_center=lhs, rhs_bound=rhs, rhs_real=rhs_real,
                            float_conservatism=float_cons, ok_real=ok_real, ok=ok)
        )
        if (not ok) and (first_fail is None):
            first_fail = (j, lhs, rhs)

    return ModeBReport(ok=all(r.ok for r in results), ok_real=all(r.ok_real for r in results),
                       xstar=xstar, pairs=results, first_failure=first_fail, max_lhs=max_lhs)


def hidden_stack_degradation_with_products(alphas: List[Q], betas: List[Q]) -> Tuple[Q, List[Q]]:
    Lh = len(alphas)
    if Lh == 0:
        return Q(0), []
    right_products = [Q(1)] * Lh
    for t in range(Lh - 2, -1, -1):
        right_products[t] = right_products[t + 1] * alphas[t + 1]
    DLm1 = sum(betas[s] * right_products[s] for s in range(Lh))
    return DLm1, right_products

def compute_linear_recursion_hidden(
    op2_norms: List[Q],        # [‖W_1‖₂, …, ‖W_{L-1}‖₂]
    op2_abs_norms: List[Q],    # [‖|W_1|‖₂, …, ‖|W_{L-1}|‖₂]
    radii_prev: List[Q],       # [r_0, …, r_{L-2}]   (length L-1; one per hidden layer input)
    network: List[Matrix],     # to read m_ℓ, n_ℓ
    sqrt_m_ells: Dict[int,Q],
    fmt: FloatFormat,
) -> Tuple[List[Q], List[Q], List[Q], List[Q]]:
    """
    Returns (alphas, betas, gammas, kappas) for hidden layers
    """
    L = len(network)
    if L == 0:
        return [], [], [], []
    if not (len(op2_norms) == L and len(op2_abs_norms) == L and len(radii_prev) == L-1):
        raise ValueError("length mismatch between norms/radii/network")

    u = float_to_q(fmt.u)                  # unit roundoff as Q
    amul = float_to_q(fmt.denorm_min) / 2  # amul = denorm_min / 2

    alphas: List[Q] = []
    betas:  List[Q] = []
    kappas: List[Q] = []
    gammas: List[Q] = []

    for ell in range(L-1):
        m_ell, n_ell = dims(network[ell])
        gamma = gamma_n(n_ell, u); gammas.append(gamma)
        kappa = gamma + u * (Q(1) + gamma); kappas.append(kappa)
        alpha = op2_norms[ell] + kappa * op2_abs_norms[ell]

        beta_data  = kappa * op2_abs_norms[ell] * radii_prev[ell]
        beta_noise = (Q(1) + u) * a_dot(n_ell, u, amul) * sqrt_m_ells[m_ell]
        betas.append(beta_data + beta_noise)
        alphas.append(alpha)

    return alphas, betas, gammas, kappas

def compute_L_and_S(W_last: Matrix):
    def row(idx: int) -> List[Q]:
        return W_last[idx]
    m, n = dims(W_last)
    L, S = {}, {}
    for i in range(m):
        Wi = row(i)
        absWi = [abs(x) for x in Wi]
        for j in range(m):
            if i == j: continue
            Wj = row(j)
            diff = [Wj[k] - Wi[k] for k in range(n)]
            L[(i,j)] = l2_norm_upper_bound_vec(diff)
            absWj = [abs(x) for x in Wj]
            sumabs = [absWi[k] + absWj[k] for k in range(n)]
            S[(i,j)] = l2_norm_upper_bound_vec(sumabs)
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
    u = float_to_q(fmt.u)
    amul = float_to_q(fmt.denorm_min) / 2
    kappa = gamma_n(n, u) + u * (Q(1) + gamma_n(n, u))
    ad = a_dot(n, u, amul)

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
        m_ell, n_ell = dims(network[ell])
        if m_ell not in res:
            res[m_ell] = sqrt_upper_bound(Q(m_ell))
    return res

def main():
    if len(sys.argv) != 6:
        print(f"Usage: {sys.argv[0]} format <neural_network_input.txt> <GRAM_ITERATIONS> <cex_file.json> <dafny-ref-json-file>")
        sys.exit(1)

    float_format = sys.argv[1]
    sys.argv = sys.argv[1:]
    network_file = sys.argv[1]
    try:
        gram_iters = int(sys.argv[2])
    except ValueError:
        print("Error: <GRAM_ITERATIONS> must be an integer.")
        sys.exit(1)

    cex_file = sys.argv[3]

    dafny_json_file = sys.argv[4]

    # load network & norms
    try:
        net = load_network_from_file(network_file, validate=True)
    except ParseError as e:
        print(f"Error parsing network file: {e}"); sys.exit(1)
    except FileNotFoundError:
        print(f"Error: file not found: {network_file}"); sys.exit(1)

    print(f"Loaded network with {len(net)} layers; running {gram_iters} Gram iterations per layer...")

    hsh = hash_file_contents(network_file)
    norms_file = hsh+f".{gram_iters}.norms.json"
    try:
        norms = load_norms(hsh, gram_iters, norms_file)
    except Exception as e:
        print(f"Failed to load pre-computed norms. Got error: {e}")
        norms = compute_norms(net, gram_iters)
        save_norms(hsh, gram_iters, norms, norms_file)

    max_row_inf_norms = norms.max_row_inf_norms
    op2_norms = norms.op2_norms
    op2_abs_norms = norms.op2_abs_norms
    max_row_l2_norms = norms.max_row_l2_norms

    # Extract layer widths (input dimensions) for overflow checking with margin terms
    layer_widths = [len(W[0]) for W in net]  # n_l = number of columns in W_l

    print("Computing margin Lipschitz bounds...")
    L_real = margin_lipschitz_bounds(net, op2_norms)
    check_margin_lipschitz_bounds(L_real, gram_iters, dafny_json_file)
    print("Computed margin Lipschitz bounds match Dafny reference numbers exactly.")

    fmt = get_float_format(float_format)
    check_rounding_preconditions(net, fmt)

    # inputs to certify
    to_certify = []
    with open(cex_file, "r") as f:
        cexs = json.load(f)
    for cex in cexs:
        d = os.path.dirname(cex_file)
        x1_file = os.path.join(d, os.path.basename(cex["x1_file"]))
        x = load_vector_from_npy_file(x1_file)
        epsilon = Q(cex["max_eps"])
        y_f32 = cex.get("y1", None)
        if y_f32 is None:
            print("cex file missing \"y1\" field")
            sys.exit(1)
        to_certify.append((x,epsilon,y_f32))

    sqrt_m_ells = compute_sqrt_m_ells(net)
    W_last = net[-1]
    L_pairs,S_pairs = compute_L_and_S(W_last)

    L = len(net)      # total layers
    H = L - 1         # hidden layers (L - 1)
    layer_fanins = [dims(net[ell])[1] for ell in range(H)]
    abs_signed_ratios = [float(op2_abs_norms[ell] / op2_norms[ell]) for ell in range(H)]
    u = fmt.u  # scalar float

    print(f"Got {len(to_certify)} instances to certify...")
    results = []

    times = {"radii":0.0, "overflow_check":0.0, "components":0.0, "certification":0.0}

    # essential aggregators
    r0_all: List[float] = []
    DLm1s: List[float] = []
    float_cons_all: List[float] = []
    real_rhs_all: List[float] = []

    # layerwise means we care about (across examples)
    layer_alpha_sum    = [0.0]*H
    layer_kappa_sum    = [0.0]*H
    layer_r_input_sum  = [0.0]*H
    layer_rightprod_sum= [0.0]*H
    layer_beta_sum     = [0.0]*H
    layer_contrib_sum  = [0.0]*H
    layer_count = 0

    for idx, (x,epsilon,y_f32) in enumerate(to_certify):
        if idx % 100 == 0:
            print(f"Certifying {idx} of {len(to_certify)}")
        first_cols = len(net[0][0])
        if len(x) != first_cols:
            print(f"Error: input length {len(x)} does not match first layer column count {first_cols}.")
            sys.exit(1)

        # ===== Step 1: Compute exact radii (unchanged) =====
        t0 = time.perf_counter()
        rs = radii(op2_norms, x, epsilon)
        t1 = time.perf_counter()
        times["radii"] += (t1 - t0)
        r0_all.append(float(rs[0]))

        # ===== Step 2: Forward induction through layers =====
        # Interleave overflow checking and deviation computation
        t2 = time.perf_counter()

        D_prev = Q(0)  # D_{-1} = 0 for exact input
        layer_params = []  # Track deviation params for each hidden layer
        deviation_bounds = []  # Track [D_0, D_1, ..., D_{L-2}]

        # Debug header for first example
        if idx == 0:
            print(f"\nDEBUG: Deviation computation trace for {float_format}:")
            x_norm = l2_norm_upper_bound_vec(x)
            print(f"  Input: x norm = {float(x_norm):.6e}, epsilon = {float(epsilon):.6e}")
            print(f"  Network: {H} hidden layers + 1 output layer")
            print(f"  Radii: {[f'{float(r):.2e}' for r in rs]}")
            print(f"  ||W||_2 norms: {[f'{float(n):.2e}' for n in op2_norms[:H]]}")
            print(f"  |||W|||_2 norms: {[f'{float(n):.2e}' for n in op2_abs_norms[:H]]}")
            print(f"  Layer widths: {[dims(net[i])[1] for i in range(H)]}")
            print(f"  Unit roundoff u: {float(float_to_q(fmt.u)):.6e}")
            print()

        # Hidden layers (0 to L-2)
        for ell in range(H):  # H = L - 1 hidden layers
            m_ell, n_ell = dims(net[ell])

            # (a) Check overflow at layer ℓ using r_{ℓ-1} + D_{ℓ-1}
            fp_bound = rs[ell] + D_prev
            overflow_stats = check_overflow_single_layer(
                layer_idx=ell,
                max_row_l2=max_row_l2_norms[ell],
                max_abs_entry=max_row_inf_norms[ell],
                fp_activation_bound=fp_bound,
                layer_width=n_ell,
                fmt=fmt,
                bias_norm=Q(0)  # Assuming no biases in hidden layers
            )
            if not overflow_stats.ok:
                print(f"WARNING: Overflow certification failed at layer {ell}:")
                print(f"  S_layer check: slack = {float(overflow_stats.slack_2):.15e}")
                print(f"  M_layer check: slack = {float(overflow_stats.slack_inf):.15e}")
                print(f"  Continuing with robustness certification anyway...")

            # (b) Compute deviation D_ℓ for this layer
            params = compute_layer_deviation_params(
                layer_idx=ell,
                op2_norm=op2_norms[ell],
                op2_abs_norm=op2_abs_norms[ell],
                r_prev=rs[ell],  # r_{ℓ-1} (exact radius)
                layer_width=n_ell,
                output_dim=m_ell,
                sqrt_m=sqrt_m_ells[m_ell],
                fmt=fmt
            )
            D_curr = compute_deviation_bound(D_prev, params)

            # Debug trace for first example
            if idx == 0:
                print(f"  Layer {ell}: α={float(params.alpha):.6e}, β={float(params.beta):.6e}, "
                      f"r_{{{ell}}}={float(rs[ell]):.6e}, D_{{{ell-1}}}={float(D_prev):.6e}, "
                      f"D_{{{ell}}}={float(D_curr):.6e}, D/r={float(D_curr/rs[ell]) if rs[ell] > 0 else 'N/A'}")

            layer_params.append(params)
            deviation_bounds.append(D_curr)
            D_prev = D_curr

        # Final layer overflow check (uses D_{L-2})
        if L > 0:
            m_last, n_last = dims(net[-1])
            fp_bound_final = rs[H] + D_prev  # rs[H] = r_{L-1}

            overflow_stats_final = check_overflow_single_layer(
                layer_idx=H,
                max_row_l2=max_row_l2_norms[H],
                max_abs_entry=max_row_inf_norms[H],
                fp_activation_bound=fp_bound_final,
                layer_width=n_last,
                fmt=fmt,
                bias_norm=Q(0)
            )
            if not overflow_stats_final.ok:
                print(f"\nWARNING: Overflow certification failed at final layer {H}:")
                print(f"WARNING: Continuing with robustness certification anyway...")

        t3 = time.perf_counter()
        times["overflow_check"] += (t3 - t2)

        # ===== Step 3: Build components and certify =====
        t4 = time.perf_counter()

        # DLm1 is now D_{L-2} from forward induction
        DLm1 = D_prev

        comp_ball = build_modeb_components(net, sqrt_m_ells, op2_norms, op2_abs_norms, x, epsilon, fmt, L_pairs, S_pairs)
        comp_ctr  = build_modeb_components(net, sqrt_m_ells, op2_norms, op2_abs_norms, x, Q(0),   fmt, L_pairs, S_pairs)
        t5 = time.perf_counter()
        modeb = certify_mode_b_theorem4(y_f32, epsilon, L_real, comp_ctr, comp_ball)
        t6 = time.perf_counter()

        times["components"] += (t5 - t4)
        times["certification"] += (t6 - t5)
        results.append(modeb)

        # essentials
        DLm1s.append(float(comp_ball.DLm1))

        # layerwise means we care about (only in standard mode)
        for ell in range(H):
            layer_alpha_sum[ell]    += float(comp_ball.alphas[ell])
            layer_kappa_sum[ell]    += float(comp_ball.kappas[ell])
            layer_r_input_sum[ell]  += float(comp_ball.r_prev[ell])
            layer_rightprod_sum[ell]+= float(comp_ball.right_products[ell])
            layer_beta_sum[ell]     += float(comp_ball.betas[ell])
            layer_contrib_sum[ell]  += float(comp_ball.contribs[ell])

        result = results[-1]
        if result.pairs:
            fc = [float(p.float_conservatism) for p in result.pairs]
            rr = [float(p.rhs_real) for p in result.pairs]
            float_cons_all.append(sum(fc)/len(fc))
            real_rhs_all.append(sum(rr)/len(rr))

        layer_count += 1

    results_ok = [r for r in results if r.ok]
    results_fail = [r for r in results if not r.ok]
    results_ok_real = [r for r in results if r.ok_real]

    print("\nCERTIFIER RESULTS")
    print(f"Norms file: {norms_file}")
    print(f"Of {len(results)} instances we attempted to certify:")
    print(f"  Certified {len(results_ok)} instances as robust")
    print(f"  Failed to certify {len(results_fail)} instances as robust")
    print(f"  Real certifier would have certified {len(results_ok_real)} instances as robust")

    # essentials summary
    stats_fc  = compute_stats(float_cons_all)
    stats_rr  = compute_stats(real_rhs_all)
    stats_D   = compute_stats(DLm1s)
    stats_r0  = compute_stats(r0_all)

    pct_fc = (stats_fc.mean / stats_rr.mean) if (stats_fc.mean is not None and stats_rr.mean and stats_rr.mean != 0.0) else None

    print("\nFloat conservatism:")
    print(f"  mean float_conservatism per pair: {stats_fc.mean}")
    print(f"  mean real RHS per pair:           {stats_rr.mean}")
    print(f"  float_conservatism / real_RHS:    {pct_fc}")

    print(f"\nD_(L-1) mean: {stats_D.mean}")
    print(f"r0 (per example) mean: {stats_r0.mean}")

    # layerwise spotlight (first 2 layers) + top-3 contributors
    if layer_count > 0 and H > 0:
        mean_alpha    = [a/layer_count for a in layer_alpha_sum]
        mean_kappa    = [k/layer_count for k in layer_kappa_sum]
        mean_r_in     = [r/layer_count for r in layer_r_input_sum]
        mean_right    = [p/layer_count for p in layer_rightprod_sum]
        mean_beta     = [b/layer_count for b in layer_beta_sum]
        mean_contrib  = [c/layer_count for c in layer_contrib_sum]
        total_contrib = sum(mean_contrib) if sum(mean_contrib) != 0.0 else 1.0
        shares = [c/total_contrib for c in mean_contrib]

        def print_layer(ell: int):
            n_in = layer_fanins[ell]
            nu   = n_in * u
            print(f"  Layer {ell}: n_in={n_in}, nu=n*u={nu:.7e}, kappa={mean_kappa[ell]:.7e}, "
                  f"|W|/|W|={abs_signed_ratios[ell]:.3f}, alpha={mean_alpha[ell]:.5f}, "
                  f"r_in={mean_r_in[ell]:.5f}, right_prod={mean_right[ell]:.5f}, "
                  f"beta={mean_beta[ell]:.6f}, contrib={mean_contrib[ell]:.6f}, share(D)={shares[ell]:.3f}")

        idx_sorted = sorted(range(H), key=lambda e: mean_contrib[e], reverse=True)
        topk = idx_sorted[:min(3, H)]
        print("\nTop contributing hidden layers:")
        for rank, ell in enumerate(topk, 1):
            print(f"  {rank}. ", end="")
            print_layer(ell)

    N = len(to_certify) if to_certify else 1
    print(f"\nAverage Times (ms, over {len(to_certify)} runs):")
    print(f"  Radii computation: {times['radii'] * 1000 / N}")
    print(f"  Overflow check:    {times['overflow_check'] * 1000 / N}")
    print(f"  Components:        {times['components'] * 1000 / N}")
    print(f"  Certification:     {times['certification'] * 1000 / N}")
    print(f"  Overall:           {(times['radii'] + times['overflow_check'] + times['components'] + times['certification']) * 1000 / N}")

if __name__ == "__main__":
    main()
