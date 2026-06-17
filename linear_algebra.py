import time
from typing import List, Tuple, Optional
from arithmetic import Q, sqrt_upper_bound, round_up, round_down, qstr

Matrix = List[List[Q]]
Vector = List[Q]

def dims(M: Matrix) -> Tuple[int, int]:
    return len(M), len(M[0]) if M else (0, 0)

def zeros(m: int, n: int) -> Matrix:
    return [[Q(0) for _ in range(n)] for _ in range(m)]

def is_zero_matrix(M: Matrix) -> bool:
    return all(x == 0 for row in M for x in row)

def transpose(M: Matrix) -> Matrix:
    m, n = dims(M)
    return [[M[i][j] for i in range(m)] for j in range(n)]

def mv_product(M: Matrix, v: Vector) -> Vector:
    m, n = dims(M)
    assert len(v) == n
    out: Vector = []
    for i in range(m):
        s = Q(0)
        row = M[i]
        for j in range(n):
            s += row[j] * v[j]
        out.append(s)
    return out

def mm_product(A: Matrix, B: Matrix) -> Matrix:
    m, k = dims(A)
    k2, n = dims(B)
    assert k == k2
    Bt = transpose(B)
    out = zeros(m, n)
    for i in range(m):
        Ai = A[i]
        for j in range(n):
            s = Q(0)
            Bj = Bt[j]
            for t in range(k):
                s += Ai[t] * Bj[t]
            out[i][j] = s
    return out

# Optional progress tracing for the very expensive norm computation. compute_norms
# turns this on; it is off by default so ordinary (cheap) cert runs stay quiet.
# The CIFAR-10 gram-12 norm computation is ~43h (almost all in layer 0's 3072x3072
# Gram), so without this the run is a silent black box.
PROGRESS = False
_PROGRESS_MIN_N = 200  # only trace mtm for matrices at least this wide (skip tiny layers)


def mtm(M: Matrix) -> Matrix:
    """
    Specialised MTM: returns M^T * M using symmetry, row-wise scanning.
    """
    m, n = dims(M)  # m rows, n cols
    out = zeros(n, n)
    _prog = PROGRESS and n >= _PROGRESS_MIN_N
    if _prog:
        _t0 = time.perf_counter()
        _step = max(1, n // 50)
        print(f"      [mtm] {n}x{n} Gram (inner length {m})...", flush=True)
    # compute upper triangle, reuse symmetry
    for i in range(n):
        for j in range(i, n):
            s = Q(0)
            for r in range(m):
                s += M[r][i] * M[r][j]
            out[i][j] = s
            if j != i:
                out[j][i] = s
        if _prog and (i % _step == 0 or i == n - 1):
            el = time.perf_counter() - _t0
            # upper-triangle work done after row i is proportional to (2i*n - i^2)/n^2
            done = i + 1
            frac = (2 * done * n - done * done) / (n * n)
            eta = (el / frac - el) if frac > 0 else 0.0
            print(f"      [mtm] row {done}/{n}  {frac*100:5.1f}%  "
                  f"elapsed {el/60:6.1f}min  eta {eta/60:6.1f}min", flush=True)
    return out

def matrix_div_scalar(M: Matrix, r: Q) -> Matrix:
    if r == 0:
        raise ZeroDivisionError("division by zero in matrix_div_scalar")
    m, n = dims(M)
    out = zeros(m, n)
    for i in range(m):
        for j in range(n):
            out[i][j] = M[i][j] / r
    return out

def matrix_sub(A: Matrix, B: Matrix) -> Matrix:
    m, n = dims(A)
    assert dims(B) == (m, n)
    out = zeros(m, n)
    for i in range(m):
        for j in range(n):
            out[i][j] = A[i][j] - B[i][j]
    return out

def truncate_with_error(M: Matrix) -> Tuple[Matrix, Q]:
    m, n = dims(M)
    T = zeros(m, n)
    sq_sum = Q(0)
    for i in range(m):
        for j in range(n):
            x = M[i][j]
            if x > 0:
                t = round_up(x)
            elif x < 0:
                t = round_down(x)
            else:
                t = Q(0)
            T[i][j] = t
            e = t - x
            sq_sum += e * e
    e_frob = sqrt_upper_bound(sq_sum)
    return T, e_frob

def frobenius_norm_upper_bound(M: Matrix) -> Q:
    sq_sum = Q(0)
    for row in M:
        for x in row:
            sq_sum += x * x
    return sqrt_upper_bound(sq_sum)

def l2_norm_upper_bound_vec(v: Vector) -> Q:
    # Exact sum of squares, then sqrt_upper_bound
    sq_sum = Q(0)
    for x in v:
        sq_sum += x * x
    return sqrt_upper_bound(sq_sum)

def _gram_unwind(s0: Q, a: List[Tuple[Q, Q]]) -> Q:
    """Backward pass of the Gram iteration: recover the spectral-norm upper
    bound from the final Frobenius norm s0 and the stored (scale, error) pairs."""
    ret = s0
    for (r, e) in a:
        ret = sqrt_upper_bound(r * (ret + e))
    return ret


def gram_iteration(M: Matrix, n: int,
                   trace: Optional[List[Q]] = None,
                   time_trace: Optional[List[float]] = None) -> Q:
    """Spectral-norm upper bound of M via n Gram iterations.

    If `trace` is given (a list), the bound after each iteration k=1..n is
    appended to it. trace[k-1] is exactly what gram_iteration(M, k) returns
    (the iteration is sequential, so a length-n run passes through every
    shorter run's state) -- so one run yields the whole convergence curve.
    The trace uses the same _gram_unwind as the final result, so it is faithful.

    If `time_trace` is given, the *cumulative* wall-clock seconds spent on the
    core iteration work (mtm + normalisation + truncation) up to and including
    iteration k is appended -- recorded BEFORE the optional bound-trace unwind,
    so it reflects a plain gram_iteration(M, k) run (the per-iteration unwind
    overhead, ~0.2% of the matrix product, is excluded). Lets one run also yield
    the per-gram norm-computation cost.
    """
    M_cur = [row[:] for row in M]
    a: List[Tuple[Q, Q]] = []
    t0 = time.perf_counter() if (time_trace is not None or PROGRESS) else 0.0
    t_iter = t0
    i = 0
    while i != n:
        Mp = mtm(M_cur)
        r = Q(1) if is_zero_matrix(Mp) else frobenius_norm_upper_bound(Mp)
        Mn = matrix_div_scalar(Mp, r)
        M_trunc, e = truncate_with_error(Mn)
        a = [(r, e)] + a
        M_cur = M_trunc
        i += 1
        if time_trace is not None:
            time_trace.append(time.perf_counter() - t0)
        if PROGRESS:
            now = time.perf_counter()
            print(f"    [gram] iter {i}/{n} done: {(now-t_iter)/60:.1f}min "
                  f"(cumulative {(now-t0)/60:.1f}min)", flush=True)
            t_iter = now
        if trace is not None:
            trace.append(_gram_unwind(frobenius_norm_upper_bound(M_cur), a))

    return _gram_unwind(frobenius_norm_upper_bound(M_cur), a)

def layer_opnorm_upper_bound(W: Matrix, gram_iters: int) -> Q:
    return gram_iteration(W, gram_iters)

def layer_infinity_norm(W: Matrix) -> Q:
    """Compute max absolute entry = max_r max_k |W[r,k]| (rational exact).

    This is the maximum row infinity norm, i.e., max_r ||W_r||_∞.
    Used for the M_layer overflow check to bound max single product.

    Note: This is NOT the standard matrix infinity norm (which is max row sum).
    """
    max_abs = Q(0)
    for row in W:
        for x in row:
            ax = abs(x)
            if ax > max_abs:
                max_abs = ax
    return max_abs


def max_row_l2_norm(W: Matrix) -> Q:
    """Compute max row L2 norm = max_r ||W_r||_2 (upper bound).

    This is the maximum L2 norm of any row in the matrix.
    Used for the S_layer overflow check to bound max absolute sum of dot product.

    Note: This is tighter than the spectral norm of the absolute matrix.
    """
    max_norm = Q(0)
    for row in W:
        norm = l2_norm_upper_bound_vec(row)
        if norm > max_norm:
            max_norm = norm
    return max_norm

def abs_matrix(W: Matrix) -> Matrix:
    return [[abs(x) for x in row] for row in W]

def vecqstr(V: Vector) -> str:
    r = [qstr(q) for q in V]
    return "[" + ",".join(r) + "]"
