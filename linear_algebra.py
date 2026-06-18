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

def mtm(M: Matrix) -> Matrix:
    """
    Specialised MTM: returns M^T * M using symmetry, row-wise scanning.
    """
    m, n = dims(M)  # m rows, n cols
    out = zeros(n, n)
    # compute upper triangle, reuse symmetry
    for i in range(n):
        for j in range(i, n):
            s = Q(0)
            for r in range(m):
                s += M[r][i] * M[r][j]
            out[i][j] = s
            if j != i:
                out[j][i] = s
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


def gram_iteration(M: Matrix, n: int) -> Q:
    """Spectral-norm upper bound of M via n Gram iterations, exact-rational mtm.

    The original exact-arithmetic Gram iteration. The default certifier norm path
    uses the faster gram_iteration_fp64; this is retained for method="exact".
    """
    M_cur = [row[:] for row in M]
    a: List[Tuple[Q, Q]] = []
    i = 0
    while i != n:
        Mp = mtm(M_cur)
        r = Q(1) if is_zero_matrix(Mp) else frobenius_norm_upper_bound(Mp)
        Mn = matrix_div_scalar(Mp, r)
        M_trunc, e = truncate_with_error(Mn)
        a = [(r, e)] + a
        M_cur = M_trunc
        i += 1
    return _gram_unwind(frobenius_norm_upper_bound(M_cur), a)

# ============================================================================
# Floating-point-sound Gram iteration (writeup.tex sec:fp-gram) -- the DEFAULT
# norm path (layer_opnorm_upper_bound method="fp64").
#
# Runs each Gram product A^T A in binary64 instead of exact rational arithmetic,
# bounding the rounding error via the paper's Higham dot-product model and folding
# it into the unwind as delta_k = t_k + eps_k / r_k. The dominant O(d^3) work
# becomes a binary64 matmul; all error tracking stays exact-rational but O(d^2) on
# format-bounded-bit-length numbers.
#
# Soundness assumes binary64 round-to-nearest dot products with gradual underflow
# (the same model used for the network's matvecs), valid for any conventional
# O(d^3) summation order incl. FMA/blocked gemm, but NOT fast-matrix-multiply
# (Strassen). The default uses numpy's own fp64 sum-of-products (no BLAS trust);
# blas=True opts into the faster BLAS gemm.
# ============================================================================

def truncate_to_fp64_with_error(M: Matrix) -> Tuple[Matrix, Q]:
    """Round each rational entry of M to the nearest binary64 value.

    Returns (T, t) where T is the binary64 matrix (entries read back as exact
    rationals; Q(float(x)) is lossless since every binary64 is a rational) and
    t >= ||M - T||_F is a sound rational Frobenius bound on the rounding error.

    This is the binary64 analogue of truncate_with_error (which rounds to a
    16-dp rational). Soundness needs only that each T[i][j] is some binary64 and
    that t bounds ||M - T||_F; round-to-nearest just makes t as small as possible.
    """
    m, n = dims(M)
    T = zeros(m, n)
    sq_sum = Q(0)
    for i in range(m):
        for j in range(n):
            x = M[i][j]
            t = Q(float(x))          # round-to-nearest binary64, lossless read-back
            T[i][j] = t
            e = t - x
            sq_sum += e * e
    return T, sqrt_upper_bound(sq_sum)


def mtm_fp64_with_error(A: Matrix, fmt=None, blas: bool = False) -> Tuple[Matrix, Q]:
    """Binary64 Gram product A^T A with a sound Frobenius error bound (MTMFP64).

    Returns (Ntil, eps) where:
      - Ntil = fl(A^T A) computed in binary64, read back as an exact rational
        matrix (lossless),
      - eps >= ||Ntil - A^T A||_F, via the closed form
            ||E||_F <= gamma_m * ||A||_F^2 + a_dot_fwd(m) * n,
        where m = rows(A) is the contraction length and n = cols(A) the Gram
        dimension. (Aggregates the entrywise Higham bound
        E_ij = gamma_m (|A|^T|A|)_ij + a_dot_fwd(m); see writeup Lemma 1.)

    The Higham gamma_m bound holds for ANY order of binary64 round-to-nearest
    add/mul. By default (blas=False) the product is computed with numpy's own
    sum-of-products (np.einsum(optimize=False) -- NOT a BLAS call), so the bound
    follows from numpy's documented binary64 summation with no dependence on any
    BLAS implementation. blas=True uses the (faster) BLAS gemm A.T@A instead --
    use at your own risk: it asks the reader to trust the BLAS is a conventional
    non-Strassen/full-precision O(d^3) IEEE-754 fp64 gemm. The matmul is only ~1%
    of the runtime (the exact-rational normalise/truncate dominates), so blas=False
    costs only ~30% more -- worth it to drop the BLAS trust, hence the default.

    Precondition: A's entries are binary64 values (held as exact rationals), so
    the conversion to np.float64 is lossless.
    """
    import numpy as np
    from formats import get_float_format, gamma_n, a_dot_fwd
    if fmt is None:
        fmt = get_float_format("float64")
    m, n = dims(A)                      # m = contraction length, n = Gram dim
    Af = np.array([[float(x) for x in row] for row in A], dtype=np.float64)
    assert Af.dtype == np.float64       # guarantee binary64 (no silent downcast)
    if blas:
        Ntil_f = Af.T @ Af                                   # BLAS gemm
    else:
        Ntil_f = np.einsum('ri,rj->ij', Af, Af, optimize=False)  # numpy's own fp64 sum-of-products
    assert Ntil_f.dtype == np.float64
    if not np.all(np.isfinite(Ntil_f)):
        # overflow/non-finite: refuse (writeup Remark on overflow)
        raise OverflowError("non-finite value in fp64 Gram product; refusing to certify")
    Ntil = [[Q(float(Ntil_f[i, j])) for j in range(n)] for i in range(n)]
    u = Q(fmt.u)
    amul = Q(fmt.denorm_min) / 2
    gamma_m = gamma_n(m, u)
    frob_A = frobenius_norm_upper_bound(A)        # exact, >= ||A||_F
    eps = gamma_m * frob_A * frob_A + a_dot_fwd(m, u, amul) * Q(n)
    return Ntil, eps


def gram_iteration_fp64(M: Matrix, n: int, fmt=None,
                        trace: Optional[List[Q]] = None, blas: bool = False) -> Q:
    """FP-sound spectral-norm upper bound via Gram iteration with binary64 mtm.

    Mirrors gram_iteration() exactly, except each Gram product is computed in
    binary64 (mtm_fp64_with_error) and the per-iteration error fed to the unwind
    is delta_k = t_k + eps_k / r_k (Truncate error + fp Gram error / rescale).
    Returns s >= ||M||_2 (writeup Theorem). `trace` (if given) collects the bound
    after each iteration, as in gram_iteration. blas=False (default) uses numpy's
    own fp64 summation (no BLAS trust); blas=True uses the faster BLAS gemm.
    """
    M_cur = [row[:] for row in M]       # iterate kept binary64 (weights are binary32 subset)
    a: List[Tuple[Q, Q]] = []
    i = 0
    while i != n:
        Ntil, eps = mtm_fp64_with_error(M_cur, fmt, blas=blas)
        r = Q(1) if is_zero_matrix(Ntil) else frobenius_norm_upper_bound(Ntil)
        Mn = matrix_div_scalar(Ntil, r)
        M_trunc, t = truncate_to_fp64_with_error(Mn)
        delta = t + eps / r
        a = [(r, delta)] + a
        M_cur = M_trunc
        i += 1
        if trace is not None:
            trace.append(_gram_unwind(frobenius_norm_upper_bound(M_cur), a))
    return _gram_unwind(frobenius_norm_upper_bound(M_cur), a)


def layer_opnorm_upper_bound(W: Matrix, gram_iters: int, method: str = "fp64") -> Q:
    """Spectral-norm upper bound on W via Gram iteration.

    method="fp64" (default): binary64-mtm Gram iteration using numpy's own fp64
      sum-of-products (no BLAS) -- the scalable path; the O(d^3) Gram product runs
      in binary64 with its rounding error soundly tracked (gamma_m bound provable
      from numpy's documented summation, no BLAS trust). Bounds are >= the
      exact-rational ones (a part in ~1e12 above), so still sound w.r.t. the
      verified certifier.
    method="fp64_blas": same, but the Gram product uses the faster BLAS gemm
      (~30% faster overall; use at your own risk -- relies on the BLAS being a
      conventional non-Strassen full-precision fp64 gemm).
    method="exact": exact-rational mtm (gram_iteration); reproduces the Dafny
      reference exactly but is O(d^3) in bignum arithmetic (infeasible for CIFAR).
    """
    if method == "fp64":
        return gram_iteration_fp64(W, gram_iters, blas=False)
    elif method == "fp64_blas":
        return gram_iteration_fp64(W, gram_iters, blas=True)
    elif method == "exact":
        return gram_iteration(W, gram_iters)
    else:
        raise ValueError(f"unknown opnorm method {method!r} (expected 'fp64', 'fp64_blas', or 'exact')")

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
