"""Empirical check that numpy/OpenBLAS fp64 A^T A obeys the Higham entrywise bound
   |fl(A^T A)_ij - (A^T A)_ij| <= gamma_m (|A|^T|A|)_ij + a_dot_fwd(m),  m = rows(A).
Compares the actual BLAS error (vs exact rational A^T A) to our bound; the bound must
never be exceeded. Reports the worst-case ratio actual_error / bound across trials.

NOTE: this validates the OPT-IN BLAS gemm path (method="fp64_blas"). The DEFAULT
norm method ("fp64") computes the Gram product with numpy's own fp64 sum-of-products
(np.einsum(optimize=False), not BLAS), for which the gamma_m bound follows from
numpy's documented binary64 summation with no dependence on any BLAS internals --
so the default needs no empirical BLAS check. This script remains useful to justify
the faster fp64_blas mode and as a cross-implementation sanity check.
"""
import numpy as np
from arithmetic import Q
from formats import get_float_format, gamma_n, a_dot_fwd

fmt = get_float_format("float64")
u = Q(fmt.u); amul = Q(fmt.denorm_min) / 2

def exact_mtm(Aq, m, n):
    # exact A^T A and |A|^T|A| (rational), columns i,j dot over rows r
    cols = [[Aq[r][i] for r in range(m)] for i in range(n)]
    acols = [[abs(Aq[r][i]) for r in range(m)] for i in range(n)]
    G = [[None]*n for _ in range(n)]
    AbsG = [[None]*n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s = Q(0); sa = Q(0)
            ci, cj, ai, aj = cols[i], cols[j], acols[i], acols[j]
            for r in range(m):
                s += ci[r]*cj[r]; sa += ai[r]*aj[r]
            G[i][j] = G[j][i] = s
            AbsG[i][j] = AbsG[j][i] = sa
    return G, AbsG

def check(A, label):
    m, n = A.shape
    Aq = [[Q(float(A[r][c])) for c in range(n)] for r in range(m)]
    Ntil = A.T @ A                                   # numpy/OpenBLAS fp64
    G, AbsG = exact_mtm(Aq, m, n)
    gamma_m = gamma_n(m, u)
    adf = a_dot_fwd(m, u, amul)
    worst = Q(0); worst_ij = None
    for i in range(n):
        for j in range(n):
            err = abs(Q(float(Ntil[i][j])) - G[i][j])
            bound = gamma_m * AbsG[i][j] + adf
            if bound == 0:
                continue
            ratio = err / bound
            if ratio > worst:
                worst = ratio; worst_ij = (i, j)
    print(f"  {label:42s} m={m:5d} n={n:4d}  worst err/bound = {float(worst):.4e}  "
          f"{'OK' if worst <= 1 else 'VIOLATED!!'}")
    return worst

np.random.seed(0)
print("Higham entrywise-bound check (numpy/OpenBLAS fp64 A^T A vs exact, bound must hold):")
worst_all = Q(0)
for (m, n) in [(128,64),(512,64),(784,96),(1024,64),(3072,64)]:
    # standard normal
    worst_all = max(worst_all, check(np.random.randn(m,n), f"N(0,1)"))
    # realistic normalized-iterate scale: ||A||_F ~ 1  => entries ~ 1/sqrt(m*n)
    A = np.random.randn(m,n) / np.sqrt(m*n)
    worst_all = max(worst_all, check(A, "normalized (||A||_F~1)"))
    # adversarial near-cancellation: near-equal columns w/ alternating signs
    base = np.random.randn(m,1)
    A = base + 1e-8*np.random.randn(m,n)
    A[::2,:] *= -1
    worst_all = max(worst_all, check(A, "near-cancellation (adversarial)"))
    # wide dynamic range entries
    A = np.random.randn(m,n) * (10.0 ** np.random.randint(-6,6,size=(m,n)))
    worst_all = max(worst_all, check(A, "wide dynamic range"))
print(f"\nGLOBAL worst err/bound across all trials: {float(worst_all):.4e}")
print("=> bound HOLDS with margin" if worst_all <= 1 else "=> BOUND VIOLATED")
