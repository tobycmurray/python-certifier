# FP64 conv2d Gram iteration for zero-padded CNNs — discussion summary

Working notes from a design discussion. Goal: soundly bound the spectral norm
(Lipschitz constant) of **zero-padded** conv2d layers (typical image CNNs) by
running the expensive conv in binary64 and tracking the rounding error in
otherwise-exact-rational arithmetic — the same hybrid idea already used for dense
layers in `linear_algebra.py` (`mtm_fp64_with_error` / `gram_iteration_fp64`).
Nothing was implemented; this is analysis only.

## Background / context

- The dense fp64 Gram iteration already in the repo runs each `M^T M` in binary64,
  reads it back as exact rationals (lossless), and bounds the rounding error via
  the Higham dot-product model (`gamma_n`, `a_dot_fwd` in `formats.py`), folding it
  into the exact-rational unwind (`delta = t + eps/r`). See `linear_algebra.py`.
- We want the analogue for conv layers, following Delattre's Gram iteration.

## Which algorithm

- Delattre 2023 (arXiv 2305.16173, ICML) — **circular** padding: 2D FFT
  block-diagonalises the conv into n^2 per-frequency c_out×c_in complex matrices;
  Gram-iterate each; spectral norm = max over frequencies. Kernel does NOT grow.
- Delattre 2024 (arXiv 2402.00240) — **zero** padding. Typical image CNNs use zero
  padding (PyTorch `nn.Conv2d` default `padding_mode='zeros'`), so this is the
  target. The relevant routine is **`norm2_toep` (Algorithm 3)**.
- `norm2_toep` is purely **spatial**: no FFT, no per-frequency decomposition, no
  complex arithmetic. Each Gram step is a spatial self-convolution
  `K <- conv2d(K, K, padding = kernel_size(K) - 1)` (the kernel-domain form of
  T*T). Norm read off via a Schur bound `sqrt(||.||_1 * ||.||_inf)` on channel
  abs-sums. All real; only irrational ops are roots (handled by `sqrt_upper`).

## Existing exact-arithmetic prototype

We already had an exact algorithm `certified_norm2_toep` (integer fixed-point
kernel at scale 2^s, exact integer self-conv via CRT/NTT, power-of-two rescale,
truncation residual tracked via the Schur bound, backward unwind
`v <- sqrt_upper(r*(v+e))`, plus initial rounding term e0).

## Conclusion: adapt it to fp64, don't start clean

The skeleton IS the Gram-iteration-with-tracked-error pattern (same shape as
`gram_iteration_fp64`). Three targeted swaps:

1. **Engine:** `exact_gram_conv2d_crt_ntt(A)` -> `conv_gram_fp64(A)`: compute
   `Btil = fl(conv2d(A,A,pad=k-1) contracted over out-channels)` in binary64 as
   **explicit sums-of-products (im2col+gemm / einsum), NOT an FFT-conv** (an
   FFT-conv would smuggle in FFT rounding + twiddle + library trust). Read back
   lossless as rationals; `isfinite` overflow guard.
2. **Representation:** drop the fixed-point 2^s integer scaffolding and CRT/NTT;
   keep the iterate on the fp64 grid held as exact rationals (like the dense path).
   Consequence: `e0 = 0` for real CNNs (fp32 ⊂ fp64, lossless start); and with a
   power-of-two rescale `r` and no underflow the truncation residual `t_trunc = 0`
   (division by 2^q is exact on the fp64 grid), so the only per-iteration error is
   the conv rounding.
3. **Per-iteration error:** `e = t_trunc + eps_conv / r` (truncation post-rescale +
   conv error pre-rescale / r) — the conv analogue of the dense `delta = t + eps/r`.

Everything else — `choose_power_of_two_scale`, the Schur `sqrt_upper(col*row)`
read-outs, the backward unwind, the `(r,e)` history — is unchanged.

## The E_ij Higham bound the error accounting relies on

Exact Gram kernel: `B_{i,i',δ} = sum_{j,p} A_{j,i,p} A_{j,i',p+δ}`, a dot product
of length `m_{i,i',δ} = C_out * overlap(δ) <= m := C_out * K_h * K_w`.
Binary64 forward-error (any RN summation order, gradual underflow):

    |Btil_{i,i',δ} - B_{i,i',δ}| <= gamma_m * sum_{j,p}|A_{j,i,p}||A_{j,i',p+δ}|
                                    + a_dot_fwd(m)                     =: E_{i,i',δ}

with `gamma_m = gamma_n(m,u) = (1+u)^m - 1`,
`a_dot_fwd(m) = (1+gamma_{m-1}) * m * a_mul`, `a_mul = denorm_min/2`
(matching `formats.py` / `mtm_fp64_with_error`).

**Key efficiency point — never materialise E.** The Schur read-out needs only the
channel abs-masses of E, and the shift-sum decouples (each spatial pair (p,p')
occurs once as δ=p'-p), so:

    sum_{i,δ} E_{i,i',δ} = gamma_m * sum_j L_j * ell_{j,i'}  +  a_dot_fwd(m) * n_col
    ell_{j,i} = sum_p |A_{j,i,p}|,   L_j = sum_i ell_{j,i}
    n_col = C_in * (2K_h - 1)(2K_w - 1)

The ell/L are cheap L1 reductions of |A| (kernel-size work, no conv, no bignum).
Then `eps_conv = sqrt_upper(colmass(E) * rowmass(E))` (colmass = rowmass by
self-adjointness of T*T, but keep both). This is the conv analogue of the dense
`eps = gamma_m*||A||_F^2 + a_dot_fwd(m)*n`, with the Frobenius aggregate replaced
by the Schur channel-sum aggregate the algorithm already uses.

## Complexity — the decisive finding

Kernel spatial size **doubles every iteration**: `s_t = 2^t (k-1) + 1`
(k=3: 3,5,9,17,33,65,129,...). Exact MAC count of the self-conv at iteration t:

    MACs_t = C^3 * s_t^4        (C = channel count)

- Compute dominated by the last iteration ~ `C^3 * s_{N-1}^4 = Θ(C^3 * 16^N)` (k=3).
- Memory ~ `Θ(C^2 * 4^N)` (final conv output can be streamed).

Per-layer estimates, k=3, C=256 (×~8 compute / ×4 mem for C=512):

| N | s_{N-1} | MACs | gemm ~1e11/s | no-BLAS ~5e9/s | peak mem |
|---|--------|------|--------------|----------------|----------|
| 3 | 9  | 1.1e11 | ~1 s   | ~20 s  | ~40 MB          |
| 4 | 17 | 1.4e12 | ~14 s  | ~5 min | ~150 MB→600 MB  |
| 5 | 33 | 2.0e13 | ~3 min | ~1 hr  | ~570 MB→2.3 GB  |
| 6 | 65 | 3.0e14 | ~50 min| ~17 hr | ~2.2 GB→8.9 GB  |

Feasibility (per layer; layers are independent -> parallelisable):
- N<=3: easy for all typical CIFAR channel counts, even no-BLAS.
- N=4: feasible with gemm (tens of s); no-BLAS = minutes–tens of minutes.
- N>=5: compute AND memory blow up; marginal at moderate C w/ gemm, infeasible at
  C=512 or N=6.
- k=1 (1x1) convs don't grow (s_t≡1) -> reduce to dense Gram iteration, ~free.

**The geometric blowup is intrinsic to the SPATIAL norm2_toep formulation and is
independent of fp64-vs-exact.** fp64 buys the constant factor (machine flop vs
growing-bit-length bignum NTT), making small N fast, but does NOT cure the
exponential-in-N scaling. So the fp64 conv trick pays off exactly where the
algorithm is usable (N<=4).

## Strategic recommendation / open question

Everything hinges on **the minimum N for an acceptably tight zero-pad bound** on
the target models (Lipschitz looseness compounds multiplicatively across layers).

- If N<=4 suffices -> fp64 spatial `norm2_toep` is feasible and the fp64 trick is a
  clean win (all-real, simple Higham dot-product bound, no FFT/twiddle/complex).
- If N>=5 needed -> spatial route is the wrong vehicle regardless of arithmetic.
  Escapes (worse soundness ergonomics): (a) fp64 FFT-based conv — restores
  O(C^3 s^2 log s) but needs FFT rounding analysis + twiddle enclosure + looser
  bound; (b) circular route (Delattre 2023 / Thm 6 of 2024) — no kernel growth,
  cost O(N n^2 C^3) ~ 1e11 for CIFAR (trivial), but needs complex-arithmetic +
  twiddle-transcendental soundness work plus the exact-rational
  (1/(1-alpha))^{2^-t} correction.

**Suggested next experiment before any proof work:** run Delattre's reference
`norm2_toep` in plain float on 1–2 representative CIFAR conv layers at N=2..6,
compare against a power-iteration lower bound on the true ||T||_2, and see where
the bound stabilises. That tightness-vs-N curve decides spatial-fp64 vs circular.

## Soundness obligations (to discharge next, mostly already established in-repo)

1. Higham forward-error model for the conv dot products — valid for any RN
   summation order; binding constraint: conv is explicit sums-of-products, NOT an
   FFT-conv. (Same assumption as `mtm_fp64_with_error`.)
2. Shift-decoupling identity sum_δ sum_p |A_{j,i,p}||A_{j,i',p+δ}| = ell_{j,i}·ell_{j,i'} (exact).
3. Schur test for zero-padded multi-channel conv operators: max in/out channel
   abs-sums bound the induced 1- and inf-norms (already relied on in the exact
   algorithm; carries to the residual operators T_E, T_D).
4. Lossless read-back + isfinite overflow guard on Btil.
5. Unwind `v <- sqrt_upper(r*(v+e))` with e >= ||A_next - T_{B_exact}/r||_2, via the
   triangle split e = t_trunc + eps_conv/r.

## References

- Delattre et al., "Efficient Bound of Lipschitz Constant for Convolutional Layers
  by Gram Iteration", ICML 2023 — arXiv:2305.16173. (circular / FFT route)
- Delattre et al., "Spectral Norm of Convolutional Layers with Circular and Zero
  Paddings", 2024 — arXiv:2402.00240. (norm2_toep Algorithm 3; Theorem 6 bridge)
- Higham, "Accuracy and Stability of Numerical Algorithms", 2nd ed — Ch. 3
  (dot-product / gamma_n bounds), Ch. 24 (FFT error, if the FFT route is taken).
- In-repo: `linear_algebra.py` (mtm_fp64_with_error, gram_iteration_fp64,
  _gram_unwind), `formats.py` (gamma_n, a_dot, a_dot_fwd).
