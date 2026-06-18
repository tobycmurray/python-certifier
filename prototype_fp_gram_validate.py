"""Prototype validation: FP-sound Gram iteration (binary64 mtm) vs exact-rational.

Loads the EXACT gram-g spectral-norm bounds (and their recorded compute time) from a
precomputed .norms.json -- so we don't redundantly recompute the expensive exact Gram --
and measures only the new fp64 prototype. For each layer matrix W (op2_norms) and |W|
(op2_abs_norms) it compares:
  - exact = stored gram_iteration(.,g)        (exact rational mtm; Dafny-cross-checked)
  - fp64  = gram_iteration_fp64(.,g)          (prototype: binary64 BLAS mtm + error track)
  - svd   = numpy largest singular value      (high-accuracy ground truth)
Checks soundness (fp64>=svd, exact>=svd), tightness (fp64/exact, fp64/svd), and reports
fp64 wall-clock vs the recorded exact aggregate time.

Run from python-certifier/ with the venv python:
  venv/bin/python3 prototype_fp_gram_validate.py <model.txt> <norms.json> <gram>
"""
import sys
import json
import time
import numpy as np

from arithmetic import Q
from parsing import load_network_from_file
from linear_algebra import gram_iteration_fp64, abs_matrix, dims


def svd_top(W):
    Wn = np.array([[float(x) for x in row] for row in W], dtype=np.float64)
    return float(np.linalg.norm(Wn, 2))


def run(model_path, norms_path, gram):
    net = load_network_from_file(model_path, validate=True)
    nd = json.load(open(norms_path))
    assert nd["gram_iters"] == gram, f"norms file is gram {nd['gram_iters']}, asked {gram}"
    exact_W = [Q(s) for s in nd["op2_norms"]]
    exact_absW = [Q(s) for s in nd["op2_abs_norms"]]
    t_exact_op2 = nd["times_secs"]["op2"]          # aggregate over all layers (||W||_2)
    t_exact_op2abs = nd["times_secs"]["op2_abs"]   # aggregate over all layers (|||W|||_2)

    print(f"model: {model_path}")
    print(f"norms: {norms_path}  (gram {gram})")
    print(f"recorded EXACT time: op2 {t_exact_op2:.1f}s + op2_abs {t_exact_op2abs:.1f}s "
          f"= {t_exact_op2+t_exact_op2abs:.1f}s total (all layers)\n")
    header = (f"{'layer':>5} {'mat':>4} {'dims':>11} {'svd(true)':>15} {'exact(stored)':>17} "
              f"{'fp64':>17} {'fp64/svd':>10} {'fp64/exact':>13} {'t_fp64(s)':>10} {'sound':>6}")
    print(header); print("-" * len(header))

    tot_fp64 = 0.0
    all_sound = True
    worst_tight = 0.0
    for li, W in enumerate(net):
        for label, M, exact in (("W", W, exact_W[li]), ("|W|", abs_matrix(W), exact_absW[li])):
            m, n = dims(M)
            sv = svd_top(M)
            t0 = time.perf_counter(); fp = gram_iteration_fp64(M, gram); t_fp = time.perf_counter() - t0
            tot_fp64 += t_fp
            exf, fpf = float(exact), float(fp)
            sound = (fpf >= sv) and (exf >= sv)
            all_sound = all_sound and sound
            worst_tight = max(worst_tight, fpf / exf)
            print(f"{li:>5} {label:>4} {f'{m}x{n}':>11} {sv:>15.9f} {exf:>17.10f} "
                  f"{fpf:>17.10f} {fpf/sv:>10.6f} {fpf/exf:>13.10f} {t_fp:>10.3f} "
                  f"{'OK' if sound else 'FAIL':>6}")
    print("-" * len(header))
    t_exact = t_exact_op2 + t_exact_op2abs
    print(f"\nfp64 total time: {tot_fp64:.2f}s   vs recorded exact: {t_exact:.1f}s   "
          f"=> speedup ~{t_exact/tot_fp64:.0f}x")
    print(f"all fp64 bounds sound (>= true spectral norm): {all_sound}")
    print(f"worst fp64/exact ratio (error inflation): {worst_tight:.10f}  "
          f"(1.0 = identical; >1 means fp64 looser)")


if __name__ == "__main__":
    model_path = sys.argv[1]
    norms_path = sys.argv[2]
    gram = int(sys.argv[3])
    run(model_path, norms_path, gram)
