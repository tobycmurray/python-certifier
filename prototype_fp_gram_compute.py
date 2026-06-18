"""Compute a model's spectral-norm bounds using the fp64-mtm Gram iteration prototype.

Mirrors norms.compute_norms but uses gram_iteration_fp64 (binary64 BLAS mtm + error
tracking) for the two expensive spectral norms (||W||_2 and |||W|||_2). The cheap
max-row norms are computed exactly as usual. Saves results in the standard
.norms.json format so they can be diffed against an exact compute_norms run.

Headline purpose: measure the ABSOLUTE wall-clock time of the fp64 path (the path we
intend to make the default), not its ratio to the unscalable exact path.

Run from python-certifier/ with the venv python:
  venv/bin/python3 prototype_fp_gram_compute.py <model.txt> <gram> <output.norms.json>
"""
import sys
import time
import json

from parsing import load_network_from_file
from linear_algebra import (gram_iteration_fp64, abs_matrix, dims,
                             layer_infinity_norm, max_row_l2_norm)
from norms import Norms, save_norms, hash_file_contents


def compute_norms_fp64(net, gram):
    op2, op2_abs, mri, mrl2 = [], [], [], []
    times = {"max_row_inf": 0.0, "op2": 0.0, "op2_abs": 0.0, "max_row_l2": 0.0}
    nL = len(net)
    for i, W in enumerate(net):
        m, n = dims(W)
        print(f"[layer {i}/{nL-1}] W {m}x{n} -> Gram {n}x{n}", flush=True)

        t0 = time.perf_counter(); mri.append(layer_infinity_norm(W)); times["max_row_inf"] += time.perf_counter() - t0
        t0 = time.perf_counter(); mrl2.append(max_row_l2_norm(W)); times["max_row_l2"] += time.perf_counter() - t0

        t0 = time.perf_counter(); v = gram_iteration_fp64(W, gram); dt = time.perf_counter() - t0
        times["op2"] += dt; op2.append(v)
        print(f"           ||W||_2     = {float(v):.10f}   ({dt:.1f}s, cum op2 {times['op2']/60:.1f}min)", flush=True)

        t0 = time.perf_counter(); v = gram_iteration_fp64(abs_matrix(W), gram); dt = time.perf_counter() - t0
        times["op2_abs"] += dt; op2_abs.append(v)
        print(f"           |||W|||_2   = {float(v):.10f}   ({dt:.1f}s, cum op2_abs {times['op2_abs']/60:.1f}min)", flush=True)

    return Norms(max_row_inf_norms=mri, op2_norms=op2, op2_abs_norms=op2_abs,
                 max_row_l2_norms=mrl2, times=times)


if __name__ == "__main__":
    model_path = sys.argv[1]
    gram = int(sys.argv[2])
    out = sys.argv[3]
    net = load_network_from_file(model_path, validate=True)
    print(f"model: {model_path}  gram: {gram}  layers: {len(net)}", flush=True)
    t_all = time.perf_counter()
    norms = compute_norms_fp64(net, gram)
    elapsed = time.perf_counter() - t_all
    h = hash_file_contents(model_path)
    save_norms(h, gram, "fp64", norms, out)
    print(f"\nTOTAL fp64 norm-compute time: {elapsed:.1f}s = {elapsed/60:.2f}min", flush=True)
    print(f"  op2 {norms.times['op2']/60:.2f}min + op2_abs {norms.times['op2_abs']/60:.2f}min "
          f"+ max_row {(norms.times['max_row_inf']+norms.times['max_row_l2']):.1f}s", flush=True)
    print(f"saved norms -> {out}  (hash {h[:16]}...)", flush=True)
