"""Witness-conformance check: verify no flush-to-zero (FTZ) occurs during the
deployed Keras forward pass on the certifier's inputs.

WHY THIS EXISTS
---------------
Our robustness certificates are sound for any execution conforming to the
standard floating-point model (round-to-nearest, gradual underflow). TensorFlow
enables FTZ+DAZ at float32/float64 (it wraps every op in port::ScopedFlushDenormal
on its worker threads; there is no API to disable it -- see the paper's App. E).
A flushing execution still *conforms* at a given point iff it produces no
subnormal there: then FTZ never fires and the run is bit-identical to gradual
underflow. This script certifies exactly that, per witness point.

SOUNDNESS OF THE CHECK
----------------------
A subnormal fp32 value has magnitude in (0, lambda), lambda = 2^-126. We check a
single, order-independent sufficient condition:

    every nonzero product w_ij * z_j has magnitude >= 2^-103,  and
    every nonzero bias entry has magnitude >= 2^-103.

Then every term is an integer multiple of 2^-126, so every partial sum -- in ANY
accumulation order, including TF's opaque one -- is 0 or >= lambda, and the
bias-add preserves this. Hence no intermediate (product, partial sum, or
pre-activation) is subnormal, so FTZ cannot have fired. Because the condition is
on the *products* (elementwise, order-free), checking it certifies TF's summation
without reproducing it.

WHY KERAS ACTIVATIONS, RECOMPUTED IN NUMPY
------------------------------------------
The products are w_ij * z_j with z_j the layer's *input activation*. TF is the
witness that could flush, and a conforming numpy forward pass diverges from TF
(different accumulation order -- the reason the counterexamples don't reproduce
under numpy), so numpy's activations are the *wrong* z_j. We therefore extract
TF's actual per-layer activations (build_activation_model) and recompute the
per-term products from THEM, in numpy on the main thread -- which we verify does
NOT flush (main-thread FPCR stays clear; TF sets FZ only transiently on its
worker threads). Each product is a single multiply, so numpy reproduces TF's
product exactly, EXCEPT that numpy doesn't flush: a product TF flushed shows up
here as a nonzero subnormal and fails the 2^-103 gate. Layer-by-layer, the first
flush anywhere is caught.

A separate pure-numpy conforming forward pass is also run, purely as
corroboration (it must clear the gate too, with its own -- different --
activations).

USAGE
-----
    python check_no_ftz.py FORMAT NETWORK CEX_JSON [--biases FILE] [--limit N]

FORMAT/NETWORK/CEX_JSON/--biases are the same files the certifier consumes.
Exit code 0 iff every witness point clears the gate at every checked precision.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import sys, json, argparse, math
import numpy as np

from parsing import (load_network_from_file, load_vector_from_npy_file,
                     load_biases_from_file)

# fp32 is the deployed/certified precision; fp64 is the hybrid reference. Both
# have FTZ enabled under TF, so both witness executions must be flush-free.
# Per-format: smallest normal LAMBDA, and the sum-closure product threshold
# THRESH = 2^(emin + (p-1)), below which a partial sum could reach a subnormal.
_DT = {"float32": np.float32, "float64": np.float64}
LAMBDA = {"float32": 2.0**-126, "float64": 2.0**-1022}
THRESH = {"float32": 2.0**-103, "float64": 2.0**-970}   # 2^(emin + prec-1)


def canary(dt):
    """Assert this (main-thread) numpy environment preserves subnormals; without
    this a green result is meaningless (a flushing environment would hide the
    very subnormals we look for)."""
    if dt is np.float32:
        a, b = np.float32(2.0**-100), np.float32(2.0**-40)      # exact 2^-140 in (2^-149, 2^-126): subnormal
    else:
        a, b = np.float64(2.0**-525), np.float64(2.0**-525)     # exact 2^-1050 in (2^-1074, 2^-1022): subnormal
    c = a * b
    if c == 0:
        sys.exit("FATAL: numpy is flushing subnormals in this process -- the "
                 "recompute would be unsound. Run in a clean (TF-free thread) env.")
    return float(c)


def matvec_inputs(x, acts, dt):
    """The per-layer matvec input activations: layer 0 sees x; layer l>=1 sees the
    previous layer's output activation."""
    x = np.asarray([float(v) for v in x], dtype=dt)
    return [x] + [a.astype(dt) for a in acts[:-1]]


def scan_products(net_dt, ins, thresh):
    """Min nonzero |product| and its layer, over w_ij * z_j at every layer.
    Products are formed in the arrays' dtype (single multiplies, RN, no FTZ)."""
    m, where = math.inf, None
    for l, (W, z) in enumerate(zip(net_dt, ins)):
        P = W * z[None, :]                 # (out,in): P[i,j] = W[i,j]*z[j]
        nz = np.abs(P[P != 0])
        if nz.size:
            mn = float(nz.min())
            if mn < m:
                m, where = mn, l
    return m, where


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("format", help="deployed float format (float16/float32/float64)")
    ap.add_argument("network", help="certifier network .txt file")
    ap.add_argument("cex", help="certifier input JSON (records with x1_file)")
    ap.add_argument("--biases", default=None, help="certifier bias file (if any)")
    ap.add_argument("--limit", type=int, default=None, help="check only the first N inputs")
    args = ap.parse_args()

    if args.format not in _DT:
        sys.exit(f"this check covers float32/float64 (the certified + reference "
                 f"precisions); got {args.format!r}. float16 is out of scope "
                 f"(vacuous results; separate FZ16 question).")

    net = load_network_from_file(args.network, validate=True)
    biases = load_biases_from_file(args.biases) if args.biases else None

    # The deployed precision to certify, plus the fp64 reference used by the
    # hybrid modes -- both are witness executions that must not flush.
    policies = [args.format] + (["float64"] if args.format != "float64" else [])

    d = os.path.dirname(args.cex)
    with open(args.cex) as f:
        recs = [r for r in json.load(f) if "x1_file" in r]
    if args.limit:
        recs = recs[:args.limit]
    inputs = [os.path.join(d, os.path.basename(r["x1_file"])) for r in recs]
    print(f"network: {args.network}\n{len(net)} layers, {len(inputs)} witness inputs, "
          f"biases={'yes' if biases else 'no'}, precisions={policies}")

    # Bias-magnitude gate (from the exact parameters, precision-independent enough
    # to check per format below).
    from keras_forward import build_activation_model, forward_activations
    from compliant_forward import prepare as np_prepare, forward_activations as np_fwd

    overall_ok = True
    for policy in policies:
        dt = _DT[policy]
        thr, lam = THRESH[policy], LAMBDA[policy]
        canary(dt)
        net_dt = [np.array([[dt(float(v)) for v in row] for row in W], dtype=dt) for W in net]

        # bias gate
        bias_min = math.inf
        if biases is not None:
            for b in biases:
                nz = np.abs(np.array([dt(float(v)) for v in b], dtype=dt))
                nz = nz[nz != 0]
                if nz.size:
                    bias_min = min(bias_min, float(nz.min()))

        kmodel = build_activation_model(net, biases, policy)
        npmodel = np_prepare(net, biases, policy)

        keras_min, keras_where, keras_arg = math.inf, None, None
        numpy_min = math.inf
        act_min = math.inf
        for path in inputs:
            x = load_vector_from_npy_file(path)
            # --- Keras witness trace -> recompute products in numpy ---
            kacts = forward_activations(kmodel, x, policy)
            kins = matvec_inputs(x, kacts, dt)
            km, kw = scan_products(net_dt, kins, thr)
            if km < keras_min:
                keras_min, keras_where, keras_arg = km, kw, os.path.basename(path)
            for a in kacts:
                nz = np.abs(np.asarray(a)[np.asarray(a) != 0])
                if nz.size:
                    act_min = min(act_min, float(nz.min()))
            # --- pure conforming numpy forward (corroboration) ---
            nacts = np_fwd(npmodel, x)
            nins = matvec_inputs(x, nacts, dt)
            nm, _ = scan_products(net_dt, nins, thr)
            numpy_min = min(numpy_min, nm)

        canary(dt)  # re-assert env still preserves subnormals after all TF ops

        def orders(v):
            return "n/a" if v in (math.inf, 0) else f"{math.log10(v/thr):.1f}"
        ok = (keras_min >= thr) and (numpy_min >= thr) and (bias_min >= thr or biases is None)
        overall_ok &= ok
        print(f"\n[{policy}] lambda={lam:.2e}  gate(2^-... )={thr:.2e}")
        print(f"  Keras products : min |w*z| = {keras_min:.3e}  "
              f"({orders(keras_min)} orders above gate)  worst @layer {keras_where} in {keras_arg}")
        print(f"  numpy products : min |w*z| = {numpy_min:.3e}  ({orders(numpy_min)} orders above gate)  [corroboration]")
        print(f"  activations    : min |z|   = {act_min:.3e}")
        if biases is not None:
            print(f"  biases         : min |b|   = {bias_min:.3e}  ({orders(bias_min)} orders above gate)")
        print(f"  --> {'PASS: no subnormal can arise -> no FTZ -> witness conforms' if ok else 'FAIL: gate not cleared'}")

    print("\n" + ("ALL PRECISIONS PASS: no FTZ on any witness point." if overall_ok
                  else "CHECK FAILED on at least one precision/point."))
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
