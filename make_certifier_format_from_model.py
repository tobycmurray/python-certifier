"""Produce a certifier .txt weight file that is SOUND BY CONSTRUCTION.

Unlike the original Tobler et al. script (verified-certified-robustness/scripts/
make_certifier_format.py), which read the CSV weights as float64 and (in its
original Jan-2025 form) rounded them to 5 decimal places, this script writes the
*exact* weights that the deployed Keras model executes with.

It does so the only way that makes soundness obvious: it builds the actual model
via doitlib, loads the weights exactly as the model is run (mixed_precision
float32 policy, load_and_set_weights), and then writes out `model.get_weights()`
-- i.e. the precise float32 values the forward pass multiplies with. Every
float32 is an exact (short) dyadic rational, and `f"{float(w):.150f}"` writes its
exact decimal, which the certifier parses back to the exact rational. Hence the
certifier reasons about *exactly* the model that executes.

Output format matches the original script byte-for-format (nested
[[...],[...]] of rows of one weight matrix per layer, kernels in (in,out)
orientation, weights only -- the models use use_bias=False), so the certifier's
existing parser reads it unchanged.

Run in the TF 2.13 environment that has doitlib on the path, e.g.:
  DOITLIB_DIR=/path/to/verified-certified-robustness/scripts \
  <cav2025-artifact-venv>/bin/python make_certifier_format_from_model.py \
      <dataset> "<INTERNAL_LAYER_SIZES>" <input_size> <csv_dir> <out_txt>
"""
import os
import sys
from fractions import Fraction

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras import mixed_precision

# The certifier's OWN loader -- used by the self-check so we validate the exact
# code path the certifier will use to read this file.
from parsing import load_network_from_file

# doitlib lives in the sibling verified-certified-robustness repo's scripts dir.
# Default is resolved relative to THIS file (not the cwd), so it works regardless
# of where the script is invoked from; override with DOITLIB_DIR if your layout
# differs.
_DOITLIB_DIR = os.environ.get(
    "DOITLIB_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "verified-certified-robustness", "scripts"),
)
sys.path.insert(0, _DOITLIB_DIR)
import doitlib  # noqa: E402


def _exact_rational(q):
    """Exact rational value of a certifier Q (gmpy2.mpq-like) as a Fraction."""
    if hasattr(q, "numerator") and hasattr(q, "denominator"):
        return Fraction(int(q.numerator), int(q.denominator))
    return Fraction(q)


def self_check(out_txt, weight_mats):
    """Reload the just-written .txt through the certifier's real loader and assert
    it encodes EXACTLY the float32 weights the model executes -- i.e. every
    parsed rational equals the exact rational of the corresponding
    model.get_weights() float32 value. Raises SystemExit on any discrepancy so a
    non-sound file can never be produced silently.
    """
    net = load_network_from_file(out_txt)
    if len(net) != len(weight_mats):
        raise SystemExit(
            f"SELF-CHECK FAILED: parsed {len(net)} layers, expected {len(weight_mats)}")
    total = 0
    for l, (parsed, W) in enumerate(zip(net, weight_mats)):
        # The certifier's loader transposes on load, so a Keras kernel written in
        # (in,out) form comes back as (out,in). Compare against W.T accordingly.
        WT = np.asarray(W).T
        a, b = len(parsed), len(parsed[0])
        if (a, b) != tuple(WT.shape):
            raise SystemExit(
                f"SELF-CHECK FAILED: layer {l} parsed shape {(a, b)} != {tuple(WT.shape)} (=W.T)")
        for i in range(a):
            for j in range(b):
                total += 1
                if _exact_rational(parsed[i][j]) != Fraction(float(WT[i, j])):
                    raise SystemExit(
                        f"SELF-CHECK FAILED at layer {l}[{i},{j}]: "
                        f".txt={parsed[i][j]} != exact float32 {float(WT[i, j])!r}")
    print(f"SELF-CHECK PASSED: all {total} weights parse back through the certifier "
          f"loader to the exact float32 values the model executes.")


def main():
    if len(sys.argv) != 6:
        print(f"Usage: {sys.argv[0]} <dataset> <INTERNAL_LAYER_SIZES> "
              f"<input_size> <model_weights_csv_dir> <out_txt>")
        sys.exit(1)

    dataset = sys.argv[1]
    internal_layer_sizes = eval(sys.argv[2])
    input_size = int(sys.argv[3])
    csv_dir = sys.argv[4].rstrip("/") + "/"
    out_txt = sys.argv[5]

    # Build and load the model exactly as it is executed (float32 deployment).
    mixed_precision.set_global_policy("float32")
    inputs, outputs = doitlib.build_model(
        Input, Flatten, Dense,
        input_size=input_size, dataset=dataset,
        internal_layer_sizes=internal_layer_sizes,
    )
    model = Model(inputs, outputs)
    doitlib.load_and_set_weights(csv_dir, internal_layer_sizes, model)

    # Read the EXACT weights the model executes with, in layer order.
    dense_layers = [l for l in model.layers if isinstance(l, Dense)]
    weight_mats = []
    for dl in dense_layers:
        w = dl.get_weights()
        assert len(w) == 1, f"expected use_bias=False, got {len(w)} arrays"
        W = w[0]
        # NOTE: float32 is required, and not only because that is the deployment
        # dtype -- the 150-decimal-place formatting below is exact *only* for
        # float32 (see the comment at the f-string). A float64-weighted model
        # would be silently truncated and the output would NOT be sound.
        assert W.dtype == np.float32, f"executing weights must be float32, got {W.dtype}"
        weight_mats.append(W)  # (in, out), same orientation the old script wrote

    # Write in the original nested-bracket format, exact float32 decimals.
    with open(out_txt, "w") as f:
        for w_idx, mat in enumerate(weight_mats):
            f.write("[")
            n_rows = mat.shape[0]
            for r in range(n_rows):
                f.write("[")
                row = mat[r]
                n = row.shape[0]
                for c in range(n):
                    # Write the EXACT value of the weight as a decimal.
                    #
                    # float(np.float32) is the exact float32 value as a float64
                    # (float32 is a subset of float64, so this is lossless), and
                    # the certifier parses the decimal below back to an exact
                    # rational.
                    #
                    # Why 150 places is sound -- AND ONLY FOR float32:
                    #   Every float32 is an exact integer multiple of 2^-149 (the
                    #   smallest subnormal). A value k/2^n terminates in decimal at
                    #   exactly n places (1/2^n = 5^n/10^n), so every float32's
                    #   exact decimal terminates at <= 149 places. Hence :.150f
                    #   reproduces every float32 exactly, with one digit to spare
                    #   (and does no rounding -- the tail is genuine zeros). 149
                    #   would already suffice; 150 is the same bound rounded up to
                    #   a nice number. These weights (all normal, ~1e-3..2) only
                    #   use ~30-50 places in practice.
                    #   This bound is specific to float32 (enforced by the assert
                    #   above): a float64's smallest subnormal is 2^-1074, so
                    #   float64 weights would need up to 1074 places and 150 would
                    #   silently truncate them.
                    f.write(f"{float(row[c]):.150f}")
                    if c < n - 1:
                        f.write(",")
                f.write("]")
                if r < n_rows - 1:
                    f.write(",")
            f.write("]")
            if w_idx < len(weight_mats) - 1:
                f.write(",")

    total = sum(int(m.size) for m in weight_mats)
    print(f"Wrote {len(weight_mats)} weight matrices ({total} weights) to {out_txt}")
    print("Source: model.get_weights() from the doitlib-built model (exact executing float32 weights).")

    # Sound-by-construction is now ENFORCED, not just claimed: round-trip the file
    # through the certifier's own loader and verify exact rational equality.
    self_check(out_txt, weight_mats)


if __name__ == "__main__":
    main()
