#!/usr/bin/env bash
# Provision the HIGGS + EMNIST certifier inputs (HIGGS width sweep + EMNIST-byclass)
# that tests/run_tests.sh consumes but which are gitignored derived artifacts:
# the sound-by-construction certifier .txt weight files AND the per-model
# certification-input directories, from the committed gloro-trained CSV weights
# in the sibling verified-certified-robustness repo (branch higgs-gloro:
# scripts/sweep_results/<tag>/model_weights_csv/).
#
# The AUTHOR runs this once locally BEFORE building the artifact
# (docker/build_artifact.sh, which bakes the resulting inputs into the image and
# fails loudly if they are absent). It is the counterpart of
# models/regenerate.sh (the originals); it mirrors that script AND the proven
# pilot driver scripts/certify_e1_e2.sh, so we reproduce a known-good invocation:
#
#   net.txt  <- make_certifier_format_from_model.py : builds the model via doitlib,
#               loads the exact executing float32 weights, and self-checks by
#               reloading through the certifier's own parser (refuses to emit a
#               file that does not round-trip). Sound by construction.
#   inputs   <- get_all_test_inputs.py : writes per-input orig_*.npy plus an
#               inputs.json carrying each point's logits y1 and true label. The
#               certifier resolves each .npy RELATIVE TO the inputs.json directory
#               (robust_certifier.py: os.path.join(dir(json), basename(x1_file))),
#               so every (model, N) combination gets its OWN directory.
#
# All outputs are large derived artifacts and are gitignored (models/*.txt and
# tests/inputs_*/); this script is the committed source of truth. HIGGS
# standardization is recomputed deterministically from the canonical Baldi split
# (HIGGS_N_TRAIN/HIGGS_N_TEST), identical to what the models were trained with.
#
# Requires the cav2025-artifact venv (TF 2.13 + doitlib + tfds, and the HIGGS +
# EMNIST tfds downloads). Idempotent: any artifact already present is skipped.
#
# Override layout with VCRS=/path/to/verified-certified-robustness/scripts and
# PY=/path/to/python if your checkout differs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # python-certifier/models
ROOT="$(cd "$HERE/.." && pwd)"                          # python-certifier
TESTS="$ROOT/tests"
VCRS="${VCRS:-$ROOT/../verified-certified-robustness/scripts}"
SWEEP="$VCRS/sweep_results"
PY="${PY:-$VCRS/cav2025-artifact-venv/bin/python3}"
GEN="$ROOT/make_certifier_format_from_model.py"
INPUTS_GEN="$VCRS/get_all_test_inputs.py"

# HIGGS canonical Baldi split (last 500k test) -- identical to training.
H_TRAIN="10500000"; H_TEST="500000"

make_net () {  # tag dataset layers isize
  local tag="$1" dataset="$2" layers="$3" isize="$4"
  local csv="$SWEEP/$tag/model_weights_csv"
  local out="$HERE/neural_net_${tag}.txt"
  [ -f "$csv/layer_0_weights.csv" ] || { echo "!! no weights for $tag ($csv)"; return 1; }
  if [ -f "$out" ]; then echo "  net.txt exists, skip: $(basename "$out")"; return 0; fi
  echo ">>> [$tag] make net.txt"
  ( cd "$ROOT" && DOITLIB_DIR="$VCRS" "$PY" "$GEN" "$dataset" "$layers" "$isize" "$csv" "$out" )
}

make_inputs () {  # tag dataset layers isize eps outdir [maxN] [higgs]
  local tag="$1" dataset="$2" layers="$3" isize="$4" eps="$5" outdir="$6" maxN="${7:-}" higgs="${8:-}"
  local csv="$SWEEP/$tag/model_weights_csv"
  mkdir -p "$outdir"
  if [ -f "$outdir/inputs.json" ]; then echo "  inputs exist, skip: $(basename "$outdir")"; return 0; fi
  echo ">>> [$tag] make inputs -> $(basename "$outdir") (n=${maxN:-all})"
  ( cd "$outdir" && env PYTHONPATH="$VCRS" \
      ${higgs:+HIGGS_N_TRAIN="$H_TRAIN"} ${higgs:+HIGGS_N_TEST="$H_TEST"} \
      "$PY" "$INPUTS_GEN" float32 "$dataset" "$layers" "$csv" "$isize" "inputs.json" "$eps" ${maxN:+$maxN} )
}

# ---- HIGGS width sweep (tabular; isize is a dummy -- build_model uses num_features; eps 0.1) ----
for w in 128 256 512 1024; do
  tag="higgs_w${w}_d5_full"
  L="[$w,$w,$w,$w,$w]"
  make_net    "$tag" higgs "$L" 1
  make_inputs "$tag" higgs "$L" 1 0.1 "$TESTS/inputs_higgs_w${w}_n10000" 10000 higgs
done
# HIGGS-1024 also over its FULL 500k canonical test split for the RQ3 master-table
# row (pre-deployment VRA at the entire test distribution). ~1-2h to generate.
make_inputs "higgs_w1024_d5_full" higgs "[1024,1024,1024,1024,1024]" 1 0.1 \
            "$TESTS/inputs_higgs_w1024_n500000" 500000 higgs

# ---- EMNIST byclass (CIFAR architecture; isize 28; eps 0.3) ----
BYC_L="[512,256,128,128,128,128,128,128]"
make_net    "emnistbyc_cifar" emnist/byclass "$BYC_L" 28
make_inputs "emnistbyc_cifar" emnist/byclass "$BYC_L" 28 0.3 "$TESTS/inputs_emnist_byclass_full"  ""

# ---- EMNIST balanced (47-class, width-512 depth-8; isize 28; eps 0.3) ----
# Balanced test set is ~18.8k, so we run all modes over the full set (no subset).
BAL_L="[512,512,512,512,512,512,512,512]"
make_net    "emnistbal_w512_d8_ep500" emnist/balanced "$BAL_L" 28
make_inputs "emnistbal_w512_d8_ep500" emnist/balanced "$BAL_L" 28 0.3 "$TESTS/inputs_emnist_balanced_full" ""

echo "All HIGGS + EMNIST artifacts provisioned and self-checked."
