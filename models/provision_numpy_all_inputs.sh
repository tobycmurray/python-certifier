#!/usr/bin/env bash
# Provision the *_numpy.json variants of every "all" certification-input set in
# tests/run_tests.sh's matrix: same schema and same orig_*.npy files as the
# existing Keras-generated sets, but with y1 computed by the IEEE-754-compliant
# numpy execution (compliant_forward) instead of Keras/TF (which runs FTZ here).
#
# The variants are written INTO the existing input directories under a distinct
# name (suffix _numpy), because robust_certifier.py resolves each x1_file
# RELATIVE to the JSON's own directory -- so both variants provably share the
# same .npy inputs (get_all_test_inputs_numpy.py verifies equality and refuses
# to overwrite any existing .npy). Consume them with robust_certifier's
# --numpy-exec flag (hybrid modes measure from the same compliant execution
# that produced these y1).
#
# Also generates, unlike the Keras provisioning:
#   - BIASED-model variants (_numpy_biased_<B>_end.json) for the three image
#     models, with y1 from the biased numpy execution (BIASES_FILE = the same
#     biases.txt the certifier is given via --biases). The Keras matrix reuses
#     the natural y1 for the biased "all" runs; the numpy variants carry the
#     biased model's own logits.
#   - first-100 slices (*_first100_numpy.json), by slicing the full numpy sets
#     (same points as ERAN's 100).
#
# Requires: the existing input dirs (provision_image... / provision_higgs_emnist...
# already run), the python-certifier venv, and the sibling
# verified-certified-robustness checkout. Idempotent: existing outputs skipped.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # python-certifier/models
ROOT="$(cd "$HERE/.." && pwd)"                          # python-certifier
TESTS="$ROOT/tests"
VCRS="${VCRS:-$ROOT/../verified-certified-robustness/scripts}"
CAV="${CAV_MODELS:-$ROOT/../verified-certified-robustness/cav2025-models}"
SWEEP="$VCRS/sweep_results"
PY="${PY:-$ROOT/venv/bin/python}"
GEN="$VCRS/get_all_test_inputs_numpy.py"

# HIGGS canonical Baldi split (last 500k test) -- identical to training.
H_TRAIN="10500000"; H_TEST="500000"

gen () {  # dataset layers csv isize eps outdir outjson [maxN] [higgs] [biases]
  local dataset="$1" layers="$2" csv="$3" isize="$4" eps="$5" outdir="$6" outjson="$7"
  local maxN="${8:-}" higgs="${9:-}" biases="${10:-}"
  [ -f "$csv/layer_0_weights.csv" ] || { echo "!! no weights: $csv"; return 1; }
  [ -d "$outdir" ] || { echo "!! input dir missing (provision the Keras set first): $outdir"; return 1; }
  if [ -f "$outdir/$outjson" ]; then echo "  exists, skip: $outjson"; return 0; fi
  echo ">>> $outjson (n=${maxN:-all})"
  ( cd "$outdir" && env PYTHONPATH="$VCRS" PYTHON_CERTIFIER="$ROOT" \
      ${higgs:+HIGGS_N_TRAIN="$H_TRAIN"} ${higgs:+HIGGS_N_TEST="$H_TEST"} \
      ${biases:+BIASES_FILE="$biases"} \
      "$PY" "$GEN" float32 "$dataset" "$layers" "$csv" "$isize" "$outjson" "$eps" ${maxN:+$maxN} )
}

first100 () {  # dir full_json out_json
  local dir="$1" full="$2" out="$3"
  if [ -f "$dir/$out" ]; then echo "  exists, skip: $out"; return 0; fi
  [ -f "$dir/$full" ] || { echo "!! missing $dir/$full"; return 1; }
  echo ">>> $out (slice of $full)"
  "$PY" - "$dir/$full" "$dir/$out" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
with open(sys.argv[2], "w") as f:
    json.dump(data[:100], f, indent=2)
PYEOF
}

# ---- image models (cav2025 CSV weights; same eps as the Keras sets) ----
MNIST_CSV="$CAV/2025-01-25_09:27:46-mnist/model_weights_epsilon_0.45_[128,128,128,128,128,128,128,128]_500"
FASHION_CSV="$CAV/2025-01-30_10:58:01-fashion_mnist/model_weights_epsilon_0.26_[256,128,128,128,128,128,128,128,128,128,128,128]_500"
CIFAR_CSV="$CAV/2025-01-28_20:39:32-cifar10/model_weights_epsilon_0.1551_[512,256,128,128,128,128,128,128]_800"
MNIST_L="[128,128,128,128,128,128,128,128]"
FASHION_L="[256,128,128,128,128,128,128,128,128,128,128,128]"
CIFAR_L="[512,256,128,128,128,128,128,128]"

gen mnist         "$MNIST_L"   "$MNIST_CSV"   28 0.3   "$TESTS/all_mnist_test_inputs"         test_inputs_epsilon_0.3_numpy.json
gen fashion_mnist "$FASHION_L" "$FASHION_CSV" 28 0.25  "$TESTS/all_fashion_mnist_test_inputs" test_inputs_epsilon_0.25_numpy.json
gen cifar10       "$CIFAR_L"   "$CIFAR_CSV"   32 0.141 "$TESTS/all_cifar10_test_inputs"       all_test_inputs_numpy.json

# ---- biased image models (natural weights + adversarial biases; y1 = biased numpy execution) ----
gen mnist         "$MNIST_L"   "$MNIST_CSV"   28 0.3   "$TESTS/all_mnist_test_inputs"         test_inputs_epsilon_0.3_numpy_biased_1e6_end.json   "" "" "$TESTS/cex_mnist_float32_biased_1e6_end/biases.txt"
gen fashion_mnist "$FASHION_L" "$FASHION_CSV" 28 0.25  "$TESTS/all_fashion_mnist_test_inputs" test_inputs_epsilon_0.25_numpy_biased_3e6_end.json  "" "" "$TESTS/cex_fashion_mnist_float32_biased_3e6_end/biases.txt"
gen cifar10       "$CIFAR_L"   "$CIFAR_CSV"   32 0.141 "$TESTS/all_cifar10_test_inputs"       all_test_inputs_numpy_biased_4e6_end.json           "" "" "$TESTS/cex_cifar10_float32_biased_4e6_end/biases.txt"

# ---- HIGGS width sweep (tabular; isize dummy; eps 0.1) ----
for w in 128 256 512 1024; do
  gen higgs "[$w,$w,$w,$w,$w]" "$SWEEP/higgs_w${w}_d5_full/model_weights_csv" 1 0.1 \
      "$TESTS/inputs_higgs_w${w}_n10000" inputs_numpy.json 10000 higgs
done
# HIGGS-1024 full 500k canonical test split (RQ3 master-table row). Slow (hours).
gen higgs "[1024,1024,1024,1024,1024]" "$SWEEP/higgs_w1024_d5_full/model_weights_csv" 1 0.1 \
    "$TESTS/inputs_higgs_w1024_n500000" inputs_numpy.json 500000 higgs

# ---- EMNIST byclass (CIFAR architecture) + balanced (w512 d8); eps 0.3 ----
gen emnist/byclass  "[512,256,128,128,128,128,128,128]" "$SWEEP/emnistbyc_cifar/model_weights_csv"        28 0.3 "$TESTS/inputs_emnist_byclass_full"  inputs_numpy.json
gen emnist/balanced "[512,512,512,512,512,512,512,512]" "$SWEEP/emnistbal_w512_d8_ep500/model_weights_csv" 28 0.3 "$TESTS/inputs_emnist_balanced_full" inputs_numpy.json

# ---- first-100 slices (ERAN same-100 comparison; natural models only, like run_tests.sh) ----
first100 "$TESTS/all_mnist_test_inputs"         test_inputs_epsilon_0.3_numpy.json  test_inputs_epsilon_0.3_first100_numpy.json
first100 "$TESTS/all_fashion_mnist_test_inputs" test_inputs_epsilon_0.25_numpy.json test_inputs_epsilon_0.25_first100_numpy.json
first100 "$TESTS/all_cifar10_test_inputs"       all_test_inputs_numpy.json          all_test_inputs_first100_numpy.json
first100 "$TESTS/inputs_emnist_byclass_full"    inputs_numpy.json                   inputs_first100_numpy.json
first100 "$TESTS/inputs_emnist_balanced_full"   inputs_numpy.json                   inputs_first100_numpy.json

echo "All numpy-execution input variants provisioned."
