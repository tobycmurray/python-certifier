#!/usr/bin/env bash
# Provision the *_numpy16.json variants of the six master-table models' "all" input sets: same schema and the
# same orig_*.npy files as the existing sets, but with y1 computed by the IEEE-754-compliant numpy execution of
# the float32-weighted network AT FLOAT16 (compliant_forward policy float16; weights loaded at float32 and cast
# down to float16, the deployment rounding, see get_all_test_inputs_numpy.py). These feed the float16 full-set
# runs (FLOAT16_ALL=1 NUMPY_EXEC=1 tests/run_tests.sh) behind the paper's claim that robustness certification
# at float16 is vacuous on every test set (2026-11-04; the earlier evidence was a Keras-era MNIST run only).
# Mirrors provision_numpy64_all_inputs.sh. Idempotent: existing outputs skipped. Runs the six models in
# parallel (CPU-bound numpy); HIGGS-1024 over its 500k test split is the long one (~20 min at float64).
# Exact invocation (2026-11-04):  nohup bash models/provision_numpy16_all_inputs.sh > models/provision_numpy16_all_inputs.log 2>&1 &
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$HERE/.." && pwd)"; TESTS="$ROOT/tests"
VCRS="${VCRS:-$ROOT/../verified-certified-robustness/scripts}"
CAV="${CAV_MODELS:-$ROOT/../verified-certified-robustness/cav2025-models}"
SWEEP="$VCRS/sweep_results"; PY="${PY:-$ROOT/venv/bin/python}"; GEN="$VCRS/get_all_test_inputs_numpy.py"
H_TRAIN="10500000"; H_TEST="500000"
gen () {  # dataset layers csv isize eps outdir outjson [maxN] [higgs]
  local dataset="$1" layers="$2" csv="$3" isize="$4" eps="$5" outdir="$6" outjson="$7" maxN="${8:-}" higgs="${9:-}"
  [ -f "$csv/layer_0_weights.csv" ] || { echo "!! no weights: $csv"; return 1; }
  [ -d "$outdir" ] || { echo "!! input dir missing: $outdir"; return 1; }
  if [ -f "$outdir/$outjson" ]; then echo "  exists, skip: $outjson"; return 0; fi
  echo ">>> $outjson (n=${maxN:-all})"
  ( cd "$outdir" && env PYTHONPATH="$VCRS" PYTHON_CERTIFIER="$ROOT" \
      ${higgs:+HIGGS_N_TRAIN="$H_TRAIN"} ${higgs:+HIGGS_N_TEST="$H_TEST"} \
      "$PY" "$GEN" float16 "$dataset" "$layers" "$csv" "$isize" "$outjson" "$eps" ${maxN:+$maxN} > "$outdir/${outjson%.json}.provision.log" 2>&1 )
  echo "<<< $outjson done"
}
MNIST_CSV="$CAV/2025-01-25_09:27:46-mnist/model_weights_epsilon_0.45_[128,128,128,128,128,128,128,128]_500"
FASHION_CSV="$CAV/2025-01-30_10:58:01-fashion_mnist/model_weights_epsilon_0.26_[256,128,128,128,128,128,128,128,128,128,128,128]_500"
CIFAR_CSV="$CAV/2025-01-28_20:39:32-cifar10/model_weights_epsilon_0.1551_[512,256,128,128,128,128,128,128]_800"
gen mnist         "[128,128,128,128,128,128,128,128]"                     "$MNIST_CSV"   28 0.3   "$TESTS/all_mnist_test_inputs"         test_inputs_epsilon_0.3_numpy16.json &
gen fashion_mnist "[256,128,128,128,128,128,128,128,128,128,128,128]"     "$FASHION_CSV" 28 0.25  "$TESTS/all_fashion_mnist_test_inputs" test_inputs_epsilon_0.25_numpy16.json &
gen cifar10       "[512,256,128,128,128,128,128,128]"                     "$CIFAR_CSV"   32 0.141 "$TESTS/all_cifar10_test_inputs"       all_test_inputs_numpy16.json &
gen higgs "[1024,1024,1024,1024,1024]" "$SWEEP/higgs_w1024_d5_full/model_weights_csv" 1 0.1 "$TESTS/inputs_higgs_w1024_n500000" inputs_numpy16.json 500000 higgs &
gen emnist/byclass  "[512,256,128,128,128,128,128,128]" "$SWEEP/emnistbyc_cifar/model_weights_csv"        28 0.3 "$TESTS/inputs_emnist_byclass_full"  inputs_numpy16.json &
gen emnist/balanced "[512,512,512,512,512,512,512,512]" "$SWEEP/emnistbal_w512_d8_ep500/model_weights_csv" 28 0.3 "$TESTS/inputs_emnist_balanced_full" inputs_numpy16.json &
wait
echo "All float16 numpy-execution input variants provisioned."
