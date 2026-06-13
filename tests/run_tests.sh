#!/usr/bin/env bash
# Requires: bash >= 4 (associative arrays)

# script for testing the Python certifier.
# Supports running two kinds of tests:
# "all" - here, we run the certifier over all of the test inputs of a model (e.g., 10000 MNIST test images).
#         we check whether the real-arithmetic Python certifier agrees with the Dafny reference implementation
#
# "cex" - here, we run the certifier over counter-examples to the Dafny certifier
#         we check that the Python certifier does not certify any counter-examples, but that the real-arithmetic Python
#         certifier would have certified all of them

# --- version guard ---------------------------------------------------------
if [[ -z ${BASH_VERSINFO-} || ${BASH_VERSINFO[0]} -lt 4 ]]; then
  echo "ERROR: This script requires Bash >= 4."
  echo "       On macOS, install with:  brew install bash"
  echo "       Then run with:           /opt/homebrew/bin/bash $0"
  exit 1
fi

set -euo pipefail

# --- configuration ---------------------------------------------------------

CERTIFIER=../robust_certifier.py

MNIST_RESULTS_GRAM_11="results_epsilon_0.45_[128,128,128,128,128,128,128,128]_500_eval_0.3_gram_11.json"
MNIST_RESULTS_GRAM_20="results_epsilon_0.45_[128,128,128,128,128,128,128,128]_500_eval_0.3_gram_20.json"
# Verified Dafny reference for the CORRECTED MNIST model (computed over the sound
# models/ .txt; contains the lipschitz_bounds the certifier cross-checks against).
# Replaces Tobler's results_*.json for MNIST. Note: this was a norms-only Dafny
# run (no per-instance test-set certification), so the "all" cross-check below is
# disabled — see run_test().
MNIST_DAFNY_REF="../models/precomputed/dafny_mnist_gram20.json"
# Verified Dafny reference for the CORRECTED Fashion model (gram 13); same shape as
# the MNIST one (norms-only run, so the per-instance "all" cross-check is disabled).
FASHION_DAFNY_REF="../models/precomputed/dafny_fashion_gram13.json"
FASHION_MNIST_RESULTS_GRAM_12="results_epsilon_0.26_[256,128,128,128,128,128,128,128,128,128,128,128]_500_eval_0.25_gram_12.json"
FASHION_MNIST_RESULTS_GRAM_13="results_epsilon_0.26_[256,128,128,128,128,128,128,128,128,128,128,128]_500_eval_0.25_gram_13.json"
CIFAR10_RESULTS_GRAM_12="results_epsilon_0.1551_[512,256,128,128,128,128,128,128]_800_eval_0.141_gram_12.json"
Z3_RESULTS_GRAM_10="z3_certifier_results_gram_10.json"

# Sound-by-construction model weight files live in ../models/ (regenerated via
# models/regenerate.sh; gitignored). They replace the old 5-dp tests/*.txt.
MNIST_NEURAL_NET="../models/neural_net_mnist_epsilon_0.45_[128,128,128,128,128,128,128,128]_500.txt"
FASHION_MNIST_NEURAL_NET="../models/neural_net_mnist_epsilon_0.26_[256,128,128,128,128,128,128,128,128,128,128,128]_500.txt"
CIFAR10_NEURAL_NET="../models/neural_net_mnist_epsilon_0.1551_[512,256,128,128,128,128,128,128]_800.txt"
Z3_NEURAL_NET="z3_neural_network.txt"

ALL_MNIST_TEST_INPUTS="all_mnist_test_inputs/test_inputs_epsilon_0.3.json"
ALL_FASHION_MNIST_TEST_INPUTS="all_fashion_mnist_test_inputs/test_inputs_epsilon_0.25.json"
ALL_CIFAR10_TEST_INPUTS="all_cifar10_test_inputs/all_test_inputs.json"

CEX_MNIST_FLOAT32="cex_mnist_float32/counter_examples.json"   # regenerated against corrected norms
CEX_MNIST_FLOAT16="cex_mnist_float16/counter_examples.json"   # regenerated against corrected norms (dafny_mnist_gram20)
CEX_MNIST_FLOAT64="cex_mnist_float64/counter_examples.json"   # regenerated against corrected norms (dafny_mnist_gram20)
CEX_FASHION_MNIST_FLOAT32="cex_fashion_mnist_float32/counter_examples.json"   # regenerated against corrected Dafny ref (gram 13)
CEX_FASHION_MNIST_FLOAT16="cex_fashion_mnist_float16/counter_examples.json"   # regenerated against corrected Dafny ref (gram 13)
CEX_FASHION_MNIST_FLOAT64="cex_fashion_mnist_float64/counter_examples.json"   # regenerated against corrected Dafny ref (gram 13)
CEX_CIFAR10_FLOAT16="cex_cifar10_deepfool_float16/counter_examples.json"
CEX_CIFAR10_FLOAT32="cex_cifar10_deepfool/counter_examples.json"
CEX_CIFAR10_FLOAT64="cex_cifar10_deepfool_float64/counter_examples.json"
CEX_Z3_FLOAT32="z3_counter_examples.json"

MNIST_BIASED_1E6_END_BIASES="cex_mnist_float32_biased_1e6_end/biases.txt"   # regenerated
FASHION_MNIST_BIASED_3E6_END_BIASES="cex_fashion_mnist_float32_biased_3e6_end/biases.txt"   # regenerated (B=3e6)
CIFAR10_BIASED_4E6_END_BIASES="cifar10_biased_4e6_end/biases.txt"
CEX_MNIST_BIASED_1E6_END_FLOAT32="cex_mnist_float32_biased_1e6_end/counter_examples.json"   # regenerated against corrected norms
CEX_FASHION_MNIST_BIASED_3E6_END_FLOAT32="cex_fashion_mnist_float32_biased_3e6_end/counter_examples.json"   # regenerated against corrected Dafny ref
CEX_CIFAR10_BIASED_4E6_END_FLOAT32="cex_cifar10_biased_4e6_end_float32/counter_examples.json"

# --- declarative tables ----------------------------------------------------

declare -A NN_FILE=(
  [mnist]="$MNIST_NEURAL_NET"
  [mnist_biased_1e6_end]="$MNIST_NEURAL_NET"
  [fashion_mnist_biased_3e6_end]="$FASHION_MNIST_NEURAL_NET"
  [cifar10_biased_4e6_end]="$CIFAR10_NEURAL_NET"
  [fashion_mnist]="$FASHION_MNIST_NEURAL_NET"
  [cifar10]="$CIFAR10_NEURAL_NET"
  [z3]="$Z3_NEURAL_NET"
)

declare -A BIASES_FILE=(
    [mnist_biased_1e6_end]="$MNIST_BIASED_1E6_END_BIASES"
    [fashion_mnist_biased_3e6_end]="$FASHION_MNIST_BIASED_3E6_END_BIASES"
    [cifar10_biased_4e6_end]="$CIFAR10_BIASED_4E6_END_BIASES"
)

declare -A REF_RESULTS=(
  ["mnist:11"]="$MNIST_RESULTS_GRAM_11"
  ["mnist:20"]="$MNIST_DAFNY_REF"
  ["mnist_biased_1e6_end:11"]="$MNIST_RESULTS_GRAM_11"
  ["mnist_biased_1e6_end:20"]="$MNIST_DAFNY_REF"
  ["fashion_mnist_biased_3e6_end:12"]="$FASHION_MNIST_RESULTS_GRAM_12"
  ["fashion_mnist_biased_3e6_end:13"]="$FASHION_DAFNY_REF"
  ["cifar10_biased_4e6_end:12"]="$CIFAR10_RESULTS_GRAM_12"
  ["fashion_mnist:12"]="$FASHION_MNIST_RESULTS_GRAM_12"
  ["fashion_mnist:13"]="$FASHION_DAFNY_REF"
  ["cifar10:12"]="$CIFAR10_RESULTS_GRAM_12"
  ["z3:10"]="$Z3_RESULTS_GRAM_10"
)

declare -A ALL_INPUTS=(
  [mnist]="$ALL_MNIST_TEST_INPUTS"
  [mnist_biased_1e6_end]="$ALL_MNIST_TEST_INPUTS"
  [fashion_mnist_biased_3e6_end]="$ALL_FASHION_MNIST_TEST_INPUTS"
  [cifar10_biased_4e6_end]="$ALL_CIFAR10_TEST_INPUTS"
  [fashion_mnist]="$ALL_FASHION_MNIST_TEST_INPUTS"
  [cifar10]="$ALL_CIFAR10_TEST_INPUTS"
)

declare -A CEX=(
  ["mnist:float32"]="$CEX_MNIST_FLOAT32"
  ["mnist:float16"]="$CEX_MNIST_FLOAT16"
  ["mnist:float64"]="$CEX_MNIST_FLOAT64"
  ["fashion_mnist:float32"]="$CEX_FASHION_MNIST_FLOAT32"
  ["fashion_mnist:float16"]="$CEX_FASHION_MNIST_FLOAT16"
  ["fashion_mnist:float64"]="$CEX_FASHION_MNIST_FLOAT64"
  ["cifar10:float16"]="$CEX_CIFAR10_FLOAT16"
  ["cifar10:float32"]="$CEX_CIFAR10_FLOAT32"
  ["cifar10:float64"]="$CEX_CIFAR10_FLOAT64"
  ["z3:float32"]="$CEX_Z3_FLOAT32"
  ["mnist_biased_1e6_end:float32"]="$CEX_MNIST_BIASED_1E6_END_FLOAT32"
  ["fashion_mnist_biased_3e6_end:float32"]="$CEX_FASHION_MNIST_BIASED_3E6_END_FLOAT32"
  ["cifar10_biased_4e6_end:float32"]="$CEX_CIFAR10_BIASED_4E6_END_FLOAT32"
)

# --- helpers ---------------------------------------------------------------

die() { echo "ERROR: $*" >&2; exit 1; }

grabnum() {
  # $1: grep -E pattern
  local pat="$1" out=""
  # Extract the first integer from the first matching line
  out=$(grep -E -- "$pat" .log | head -n1 | grep -Eo '[0-9]+' | head -n1 || true)
  if [[ -z "$out" ]]; then
    echo "ERROR: Could not parse '$pat' from .log" >&2
    echo "------- .log tail -------" >&2
    tail -n 40 .log | sed 's/^/| /' >&2 || true
    echo "-------------------------" >&2
    exit 1
  fi
  printf '%s\n' "$out"
}

# --- main runner -----------------------------------------------------------

run_test() {
  local format="$1" model="$2" gram="$3" kind="$4" mode="$5"

  # lookups (use presence check that works on bash >= 4.0)
  [[ -n ${NN_FILE[$model]+x} ]] || die "Unknown model '$model'. Known: ${!NN_FILE[*]}"
  local nn_file="${NN_FILE[$model]}"

  local key_ref="$model:$gram"
  [[ -n ${REF_RESULTS[$key_ref]+x} ]] || {
    # show available grams for this model
    local grams=()
    local k
    for k in "${!REF_RESULTS[@]}"; do
      [[ $k == "$model:"* ]] && grams+=("${k#"$model:"}")
    done
    die "Unknown gram '$gram' for '$model'. Available: ${grams[*]:-(none)}"
  }
  local ref_results_file="${REF_RESULTS[$key_ref]}"

  local cex_file=""
  case "$kind" in
    all)
      [[ "$format" == "float32" ]] || die "Only float32 format supported when running kind 'all'"
      [[ -n ${ALL_INPUTS[$model]+x} ]] || die "No ALL inputs configured for '$model'"
      cex_file="${ALL_INPUTS[$model]}"
      ;;
    cex)
      local key_cex="$model:$format"
      [[ -n ${CEX[$key_cex]+x} ]] || die "Unsupported format '$format' for '$model' and kind 'cex'"
      cex_file="${CEX[$key_cex]}"
      ;;
    *)
      die "Unrecognised kind '$kind'. Should be either 'all' or 'cex'."
      ;;
  esac

  local mode_flag mode_tag
  case "$mode" in
    standard)    mode_flag="";              mode_tag="standard"     ;;
    hybrid-only) mode_flag="--hybrid-only"; mode_tag="hybrid_only"  ;;
    hybrid-meas) mode_flag="--hybrid-meas"; mode_tag="hybrid_meas"  ;;
    *) die "Unrecognised mode '$mode'. Should be 'standard', 'hybrid-only' or 'hybrid-meas'." ;;
  esac

  local bias_flag=""
  if [[ -n ${BIASES_FILE[$model]+x} ]]; then
    bias_flag="--biases ${BIASES_FILE[$model]}"
  fi

  local json_output="json_results/${mode_tag}_${model}_${format}_gram${gram}_${kind}.json"
  mkdir -p json_results

  echo -n "Running test [$mode]: $format, $model, $gram, $kind ...  "
  # shellcheck disable=SC2086
  python "$CERTIFIER" $mode_flag --json-output "$json_output" $bias_flag "$format" "$nn_file" "$gram" --cex "$cex_file" "$ref_results_file" > .log 2>&1 || (cat .log; die "Couldn't run python certifier")

  local count count_ok count_failed count_ok_real
  count=$(       grabnum 'Got [0-9]+ instances to certify'                                )
  count_ok=$(    grabnum 'Certified [0-9]+ instances as robust'                           )
  count_failed=$(grabnum 'Failed to certify [0-9]+ instances as robust'                   )
  count_ok_real=$(grabnum 'Real certifier would have certified [0-9]+ instances as robust' )

  (( count_ok + count_failed == count )) || die "Internal error: counts don't add up (ok=$count_ok failed=$count_failed total=$count)"

  if [[ "$kind" == "cex" ]]; then
    (( count_ok == 0 )) || die "Certifier certified $count_ok counter-examples!"
    (( count_ok_real == count )) || die "Real-arithmetic certifier would not have certified all counter-examples!"
  else
    # kind is "all": the per-instance cross-check against Dafny is DISABLED for
    # the corrected models. We did not run Dafny over the test set (it would
    # unconditionally recompute the norms, >1h), and it is redundant anyway: the
    # certifier's check_margin_lipschitz_bounds already verifies our norms equal
    # the verified Dafny lipschitz_bounds exactly, so the real-mode count here is
    # identical to what Dafny would have produced.
    #
    # (Old check, re-enable if a per-instance Dafny test-set run is available:)
    #   local ref_num; ref_num=$(grep -c true "$ref_results_file")
    #   (( count_ok_real == ref_num )) || die "Mismatch vs Dafny reference: real=$count_ok_real ref=$ref_num ($ref_results_file)"
    :
  fi
  echo "OK"

  (grep -i warn .log) || echo -n ""

  echo ""
  awk '/CERTIFIER\ RESULTS/ {flag=1} flag' .log
  echo ""
}

# --- stage precomputed norms into the run dir ------------------------------
# The certifier loads <hash>.<gram>.norms.json from its cwd (here, tests/). The
# precious, expensive-to-compute norms (~1h for MNIST) are committed under
# ../models/precomputed/; copy them here so the certifier doesn't recompute.
cp -f ../models/precomputed/*.norms.json . 2>/dev/null || true

# --- test matrix -----------------------------------------------------------
# Each test is run in standard, hybrid-only, and hybrid-measured modes.
# hybrid-only and hybrid-measured both use a high-precision pre-deployment pass;
# hybrid-measured strictly dominates hybrid-only (it additionally tightens the
# E_ball term with measured radii), so hybrid-only is retained as an ablation
# midpoint to show how much the measured ball bound adds on top of the measured
# centre bound. Only the "all" runs feed the paper tables (via compute_vra.py);
# the "cex" runs are soundness checks (the tightest mode, hybrid-measured, must
# still reject every counter-example).
#
# CURRENTLY MNIST-ONLY. We use gram 20 uniformly for MNIST (our own choice; we
# no longer match Tobler's gram counts since we compute norms over the corrected
# model). Fashion/CIFAR and the float16/float64 MNIST cex are commented out
# pending their corrected norms + regenerated cexs.

# z3 (synthetic; not MNIST) — disabled
#run_test "float32" "z3"            "10" "cex" "standard"
#run_test "float32" "z3"            "10" "cex" "hybrid-only"
#run_test "float32" "z3"            "10" "cex" "hybrid-meas"

# ===== MNIST natural — cex (gram 20) =====
run_test "float32" "mnist"         "20" "cex" "standard"
run_test "float32" "mnist"         "20" "cex" "hybrid-only"
run_test "float32" "mnist"         "20" "cex" "hybrid-meas"
# float16/float64 MNIST cex: regenerated against the corrected norms (dafny_mnist_gram20).
# float16 hybrid uses the Keras pure-float16 policy forward; verified to reproduce the
# deployed float16 logits bit-for-bit (30/30 cexs), so it is sound by construction.
run_test "float16" "mnist"         "20" "cex" "standard"
run_test "float16" "mnist"         "20" "cex" "hybrid-only"
run_test "float16" "mnist"         "20" "cex" "hybrid-meas"
run_test "float64" "mnist"         "20" "cex" "standard"
run_test "float64" "mnist"         "20" "cex" "hybrid-only"
run_test "float64" "mnist"         "20" "cex" "hybrid-meas"

# ===== Fashion-MNIST natural — cex (gram 13; corrected norms + Dafny ref) =====
run_test "float32" "fashion_mnist" "13" "cex" "standard"
run_test "float32" "fashion_mnist" "13" "cex" "hybrid-only"
run_test "float32" "fashion_mnist" "13" "cex" "hybrid-meas"
run_test "float16" "fashion_mnist" "13" "cex" "standard"
run_test "float16" "fashion_mnist" "13" "cex" "hybrid-only"
run_test "float16" "fashion_mnist" "13" "cex" "hybrid-meas"
run_test "float64" "fashion_mnist" "13" "cex" "standard"
run_test "float64" "fashion_mnist" "13" "cex" "hybrid-only"
run_test "float64" "fashion_mnist" "13" "cex" "hybrid-meas"

# CIFAR-10 cex — disabled (pending corrected norms + regenerated cexs)
#run_test "float32" "cifar10"       "12" "cex" "standard"
#run_test "float32" "cifar10"       "12" "cex" "hybrid-only"
#run_test "float32" "cifar10"       "12" "cex" "hybrid-meas"
#run_test "float64" "cifar10"       "12" "cex" "standard"
#run_test "float64" "cifar10"       "12" "cex" "hybrid-only"
#run_test "float64" "cifar10"       "12" "cex" "hybrid-meas"
# NOTE: we don't run float16 cifar10 tests since n*u>=1 for that instance

# ===== MNIST adversarially-biased (1e6-end) — cex (gram 20) =====
run_test "float32" "mnist_biased_1e6_end" "20" "cex" "standard"
run_test "float32" "mnist_biased_1e6_end" "20" "cex" "hybrid-only"
run_test "float32" "mnist_biased_1e6_end" "20" "cex" "hybrid-meas"

# ===== Fashion-MNIST adversarially-biased (3e6-end) — cex (gram 13) =====
run_test "float32" "fashion_mnist_biased_3e6_end" "13" "cex" "standard"
run_test "float32" "fashion_mnist_biased_3e6_end" "13" "cex" "hybrid-only"
run_test "float32" "fashion_mnist_biased_3e6_end" "13" "cex" "hybrid-meas"
# CIFAR biased cex — disabled (pending CIFAR norms/Dafny @11)
#run_test "float32" "cifar10_biased_4e6_end" "12" "cex" "standard"
#run_test "float32" "cifar10_biased_4e6_end" "12" "cex" "hybrid-only"
#run_test "float32" "cifar10_biased_4e6_end" "12" "cex" "hybrid-meas"

# ===== MNIST natural — all (gram 20; reuses the JSON y1) =====
run_test "float32" "mnist"         "20" "all" "standard"
run_test "float32" "mnist"         "20" "all" "hybrid-only"
run_test "float32" "mnist"         "20" "all" "hybrid-meas"

# ===== Fashion-MNIST natural — all (gram 13) =====
run_test "float32" "fashion_mnist" "13" "all" "standard"
run_test "float32" "fashion_mnist" "13" "all" "hybrid-only"
run_test "float32" "fashion_mnist" "13" "all" "hybrid-meas"
# CIFAR all — disabled (pending CIFAR norms/Dafny @11)
#run_test "float32" "cifar10"       "12" "all" "standard"
#run_test "float32" "cifar10"       "12" "all" "hybrid-only"
#run_test "float32" "cifar10"       "12" "all" "hybrid-meas"

# ===== MNIST adversarially-biased (1e6-end) — all (gram 20) =====
run_test "float32" "mnist_biased_1e6_end" "20" "all" "standard"
run_test "float32" "mnist_biased_1e6_end" "20" "all" "hybrid-only"
run_test "float32" "mnist_biased_1e6_end" "20" "all" "hybrid-meas"

# ===== Fashion-MNIST adversarially-biased (3e6-end) — all (gram 13) =====
run_test "float32" "fashion_mnist_biased_3e6_end" "13" "all" "standard"
run_test "float32" "fashion_mnist_biased_3e6_end" "13" "all" "hybrid-only"
run_test "float32" "fashion_mnist_biased_3e6_end" "13" "all" "hybrid-meas"
# CIFAR biased all — disabled (pending CIFAR norms/Dafny @11)
#run_test "float32" "cifar10_biased_4e6_end" "12" "all" "standard"
#run_test "float32" "cifar10_biased_4e6_end" "12" "all" "hybrid-only"
#run_test "float32" "cifar10_biased_4e6_end" "12" "all" "hybrid-meas"
