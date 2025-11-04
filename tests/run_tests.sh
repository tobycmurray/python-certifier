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
FASHION_MNIST_RESULTS_GRAM_12="results_epsilon_0.26_[256,128,128,128,128,128,128,128,128,128,128,128]_500_eval_0.25_gram_12.json"
FASHION_MNIST_RESULTS_GRAM_13="results_epsilon_0.26_[256,128,128,128,128,128,128,128,128,128,128,128]_500_eval_0.25_gram_13.json"
CIFAR10_RESULTS_GRAM_12="results_epsilon_0.1551_[512,256,128,128,128,128,128,128]_800_eval_0.141_gram_12.json"
Z3_RESULTS_GRAM_10="z3_certifier_results_gram_10.json"

MNIST_NEURAL_NET="neural_net_mnist_epsilon_0.45_[128,128,128,128,128,128,128,128]_500.txt"
FASHION_MNIST_NEURAL_NET="neural_net_mnist_epsilon_0.26_[256,128,128,128,128,128,128,128,128,128,128,128]_500.txt"
CIFAR10_NEURAL_NET="neural_net_mnist_epsilon_0.1551_[512,256,128,128,128,128,128,128]_800.txt"
Z3_NEURAL_NET="z3_neural_network.txt"

ALL_MNIST_TEST_INPUTS="all_mnist_test_inputs/test_inputs_epsilon_0.3.json"
ALL_FASHION_MNIST_TEST_INPUTS="all_fashion_mnist_test_inputs/test_inputs_epsilon_0.25.json"
ALL_CIFAR10_TEST_INPUTS="all_cifar10_test_inputs/all_test_inputs.json"

CEX_MNIST_FLOAT32="cex_mnist_deepfool/counter_examples.json"
CEX_MNIST_FLOAT16="cex_mnist_deepfool_float16/counter_examples.json"
CEX_MNIST_FLOAT64="cex_mnist_deepfool_float64/counter_examples.json"
CEX_FASHION_MNIST_FLOAT32="cex_fashion_mnist_deepfool/counter_examples.json"
CEX_CIFAR10_FLOAT32="cex_cifar10_deepfool/counter_examples.json"
CEX_Z3_FLOAT32="z3_counter_examples.json"

# --- declarative tables ----------------------------------------------------

declare -A NN_FILE=(
  [mnist]="$MNIST_NEURAL_NET"
  [fashion_mnist]="$FASHION_MNIST_NEURAL_NET"
  [cifar10]="$CIFAR10_NEURAL_NET"
  [z3]="$Z3_NEURAL_NET"
)

declare -A REF_RESULTS=(
  ["mnist:11"]="$MNIST_RESULTS_GRAM_11"
  ["mnist:20"]="$MNIST_RESULTS_GRAM_20"
  ["fashion_mnist:12"]="$FASHION_MNIST_RESULTS_GRAM_12"
  ["fashion_mnist:13"]="$FASHION_MNIST_RESULTS_GRAM_13"
  ["cifar10:12"]="$CIFAR10_RESULTS_GRAM_12"
  ["z3:10"]="$Z3_RESULTS_GRAM_10"
)

declare -A ALL_INPUTS=(
  [mnist]="$ALL_MNIST_TEST_INPUTS"
  [fashion_mnist]="$ALL_FASHION_MNIST_TEST_INPUTS"
  [cifar10]="$ALL_CIFAR10_TEST_INPUTS"
)

declare -A CEX=(
  ["mnist:float32"]="$CEX_MNIST_FLOAT32"
  ["mnist:float16"]="$CEX_MNIST_FLOAT16"
  ["mnist:float64"]="$CEX_MNIST_FLOAT64"
  ["fashion_mnist:float32"]="$CEX_FASHION_MNIST_FLOAT32"
  ["cifar10:float32"]="$CEX_CIFAR10_FLOAT32"
  ["z3:float32"]="$CEX_Z3_FLOAT32"
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
  local format="$1" model="$2" gram="$3" kind="$4"

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

  echo -n "Running test: $format, $model, $gram, $kind ...  "
  python "$CERTIFIER" "$format" "$nn_file" "$gram" --cex "$cex_file" "$ref_results_file" > .log 2>&1 || (cat .log; die "Couldn't run python certifier")

  local count count_ok count_failed count_ok_real
  count=$(       grabnum '^Got [0-9]+ instances to certify'                                )
  count_ok=$(    grabnum '^Certified [0-9]+ instances as robust'                           )
  count_failed=$(grabnum '^Failed to certify [0-9]+ instances as robust'                   )
  count_ok_real=$(grabnum 'Real certifier would have certified [0-9]+ instances as robust' )

  (( count_ok + count_failed == count )) || die "Internal error: counts don't add up (ok=$count_ok failed=$count_failed total=$count)"

  if [[ "$kind" == "cex" ]]; then
    (( count_ok == 0 )) || die "Certifier certified $count_ok counter-examples!"
    (( count_ok_real == count )) || die "Real-arithmetic certifier would not have certified all counter-examples!"
  else
    # kind is "all"
    local ref_num
    ref_num=$(grep -c true "$ref_results_file")
    (( count_ok_real == ref_num )) || die "Mismatch vs Dafny reference: real=$count_ok_real ref=$ref_num ($ref_results_file)"
  fi

  echo "OK"
}

# --- test matrix -----------------------------------------------------------

run_test "float32" "z3"            "10" "cex"

run_test "float32" "mnist"         "20" "cex"
run_test "float16" "mnist"         "20" "cex"
run_test "float64" "mnist"         "20" "cex"

run_test "float32" "fashion_mnist" "13" "cex"

run_test "float32" "mnist"         "11" "all"
run_test "float32" "mnist"         "20" "all"
run_test "float32" "fashion_mnist" "12" "all"
run_test "float32" "fashion_mnist" "13" "all"
