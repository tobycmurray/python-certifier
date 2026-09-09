#!/usr/bin/env bash
# Re-verification of the paper's reported numbers after the audit fixes
# (overflow refusal, final-layer bias term, bias-aware D^hi, exact-rational
# measured norms, hybrid-mode logits consistency check).
#
# Reproduces the exact run_tests.sh invocations (function run_test; NN_FILE /
# ALL_INPUTS / CEX / REF_RESULTS / BIASES_FILE tables) under NUMPY_EXEC=1
# semantics -- i.e. the *_numpy input variants and, for the hybrid modes,
# --numpy-exec -- for the three image models (natural + adversarially biased),
# and compares the per-instance verdicts against the baseline outputs in
# tests/json_results_numpy/ (produced at commit b01fedad, before the fixes).
# HIGGS and EMNIST are NOT re-run here (long); they remain to be re-run.
#
# Usage (from tests/):
#   ./reverify_audit_fixes.sh run [mnist|fashion_mnist|cifar10|all]   # run one model group (default all)
#   ./reverify_audit_fixes.sh compare                                  # compare all new outputs vs baseline
# The three groups are independent and may be run concurrently (one process
# each); outputs go to json_results_numpy_reverify/, logs and the count
# summary to reverify_audit_fixes_out/.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-../venv/bin/python}"
CERTIFIER=../robust_certifier.py
OUT_DIR=json_results_numpy_reverify
LOG_DIR=reverify_audit_fixes_out
BASELINE_DIR=json_results_numpy
mkdir -p "$OUT_DIR" "$LOG_DIR"

# --- tables (verbatim from run_tests.sh) -----------------------------------
MNIST_NEURAL_NET="../models/neural_net_mnist_epsilon_0.45_[128,128,128,128,128,128,128,128]_500.txt"
FASHION_MNIST_NEURAL_NET="../models/neural_net_fashion_mnist_epsilon_0.26_[256,128,128,128,128,128,128,128,128,128,128,128]_500.txt"
CIFAR10_NEURAL_NET="../models/neural_net_cifar10_epsilon_0.1551_[512,256,128,128,128,128,128,128]_800.txt"

MNIST_DAFNY_REF="../models/precomputed/dafny_mnist_gram20.json"
FASHION_DAFNY_REF="../models/precomputed/dafny_fashion_gram13.json"
MNIST_DAFNY_REF_12="../models/precomputed/dafny_mnist_gram12.json"
FASHION_DAFNY_REF_12="../models/precomputed/dafny_fashion_mnist_gram12.json"
CIFAR_DAFNY_REF_12="../models/precomputed/dafny_cifar10_gram12.json"

# NUMPY_EXEC=1 remapping of run_tests.sh's ALL_INPUTS (numpy_variant()):
ALL_MNIST="all_mnist_test_inputs/test_inputs_epsilon_0.3_numpy.json"
ALL_MNIST_BIASED="all_mnist_test_inputs/test_inputs_epsilon_0.3_numpy_biased_1e6_end.json"
ALL_FASHION="all_fashion_mnist_test_inputs/test_inputs_epsilon_0.25_numpy.json"
ALL_FASHION_BIASED="all_fashion_mnist_test_inputs/test_inputs_epsilon_0.25_numpy_biased_3e6_end.json"
ALL_CIFAR="all_cifar10_test_inputs/all_test_inputs_numpy.json"
ALL_CIFAR_BIASED="all_cifar10_test_inputs/all_test_inputs_numpy_biased_4e6_end.json"

MNIST_BIASES="cex_mnist_float32_biased_1e6_end/biases.txt"
FASHION_BIASES="cex_fashion_mnist_float32_biased_3e6_end/biases.txt"
CIFAR_BIASES="cex_cifar10_float32_biased_4e6_end/biases.txt"

MODES=(standard hybrid-only hybrid-meas)

# Stage the committed norm caches exactly as run_tests.sh does.
cp -f ../models/precomputed/*.norms.json . 2>/dev/null || true

# run_one FORMAT MODEL GRAM KIND MODE NN CEX_FILE REF BIASES
run_one() {
  local format="$1" model="$2" gram="$3" kind="$4" mode="$5" nn="$6" cex="$7" ref="$8" biases="$9"
  local mode_tag; mode_tag="${mode//-/_}"
  local name="${mode_tag}_${model}_${format}_gram${gram}_${kind}"
  local json_output="$OUT_DIR/${name}.json"
  local log="$LOG_DIR/${name}.log"
  local bias_flag=""; [[ -n "$biases" ]] && bias_flag="--biases $biases"
  local numpy_flag=""; [[ "$mode" == hybrid-* ]] && numpy_flag="--numpy-exec"
  echo "[$(date +%H:%M:%S)] running $name"
  # shellcheck disable=SC2086
  "$PY" "$CERTIFIER" --mode "$mode" --json-output "$json_output" $bias_flag $numpy_flag \
      "$format" "$nn" "$gram" "$cex" ${ref:+"$ref"} > "$log" 2>&1 \
      || { tail -n 30 "$log"; echo "ERROR: certifier failed for $name" >&2; exit 1; }
  local got ok fail ref_ok ovf lg
  got=$(grep -Eo 'Got [0-9]+ instances' "$log" | grep -Eo '[0-9]+')
  ok=$(grep -Eo 'Certified [0-9]+ instances as robust' "$log" | grep -Eo '[0-9]+')
  fail=$(grep -Eo 'Failed to certify [0-9]+' "$log" | grep -Eo '[0-9]+')
  ref_ok=$(grep -Eo '(Real|Dafny) certifier would have certified [0-9]+' "$log" | grep -Eo '[0-9]+$')
  ovf=$(grep -Eo 'Refused [0-9]+ instances: overflow' "$log" | grep -Eo '[0-9]+')
  lg=$(grep -Eo 'Refused [0-9]+ instances: measuring' "$log" | grep -Eo '[0-9]+' || echo "-")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$got" "$ok" "$fail" "$ref_ok" "$ovf" "$lg" >> "$LOG_DIR/counts_${GROUP}.tsv"
}

group_mnist() {
  local m
  for m in "${MODES[@]}"; do
    # cex (gram 20)
    run_one float32 mnist 20 cex "$m" "$MNIST_NEURAL_NET" cex_mnist_float32_numpy/counter_examples.json "$MNIST_DAFNY_REF" ""
    run_one float16 mnist 20 cex "$m" "$MNIST_NEURAL_NET" cex_mnist_float16_numpy/counter_examples.json "$MNIST_DAFNY_REF" ""
    run_one float64 mnist 20 cex "$m" "$MNIST_NEURAL_NET" cex_mnist_float64_numpy/counter_examples.json "$MNIST_DAFNY_REF" ""
    run_one float32 mnist_biased_1e6_end 20 cex "$m" "$MNIST_NEURAL_NET" cex_mnist_float32_biased_1e6_end_numpy/counter_examples.json "$MNIST_DAFNY_REF" "$MNIST_BIASES"
  done
  for m in "${MODES[@]}"; do
    # all (gram 12)
    run_one float32 mnist 12 all "$m" "$MNIST_NEURAL_NET" "$ALL_MNIST" "$MNIST_DAFNY_REF_12" ""
    run_one float32 mnist_biased_1e6_end 12 all "$m" "$MNIST_NEURAL_NET" "$ALL_MNIST_BIASED" "$MNIST_DAFNY_REF_12" "$MNIST_BIASES"
  done
}

group_fashion_mnist() {
  local m
  for m in "${MODES[@]}"; do
    run_one float32 fashion_mnist 13 cex "$m" "$FASHION_MNIST_NEURAL_NET" cex_fashion_mnist_float32_numpy/counter_examples.json "$FASHION_DAFNY_REF" ""
    run_one float16 fashion_mnist 13 cex "$m" "$FASHION_MNIST_NEURAL_NET" cex_fashion_mnist_float16_numpy/counter_examples.json "$FASHION_DAFNY_REF" ""
    run_one float64 fashion_mnist 13 cex "$m" "$FASHION_MNIST_NEURAL_NET" cex_fashion_mnist_float64_numpy/counter_examples.json "$FASHION_DAFNY_REF" ""
    run_one float32 fashion_mnist_biased_3e6_end 13 cex "$m" "$FASHION_MNIST_NEURAL_NET" cex_fashion_mnist_float32_biased_3e6_end_numpy/counter_examples.json "$FASHION_DAFNY_REF" "$FASHION_BIASES"
  done
  for m in "${MODES[@]}"; do
    run_one float32 fashion_mnist 12 all "$m" "$FASHION_MNIST_NEURAL_NET" "$ALL_FASHION" "$FASHION_DAFNY_REF_12" ""
    run_one float32 fashion_mnist_biased_3e6_end 12 all "$m" "$FASHION_MNIST_NEURAL_NET" "$ALL_FASHION_BIASED" "$FASHION_DAFNY_REF_12" "$FASHION_BIASES"
  done
}

group_cifar10() {
  local m
  for m in "${MODES[@]}"; do
    run_one float32 cifar10 12 cex "$m" "$CIFAR10_NEURAL_NET" cex_cifar10_float32_numpy/counter_examples.json "$CIFAR_DAFNY_REF_12" ""
    run_one float64 cifar10 12 cex "$m" "$CIFAR10_NEURAL_NET" cex_cifar10_float64_numpy/counter_examples.json "$CIFAR_DAFNY_REF_12" ""
    run_one float32 cifar10_biased_4e6_end 12 cex "$m" "$CIFAR10_NEURAL_NET" cex_cifar10_float32_biased_4e6_end_numpy/counter_examples.json "$CIFAR_DAFNY_REF_12" "$CIFAR_BIASES"
  done
  for m in "${MODES[@]}"; do
    run_one float32 cifar10 12 all "$m" "$CIFAR10_NEURAL_NET" "$ALL_CIFAR" "$CIFAR_DAFNY_REF_12" ""
    run_one float32 cifar10_biased_4e6_end 12 all "$m" "$CIFAR10_NEURAL_NET" "$ALL_CIFAR_BIASED" "$CIFAR_DAFNY_REF_12" "$CIFAR_BIASES"
  done
}

cmd="${1:-run}"
case "$cmd" in
  run)
    GROUP="${2:-all}"
    : > "$LOG_DIR/counts_${GROUP}.tsv"
    case "$GROUP" in
      mnist)         group_mnist ;;
      fashion_mnist) group_fashion_mnist ;;
      cifar10)       group_cifar10 ;;
      all)           group_mnist; group_fashion_mnist; group_cifar10 ;;
      *) echo "unknown group $GROUP" >&2; exit 1 ;;
    esac
    echo "[$(date +%H:%M:%S)] group $GROUP done; counts in $LOG_DIR/counts_${GROUP}.tsv"
    ;;
  compare)
    "$PY" reverify_audit_fixes_compare.py "$BASELINE_DIR" "$OUT_DIR" "$LOG_DIR" | tee "$LOG_DIR/summary.txt"
    ;;
  *) echo "usage: $0 run [group] | compare" >&2; exit 1 ;;
esac
