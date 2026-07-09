#!/usr/bin/env bash
# Witness-conformance (no-FTZ) evidence run, mirroring run_tests.sh's model tables.
#
# For every model we certify, verify that no flush-to-zero occurs during the
# deployed Keras forward pass on its witness inputs -- so each measured witness
# execution conforms to the standard (gradual-underflow) FP model and the
# certificate is sound for it (see ../check_no_ftz.py for the argument).
#
# The check is a property of the forward pass at each input x, independent of
# epsilon / gram / mode, so ONE run per (model, input-set) suffices. Each
# float32 run also checks the float64 reference execution internally.
#
# Covers, for each model: the "all" test set (the reported-VRA witness points)
# and the float32 counterexample set. Ordered small -> large so evidence lands
# early. Set FTZ_LIMIT=N to cap inputs per set (smoke test); empty = full.
#
# Usage:  /opt/homebrew/bin/bash check_no_ftz.sh [MODEL_FILTER]
# Output: check_no_ftz.log  (grep -E 'PASS|FAIL|min .w\*z.' for the summary)
set -uo pipefail

CHECK=../check_no_ftz.py
PY=${PY:-../venv/bin/python}
LIMIT_FLAG=""; [[ -n "${FTZ_LIMIT:-}" ]] && LIMIT_FLAG="--limit ${FTZ_LIMIT}"
FILTER="${1:-}"
LOG=check_no_ftz.log; : > "$LOG"

# --- networks / inputs / biases (copied verbatim from run_tests.sh) ---------
MNIST_NET="../models/neural_net_mnist_epsilon_0.45_[128,128,128,128,128,128,128,128]_500.txt"
FASHION_NET="../models/neural_net_fashion_mnist_epsilon_0.26_[256,128,128,128,128,128,128,128,128,128,128,128]_500.txt"
CIFAR_NET="../models/neural_net_cifar10_epsilon_0.1551_[512,256,128,128,128,128,128,128]_800.txt"
HIGGS_W128_NET="../models/neural_net_higgs_w128_d5_full.txt"
HIGGS_W256_NET="../models/neural_net_higgs_w256_d5_full.txt"
HIGGS_W512_NET="../models/neural_net_higgs_w512_d5_full.txt"
HIGGS_W1024_NET="../models/neural_net_higgs_w1024_d5_full.txt"
EMNIST_BYCLASS_NET="../models/neural_net_emnistbyc_cifar.txt"
EMNIST_BALANCED_NET="../models/neural_net_emnistbal_w512_d8_ep500.txt"

ALL_MNIST="all_mnist_test_inputs/test_inputs_epsilon_0.3.json"
ALL_FASHION="all_fashion_mnist_test_inputs/test_inputs_epsilon_0.25.json"
ALL_CIFAR="all_cifar10_test_inputs/all_test_inputs.json"

# model  ->  "NET | ALL_INPUTS | FLOAT32_CEX | BIASES"   ('-' = none)
declare -A SPEC=(
  [mnist]="$MNIST_NET | $ALL_MNIST | cex_mnist_float32/counter_examples.json | -"
  [fashion_mnist]="$FASHION_NET | $ALL_FASHION | cex_fashion_mnist_float32/counter_examples.json | -"
  [cifar10]="$CIFAR_NET | $ALL_CIFAR | cex_cifar10_float32/counter_examples.json | -"
  [mnist_biased_1e6_end]="$MNIST_NET | $ALL_MNIST | cex_mnist_float32_biased_1e6_end/counter_examples.json | cex_mnist_float32_biased_1e6_end/biases.txt"
  [fashion_mnist_biased_3e6_end]="$FASHION_NET | $ALL_FASHION | cex_fashion_mnist_float32_biased_3e6_end/counter_examples.json | cex_fashion_mnist_float32_biased_3e6_end/biases.txt"
  [cifar10_biased_4e6_end]="$CIFAR_NET | $ALL_CIFAR | cex_cifar10_float32_biased_4e6_end/counter_examples.json | cex_cifar10_float32_biased_4e6_end/biases.txt"
  [higgs_w128]="$HIGGS_W128_NET | inputs_higgs_w128_n10000/inputs.json | - | -"
  [higgs_w256]="$HIGGS_W256_NET | inputs_higgs_w256_n10000/inputs.json | - | -"
  [higgs_w512]="$HIGGS_W512_NET | inputs_higgs_w512_n10000/inputs.json | - | -"
  [higgs_w1024]="$HIGGS_W1024_NET | inputs_higgs_w1024_n10000/inputs.json | - | -"
  [emnist_balanced_full]="$EMNIST_BALANCED_NET | inputs_emnist_balanced_full/inputs.json | - | -"
  [emnist_byclass_full]="$EMNIST_BYCLASS_NET | inputs_emnist_byclass_full/inputs.json | - | -"
  [higgs_w1024_500k]="$HIGGS_W1024_NET | inputs_higgs_w1024_n500000/inputs.json | - | -"
)

# Small -> large (cex sets are tiny; big test sets last).
ORDER=(mnist fashion_mnist cifar10 mnist_biased_1e6_end fashion_mnist_biased_3e6_end
       cifar10_biased_4e6_end higgs_w128 higgs_w256 higgs_w512 higgs_w1024
       emnist_balanced_full emnist_byclass_full higgs_w1024_500k)

one() {  # model  label  input_json  biases
  local model="$1" label="$2" inp="$3" biases="$4"
  [[ "$inp" == "-" || ! -f "$inp" ]] && { echo ">> SKIP $model/$label (no $inp)" | tee -a "$LOG"; return 0; }
  local bf=""; [[ "$biases" != "-" ]] && bf="--biases $biases"
  echo "======== $model  [$label]  $inp ========" | tee -a "$LOG"
  # shellcheck disable=SC2086
  "$PY" "$CHECK" float32 "$net" "$inp" $bf $LIMIT_FLAG 2>&1 \
    | grep -vE "Warning|warn|urllib3|NotOpenSSL" | tee -a "$LOG"
}

for model in "${ORDER[@]}"; do
  [[ -n "$FILTER" && "$model" != "$FILTER"* ]] && continue
  IFS='|' read -r net all cex biases <<< "${SPEC[$model]}"
  net="$(echo "$net" | xargs)"; all="$(echo "$all" | xargs)"
  cex="$(echo "$cex" | xargs)"; biases="$(echo "$biases" | xargs)"
  one "$model" "cex"  "$cex" "$biases"
  one "$model" "all"  "$all" "$biases"
done

echo; echo "==================== SUMMARY ===================="  | tee -a "$LOG"
grep -E "========|min \|w\*z\|| PASS| FAIL|ALL PRECISIONS|CHECK FAILED|SKIP" "$LOG" | tee /dev/tty >/dev/null
if grep -qE "FAIL|CHECK FAILED" "$LOG"; then echo "RESULT: at least one FAIL -- inspect $LOG"; exit 1
else echo "RESULT: no FTZ on any witness point checked."; fi
