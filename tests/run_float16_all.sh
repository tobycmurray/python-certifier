#!/usr/bin/env bash
# SUPERSEDED (2026-11-05) by `bash run_job.sh float16-all` (same commands; checked by DRY_RUN diff). Kept because
# its log tests/log.run_float16_all.txt references it; do not extend. NB MODEL_FILTER=$m below is ignored by
# run_tests.sh (it reads $1 only), so every iteration ran the whole matrix under SKIP_EXISTING=1.
# float16 regime (2026-11-04): the full "all" sets of the master-table models whose layers all have
# n*u < 1 (every model but CIFAR-10, whose 3072-wide input layer has n*u = 1.5 at float16; by policy
# the certifier refuses such cases rather than report vacuous numbers) certified with deployment
# format float16 (float32 weights cast down to float16; inputs *_numpy16.json from
# models/provision_numpy16_all_inputs.sh), all three modes, gram 12, numpy execution. Evidence for the
# paper's §9.1 claim that at float16 overflow certification succeeds but robustness certification is
# vacuous in every mode on every test set. Mirrors run_float64_all.sh. Exact invocation:
#   cd tests && nohup bash run_float16_all.sh > log.run_float16_all.txt 2>&1 &
# Outputs: json_results_numpy/{standard,hybrid_only,hybrid_meas}_<model>_float16_gram12_all.json
set -euo pipefail
cd "$(dirname "$0")"; source ../venv/bin/activate
while ! grep -q "All float16 numpy-execution input variants provisioned" ../models/provision_numpy16_all_inputs.log 2>/dev/null; do sleep 60; done
for m in mnist fashion_mnist emnist_balanced_full emnist_byclass_full higgs_w1024_500k; do
  echo "##### $m ($(date))"
  FLOAT16_ALL=1 NUMPY_EXEC=1 SKIP_EXISTING=1 MODEL_FILTER=$m /opt/homebrew/bin/bash ./run_tests.sh
done
echo "##### float16 runs done ($(date))"
