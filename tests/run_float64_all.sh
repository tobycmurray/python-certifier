#!/usr/bin/env bash
# float64 regime of the ERAN comparison (2026-10-15): the six master-table models' full "all" sets certified with
# deployment format float64 (float32 weights executed in float64; inputs *_numpy64.json from
# models/provision_numpy64_all_inputs.sh), all three modes, gram 12, numpy execution. Exact invocation:
#   cd tests && source ../venv/bin/activate && FLOAT64_ALL=1 NUMPY_EXEC=1 SKIP_EXISTING=1 MODEL_FILTER=<m> run_tests.sh
# (SKIP_EXISTING skips the float32 tests whose outputs already exist.) Outputs:
#   json_results_numpy/{standard,hybrid_only,hybrid_meas}_<model>_float64_gram12_all.json
# VRA: float-conservatism/scripts/compute_vra.py json_results_numpy <cav2025-models> out.json float64
set -euo pipefail
cd "$(dirname "$0")"; source ../venv/bin/activate
while ! grep -q "All float64 numpy-execution input variants provisioned" ../models/provision_numpy64.log 2>/dev/null; do sleep 60; done
for m in mnist fashion_mnist cifar10 emnist_balanced_full emnist_byclass_full higgs_w1024_500k; do
  echo "##### $m ($(date))"
  FLOAT64_ALL=1 NUMPY_EXEC=1 SKIP_EXISTING=1 MODEL_FILTER=$m /opt/homebrew/bin/bash ./run_tests.sh
done
echo "##### float64 runs done ($(date))"
