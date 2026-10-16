#!/usr/bin/env bash
# RQ1 explicit re-run (float-conservatism OUTSTANDING-WORK item 6, 2026-10-17): certify, in all three
# modes under the compliant numpy float32 execution, (a) the regenerated 30/30/30 continuous-search
# instances on the natural models (tests/cex_{mnist,fashion_mnist,cifar10}_float32_numpy, replaced in
# e69a565f; their json_results_numpy outputs still dated from the PREVIOUS suites, which SKIP_EXISTING
# used to skip -- it now re-runs any output older than its input set) and (b) the 24/204/370 grid
# instances on the biased models (kind natgrid, built by make_natgrid_inputs.py). Exact invocation:
#   cd tests && source ../venv/bin/activate && bash run_rq1_instances.sh
# which runs, per model,
#   NUMPY_EXEC=1 SKIP_EXISTING=1 run_tests.sh <model>
# so only tests without an up-to-date output run (first time: the nine (a) and nine (b)). Checks are
# run_tests.sh's own: FP certifies 0, real certifies all. Outputs:
#   json_results_numpy/{standard,hybrid_only,hybrid_meas}_<model>_float32_gram<g>_cex.json      (a)
#   json_results_numpy/{standard,hybrid_only,hybrid_meas}_<model>_biased_*_float32_gram<g>_natgrid.json  (b)
# Log: tests/log.run_rq1_instances.txt
set -euo pipefail
cd "$(dirname "$0")"; source ../venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python make_natgrid_inputs.py
for m in mnist fashion_mnist cifar10; do
  echo "##### $m ($(date))"
  NUMPY_EXEC=1 SKIP_EXISTING=1 /opt/homebrew/bin/bash ./run_tests.sh "$m"
done
echo "##### RQ1 runs done ($(date))"
