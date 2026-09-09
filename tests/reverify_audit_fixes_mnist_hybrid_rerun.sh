#!/usr/bin/env bash
# Re-run ONLY the MNIST hybrid-only / hybrid-meas "all" runs (NUMPY_EXEC=1
# semantics) after adding the centre-execution finiteness check
# (center_exec_finite / "Refused N instances: execution at x not observed
# overflow-free"). Same invocation as reverify_audit_fixes.sh group mnist.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-../venv/bin/python}"
NN="../models/neural_net_mnist_epsilon_0.45_[128,128,128,128,128,128,128,128]_500.txt"
REF="../models/precomputed/dafny_mnist_gram12.json"
INP="all_mnist_test_inputs/test_inputs_epsilon_0.3_numpy.json"
for mode in hybrid-only hybrid-meas; do
  tag="${mode//-/_}"; name="${tag}_mnist_float32_gram12_all"
  echo "[$(date +%H:%M:%S)] running $name (finiteness-check rerun)"
  "$PY" ../robust_certifier.py --mode "$mode" --numpy-exec --json-output "json_results_numpy_reverify/$name.json" \
      float32 "$NN" 12 "$INP" "$REF" > "reverify_audit_fixes_out/$name.finiteness_rerun.log" 2>&1
  grep -E "Got |Certified|Failed|Refused|would have" "reverify_audit_fixes_out/$name.finiteness_rerun.log"
done
echo "[$(date +%H:%M:%S)] mnist hybrid rerun done"
