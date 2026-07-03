#!/usr/bin/env bash
# Certify HIGGS-1024 over its FULL 500k canonical test set (Baldi last-500k tail),
# all three modes, gram 12 -- to replace the 50k-sample HIGGS row of the paper's
# master table with full-test-distribution numbers.
#
# This is a faithful superset of the committed 50k run. It mirrors, byte-for-byte
# in its invocations:
#   - inputs generation : models/regenerate_revision.sh  make_inputs (get_all_test_inputs.py)
#   - certification     : tests/run_tests.sh             run_test    (robust_certifier.py)
# The ONLY differences from the 50k RQ3 run are maxN 50000 -> 500000, the output
# directory, and the "_500k" json tags. No Dafny reference is supplied (identical
# to the committed HIGGS runs), so the certifier's own L_real is the real-arithmetic
# baseline. Idempotent: regeneration/certification are skipped if outputs exist.
#
# Uses python-certifier/venv (TF 2.13 + tfds + doitlib via PYTHONPATH). The full
# 11M-row HIGGS stream is read once to reach the canonical 500k test tail.
#
# Recommended (macOS, keep awake + capture a log):
#   caffeinate -i ./run_higgs_500k.sh 2>&1 | tee /tmp/higgs_500k.log
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # python-certifier/tests
ROOT="$(cd "$HERE/.." && pwd)"                            # python-certifier
VCRS="${VCRS:-$ROOT/../verified-certified-robustness/scripts}"
PY="${PY:-$ROOT/venv/bin/python}"

CSV="$VCRS/sweep_results/higgs_w1024_d5_full/model_weights_csv"
NET="$ROOT/models/neural_net_higgs_w1024_d5_full.txt"
INPUTS_GEN="$VCRS/get_all_test_inputs.py"
CERTIFIER="$ROOT/robust_certifier.py"

LAYERS="[1024,1024,1024,1024,1024]"
EPS="0.1"
OUTDIR="$HERE/inputs_higgs_w1024_n500000"
# HIGGS canonical Baldi split (last 500k test) -- identical to training/50k run.
H_TRAIN="10500000"; H_TEST="500000"

declare -A TAG=( [standard]=standard [hybrid-only]=hybrid_only [hybrid-meas]=hybrid_meas )

echo "=== HIGGS-1024 full-500k certification run ==="
echo "    PY=$PY"
echo "    NET=$NET"
echo "    CSV=$CSV"
echo "    OUTDIR=$OUTDIR"

# ---- 1) Generate the full-500k inputs (per-point npy + inputs.json), if absent ----
if [ -f "$OUTDIR/inputs.json" ]; then
  echo ">>> inputs already present, skipping generation: $OUTDIR"
else
  echo ">>> generating 500k inputs (this reads the full 11M HIGGS stream once) ..."
  mkdir -p "$OUTDIR"
  ( cd "$OUTDIR" && env PYTHONPATH="$VCRS" HIGGS_N_TRAIN="$H_TRAIN" HIGGS_N_TEST="$H_TEST" \
      "$PY" "$INPUTS_GEN" float32 higgs "$LAYERS" "$CSV" 1 inputs.json "$EPS" 500000 )
  echo ">>> inputs generated: $(ls -1 "$OUTDIR"/orig_higgs_x_*.npy | wc -l | tr -d ' ') npy files"
fi

# ---- 2) Certify all three modes (gram 12, no Dafny ref -> L_real baseline) ----
cd "$HERE"
mkdir -p json_results
for mode in standard hybrid-only hybrid-meas; do
  out="json_results/${TAG[$mode]}_higgs_w1024_500k_float32_gram12_all.json"
  if [ -f "$out" ]; then echo ">>> [$mode] output exists, skip: $out"; continue; fi
  echo ">>> [$mode] certifying 500k -> $out"
  "$PY" "$CERTIFIER" --mode "$mode" --json-output "$out" \
      float32 "$NET" 12 "$OUTDIR/inputs.json"
done

echo ">>> DONE (all three modes). VRA/analysis is computed separately in float-conservatism/scripts."
