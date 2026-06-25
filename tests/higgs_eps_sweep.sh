#!/usr/bin/env bash
# Epsilon sweep for HIGGS-1024 (fixed gloro model, trained @ eval-eps 0.1), all 3
# modes, characterizing how the FP-soundness cost varies with the certified radius
# eps. HIGGS has NO canonical robustness eps (it is a standard ML/scalability
# benchmark but novel in the certified-robustness setting), and our perturbation is
# in standardized-sigma L2 units -- so this characterizes the certifier across eps
# rather than committing to a single value (0.1 remains the headline default).
#
# Cheap by construction: the 10k inputs (x .npy + y1 logits) are eps-INDEPENDENT, so
# we reuse inputs_higgs_w1024_n10000/ and only rewrite each record's max_eps per eps
# -- no 11M-row HIGGS reload, no .npy regeneration. Norms (gram 12, fp64) are cached.
#
# Run from python-certifier/tests/ with the certifier venv on PATH:
#   PATH=../venv/bin:$PATH ./higgs_eps_sweep.sh
# Outputs raw per-(eps,mode) results to json_results_eps_sweep/ (gitignored); tabulate
# with scripts/higgs_eps_sweep_report.py in the float-conservatism repo.
set -euo pipefail

CERT=../robust_certifier.py
NET=../models/neural_net_higgs_w1024_d5_full.txt
DIR=inputs_higgs_w1024_n10000
SRC="$DIR/inputs.json"
OUT=json_results_eps_sweep
GRAM=12
EPS_LIST="0.02 0.05 0.1 0.15 0.2 0.3"

[ -f "$NET" ] || { echo "missing $NET (run models/regenerate_revision.sh)"; exit 1; }
[ -f "$SRC" ] || { echo "missing $SRC (run models/regenerate_revision.sh)"; exit 1; }
mkdir -p "$OUT"

for eps in $EPS_LIST; do
  # eps-variant inputs: same .npy/y1, only max_eps rewritten (resolves .npy via $DIR)
  J="$DIR/inputs_eps${eps}.json"
  python3 -c "import json; d=json.load(open('$SRC')); [r.__setitem__('max_eps','$eps') for r in d]; json.dump(d, open('$J','w'))"
  for mode in standard hybrid-only hybrid-meas; do
    o="$OUT/higgs_w1024_eps${eps}_${mode}.json"
    if [ -s "$o" ]; then echo "skip (exists): eps=$eps $mode"; continue; fi
    echo ">>> $(date '+%H:%M:%S') eps=$eps mode=$mode"
    python "$CERT" --mode "$mode" --norm-method fp64 --json-output "$o" \
        float32 "$NET" "$GRAM" "$J" > /tmp/higgs_eps_sweep.log 2>&1 \
      || { echo "FAILED eps=$eps $mode"; tail -8 /tmp/higgs_eps_sweep.log; exit 1; }
    grep -E "Certified [0-9]+ instances|(Real|Dafny) certifier would have certified [0-9]+" /tmp/higgs_eps_sweep.log
  done
done
echo "HIGGS EPS SWEEP DONE"
