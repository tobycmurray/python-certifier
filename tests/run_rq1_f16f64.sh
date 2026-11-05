#!/usr/bin/env bash
# SUPERSEDED (2026-11-05) by `bash run_job.sh rq1-f16f64` (same steps and commands). Kept because its log
# tests/log.run_rq1_f16f64.txt references it; do not extend.
# RQ1 float16/float64 instances (2026-11-04/05): verify, install and certify the regenerated
# natural float16/float64 continuous-search suites (MNIST and Fashion MNIST at float16; MNIST,
# Fashion MNIST and CIFAR-10 at float64), produced by verified-certified-robustness
# scripts/regen_float16_float64_suites.sh with the attack script loading weights at the float32
# policy and casting to the target format (the July 2026 suites parsed the CSV decimals straight to
# the target format, so they fail the certifier's logits-reproduction check at float64 and the
# paper's verification standard). Exact invocation:
#   cd tests && bash run_rq1_f16f64.sh 2>&1 | tee log.run_rq1_f16f64.txt
# Steps: (1) reverify_cex_suite.py on each new suite at its format (exact rational checks on the
# points cast to the format; must verify 30/30); (2) move the July suites to stale_pre_f16f64_regen/
# and copy the new ones into tests/cex_<model>_<format>_numpy; (3) per model,
#   NUMPY_EXEC=1 SKIP_EXISTING=1 run_tests.sh <model>
# which re-runs exactly the outputs older than their (new) suites: the five float16/float64 cex
# tests in the three modes. Checks are run_tests.sh's own: FP certifies 0, real certifies all;
# this log must also contain no "REFUSED" (a refusal is not a rejection).
set -euo pipefail
cd "$(dirname "$0")"; source ../venv/bin/activate   # the certifier's own venv (gmpy2 etc.)
VCR=~/git/verified-certified-robustness/scripts; PC=$(cd .. && pwd); REFS=$PC/models/precomputed
CAV="$VCR/../cav2025-models"; PY=$VCR/cav2025-artifact-venv/bin/python
NEW=$VCR/results/natural_grid_cex/f16f64_suites_regen
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
MNIST_CSV="$CAV/2025-01-25_09:27:46-mnist/model_weights_epsilon_0.45_[128,128,128,128,128,128,128,128]_500"
FASHION_CSV="$CAV/2025-01-30_10:58:01-fashion_mnist/model_weights_epsilon_0.26_[256,128,128,128,128,128,128,128,128,128,128,128]_500"
CIFAR_CSV="$CAV/2025-01-28_20:39:32-cifar10/model_weights_epsilon_0.1551_[512,256,128,128,128,128,128,128]_800"
MNIST_L="[128,128,128,128,128,128,128,128]"
FASHION_L="[256,128,128,128,128,128,128,128,128,128,128,128]"
CIFAR_L="[512,256,128,128,128,128,128,128]"
reverify () {  # fmt dataset layers csv isize lipref suite
  echo "##### reverify $(basename "$7") ($(date))"
  ( cd "$VCR" && env PYTHONPATH="$VCR" PYTHON_CERTIFIER="$PC" "$PY" reverify_cex_suite.py "$1" "$2" "$3" "$4" "$5" "$6" --suite "$7" ) | grep -v 'DeepFool\|^$' | tail -n 12
  python3 -I -c "
import json,sys; s=json.load(open('$VCR/results/natural_grid_cex/reverify_$(basename "$7").json'))['summary']
print('SUMMARY', json.dumps(s)); sys.exit(0 if s['n_verified_float32']==s['n_pairs']==30 else 1)"
}
reverify float16 mnist         "$MNIST_L"   "$MNIST_CSV"   28 "$REFS/dafny_mnist_gram20.json"   "$NEW/cex_mnist_float16_numpy"
reverify float16 fashion_mnist "$FASHION_L" "$FASHION_CSV" 28 "$REFS/dafny_fashion_gram13.json" "$NEW/cex_fashion_mnist_float16_numpy"
reverify float64 mnist         "$MNIST_L"   "$MNIST_CSV"   28 "$REFS/dafny_mnist_gram20.json"   "$NEW/cex_mnist_float64_numpy"
reverify float64 fashion_mnist "$FASHION_L" "$FASHION_CSV" 28 "$REFS/dafny_fashion_gram13.json" "$NEW/cex_fashion_mnist_float64_numpy"
reverify float64 cifar10       "$CIFAR_L"   "$CIFAR_CSV"   32 "$REFS/dafny_cifar10_gram12.json" "$NEW/cex_cifar10_float64_numpy"
echo "##### all five suites verify 30/30; installing ($(date))"
mkdir -p stale_pre_f16f64_regen
for d in cex_mnist_float16_numpy cex_fashion_mnist_float16_numpy cex_mnist_float64_numpy cex_fashion_mnist_float64_numpy cex_cifar10_float64_numpy; do
  if [ ! -d "stale_pre_f16f64_regen/$d" ]; then   # idempotent: a rerun must not move the new suites aside
    [ -d "$d" ] && mv "$d" "stale_pre_f16f64_regen/$d"
    cp -R "$NEW/$d" "$d"
  fi
done
for m in mnist fashion_mnist cifar10; do
  echo "##### certify $m ($(date))"
  NUMPY_EXEC=1 SKIP_EXISTING=1 /opt/homebrew/bin/bash ./run_tests.sh "$m"
done
echo "##### RQ1 float16/float64 runs done ($(date))"
