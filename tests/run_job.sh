#!/usr/bin/env bash
# The one job wrapper around run_tests.sh (artifact DESIGN §7.1, 2026-11-05). Usage, from tests/:
#   bash run_job.sh <job>          jobs: rq1 | rq1-f16f64 | float16-all | float64-all | all-float32 | higgs-500k | recompute-norms
#   DRY_RUN=1 bash run_job.sh <job>   prints every certifier command line the job would run (run_tests.sh's
#                                     DRY_RUN) and skips the job's non-certifier steps, printing them to stderr.
# The first four jobs reproduce the superseded per-job wrappers EXACTLY (same environment, same model
# filters, same preparatory steps; checked by diffing DRY_RUN outputs):
#   rq1          = run_rq1_instances.sh : make_natgrid_inputs.py, then NUMPY_EXEC=1 SKIP_EXISTING=1
#                  run_tests.sh <m> for m in mnist fashion_mnist cifar10 (the cex/natgrid rows and, when
#                  their outputs are missing or stale, everything else of those models).
#   rq1-f16f64   = run_rq1_f16f64.sh : re-verify (reverify_cex_suite.py, 30/30 at the format) and install
#                  the regenerated float16/float64 natural suites, then the same three run_tests.sh calls.
#   float16-all  = run_float16_all.sh : FLOAT16_ALL=1 NUMPY_EXEC=1 SKIP_EXISTING=1 MODEL_FILTER=<m> run_tests.sh
#                  for m in mnist fashion_mnist emnist_balanced_full emnist_byclass_full higgs_w1024_500k.
#   float64-all  = run_float64_all.sh : likewise with FLOAT64_ALL=1 over mnist fashion_mnist cifar10
#                  emnist_balanced_full emnist_byclass_full higgs_w1024_500k.
#   NB (found 2026-11-05 while writing this): run_tests.sh takes its model filter from $1 only
#   (MODEL_FILTER="${1:-}"), so the MODEL_FILTER=<m> environment variable those two wrappers set is
#   ignored and each of their iterations runs the WHOLE matrix (every model, float32 rows included),
#   with SKIP_EXISTING=1 skipping what already has an up-to-date output. The jobs reproduce that
#   behaviour rather than silently changing it; fix both together if the filter is wanted.
# The two new jobs:
#   all-float32  = Table 1's full sets: the float32 "all" rows (three modes) of every registry model
#                  except higgs_w1024_500k, under NUMPY_EXEC=1 (KIND_FILTER=all MODEL_SKIP=higgs_w1024_500k).
#   higgs-500k   = Table 1's HIGGS row: the float32 "all" rows of higgs_w1024_500k (~3 h single-threaded).
#   recompute-norms = artifact run 2: all-float32 with RECOMPUTE_NORMS=1 (fresh, timed float64 Gram bounds)
#                  into json_results_numpy_run2/ (+ norms/ with the fresh caches); run 1's outputs are kept.
#   Both skip up-to-date outputs unless SKIP_EXISTING is set empty in the environment (SKIP_EXISTING= ...).
# Logs: run_tests.sh writes each certifier's stdout to $RUNLOG (default .log) and per-run count lines to
# summary/counts_<model>.tsv; redirect this script's stdout to keep the job log (e.g. > log.run_job.<job>.txt).
set -euo pipefail
cd "$(dirname "$0")"
JOB="${1:-}"
[[ -n "$JOB" ]] || { sed -n '2,4p' "$0" >&2; exit 2; }
# shellcheck disable=SC1091
source ../venv/bin/activate
BASH="${BASH_BIN:-/opt/homebrew/bin/bash}"   # run_tests.sh needs bash >= 4 (macOS ships 3.2)
DRY="${DRY_RUN:-}"
skip() { echo "# skipped (DRY_RUN): $*" >&2; }

rq1_certify() {   # the loop shared by rq1 and rq1-f16f64 (verbatim from both wrappers)
  local m
  for m in mnist fashion_mnist cifar10; do
    echo "##### $m ($(date))"
    NUMPY_EXEC=1 SKIP_EXISTING=1 "$BASH" ./run_tests.sh "$m"
  done
}

case "$JOB" in
  rq1)
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
    if [[ -n "$DRY" ]]; then skip "python make_natgrid_inputs.py"; else python make_natgrid_inputs.py; fi
    rq1_certify
    echo "##### RQ1 runs done ($(date))"
    ;;
  rq1-f16f64)
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
    if [[ -n "$DRY" ]]; then
      skip "reverify_cex_suite.py on the five regenerated float16/float64 suites and their installation (run_rq1_f16f64.sh lines 19-48)"
    else
      VCR=~/git/verified-certified-robustness/scripts; PC=$(cd .. && pwd); REFS=$PC/models/precomputed
      CAV="$VCR/../cav2025-models"; PY=$VCR/cav2025-artifact-venv/bin/python
      NEW=$VCR/results/natural_grid_cex/f16f64_suites_regen
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
    fi
    rq1_certify
    echo "##### RQ1 float16/float64 runs done ($(date))"
    ;;
  float16-all)
    if [[ -n "$DRY" ]]; then skip "wait for models/provision_numpy16_all_inputs.log to report completion"; else
      while ! grep -q "All float16 numpy-execution input variants provisioned" ../models/provision_numpy16_all_inputs.log 2>/dev/null; do sleep 60; done
    fi
    for m in mnist fashion_mnist emnist_balanced_full emnist_byclass_full higgs_w1024_500k; do
      echo "##### $m ($(date))"
      FLOAT16_ALL=1 NUMPY_EXEC=1 SKIP_EXISTING=1 MODEL_FILTER=$m "$BASH" ./run_tests.sh
    done
    echo "##### float16 runs done ($(date))"
    ;;
  float64-all)
    if [[ -n "$DRY" ]]; then skip "wait for models/provision_numpy64.log to report completion"; else
      while ! grep -q "All float64 numpy-execution input variants provisioned" ../models/provision_numpy64.log 2>/dev/null; do sleep 60; done
    fi
    for m in mnist fashion_mnist cifar10 emnist_balanced_full emnist_byclass_full higgs_w1024_500k; do
      echo "##### $m ($(date))"
      FLOAT64_ALL=1 NUMPY_EXEC=1 SKIP_EXISTING=1 MODEL_FILTER=$m "$BASH" ./run_tests.sh
    done
    echo "##### float64 runs done ($(date))"
    ;;
  all-float32)
    echo "##### all-float32 ($(date))"
    NUMPY_EXEC=1 SKIP_EXISTING="${SKIP_EXISTING-1}" KIND_FILTER=all MODEL_SKIP=higgs_w1024_500k "$BASH" ./run_tests.sh
    echo "##### all-float32 done ($(date))"
    ;;
  higgs-500k)
    echo "##### higgs-500k ($(date))"
    NUMPY_EXEC=1 SKIP_EXISTING="${SKIP_EXISTING-1}" KIND_FILTER=all "$BASH" ./run_tests.sh higgs_w1024_500k
    echo "##### higgs-500k done ($(date))"
    ;;
  recompute-norms)
    # Artifact run 2 (DESIGN §3): the same certifier runs as all-float32, but with every *.norms.json
    # cleared first so the float64 Gram bounds are recomputed and timed (RECOMPUTE_NORMS=1; the shipped
    # caches in ../models/precomputed are not touched), and the outputs written to json_results_numpy_run2/
    # so run 1's outputs survive for the count comparison. The fresh caches are copied into that
    # directory's norms/ at the end (bin/norm_check.py compares them with the shipped ones).
    echo "##### recompute-norms ($(date))"
    RECOMPUTE_NORMS=1 NUMPY_EXEC=1 SKIP_EXISTING="${SKIP_EXISTING-}" KIND_FILTER=all MODEL_SKIP=higgs_w1024_500k \
      JSON_OUT_DIR=json_results_numpy_run2 "$BASH" ./run_tests.sh
    if [[ -z "$DRY" ]]; then mkdir -p json_results_numpy_run2/norms && cp -f ./*.norms.json json_results_numpy_run2/norms/; fi
    echo "##### recompute-norms done ($(date))"
    ;;
  *) echo "run_job.sh: unknown job '$JOB' (rq1 | rq1-f16f64 | float16-all | float64-all | all-float32 | higgs-500k | recompute-norms)" >&2; exit 2 ;;
esac
