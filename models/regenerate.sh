#!/usr/bin/env bash
# Regenerate the sound-by-construction certifier .txt weight files from the
# cav2025 CSV weights.
#
# Each file is produced by make_certifier_format_from_model.py, which builds the
# actual Keras model via doitlib and writes model.get_weights() -- the exact
# float32 weights the model executes -- then SELF-CHECKS by reloading the file
# through the certifier's own loader and asserting exact rational equality. The
# script refuses to emit a file that does not round-trip, so these .txt files
# are sound by construction (unlike Tobler et al.'s 5-decimal-place serialisation
# that we replace).
#
# The .txt files are large derived artifacts and are gitignored; this script is
# the committed source of truth. Run it (in the TF 2.13.0 venv -- see
# ../requirements.txt) before running the test suite or building the artifact.
#
# All paths are derived from this script's location; no absolute paths. The
# cav2025 models are expected in the sibling verified-certified-robustness repo
# (override with CAV_MODELS=... if your layout differs).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../python-certifier/models
ROOT="$(cd "$HERE/.." && pwd)"                          # .../python-certifier
CAV="${CAV_MODELS:-$ROOT/../verified-certified-robustness/cav2025-models}"
PY="${PY:-$ROOT/venv/bin/python}"
GEN="$ROOT/make_certifier_format_from_model.py"

"$PY" "$GEN" mnist "[128,128,128,128,128,128,128,128]" 28 \
  "$CAV/2025-01-25_09:27:46-mnist/model_weights_epsilon_0.45_[128,128,128,128,128,128,128,128]_500/" \
  "$HERE/neural_net_mnist_epsilon_0.45_[128,128,128,128,128,128,128,128]_500.txt"

"$PY" "$GEN" fashion_mnist "[256,128,128,128,128,128,128,128,128,128,128,128]" 28 \
  "$CAV/2025-01-30_10:58:01-fashion_mnist/model_weights_epsilon_0.26_[256,128,128,128,128,128,128,128,128,128,128,128]_500/" \
  "$HERE/neural_net_fashion_mnist_epsilon_0.26_[256,128,128,128,128,128,128,128,128,128,128,128]_500.txt"

"$PY" "$GEN" cifar10 "[512,256,128,128,128,128,128,128]" 32 \
  "$CAV/2025-01-28_20:39:32-cifar10/model_weights_epsilon_0.1551_[512,256,128,128,128,128,128,128]_800/" \
  "$HERE/neural_net_cifar10_epsilon_0.1551_[512,256,128,128,128,128,128,128]_800.txt"

echo "All models regenerated and self-checked."
