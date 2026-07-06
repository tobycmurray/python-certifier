#!/usr/bin/env bash
# Reassemble + extract the committed provisioned certifier artifacts into the
# python-certifier tree: the .txt weight files (-> models/) and the per-model
# certification inputs (-> tests/inputs_*/). These are the gitignored derived
# artifacts that provision_image_model_weights.sh / provision_higgs_emnist_inputs.sh
# would otherwise regenerate (slowly, via TensorFlow + tfds). Committing them here
# (compressed + split) lets the artifact build skip provisioning -- and the whole
# TF/tfds venv -- entirely.
#
# Naming scheme: each bundle is  certifier-<what>.tar.gz  split into <=45 MB chunks
#   certifier-<what>.tar.gz.part-aa, .part-ab, ...   (they sort in order, so
#   `cat certifier-<what>.tar.gz.part-* > certifier-<what>.tar.gz` rebuilds it).
# Every bundle's tar paths are relative to the python-certifier root, so we extract
# with `tar -C <root>`. See repack.sh for how the bundles are (re)created.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # python-certifier/provisioned
ROOT="$(cd "$HERE/.." && pwd)"                          # python-certifier

shopt -s nullglob
firsts=("$HERE"/*.tar.gz.part-aa)
if (( ${#firsts[@]} == 0 )); then
  echo "extract.sh: no bundle parts (*.tar.gz.part-aa) found in $HERE" >&2
  exit 1
fi
for first in "${firsts[@]}"; do
  base="${first%.part-aa}"           # .../certifier-<what>.tar.gz
  echo ">>> $(basename "$base")"
  cat "$base".part-* > "$base"       # reassemble (parts sort aa,ab,ac,... in order)
  tar xzf "$base" -C "$ROOT"         # extract into python-certifier/{models,tests}
  rm -f "$base"
done
echo "Provisioned artifacts extracted into $ROOT/{models, tests}."
