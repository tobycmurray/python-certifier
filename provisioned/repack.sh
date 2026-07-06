#!/usr/bin/env bash
# Re-create the committed split bundles from the currently-provisioned artifacts
# in ../models and ../tests. Run this after (re-)provisioning to refresh them.
# This is the inverse of extract.sh and the source of truth for how the bundles
# in this directory are made.
#
# Each bundle is tarred RELATIVE TO the python-certifier root (so paths are
# models/... and tests/...), gzipped, then split into <=45 MB chunks named
# <bundle>.tar.gz.part-aa, .part-ab, ... to stay under GitHub's 100 MB limit.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # python-certifier/provisioned
ROOT="$(cd "$HERE/.." && pwd)"                          # python-certifier
cd "$ROOT"

STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
rm -f "$HERE"/*.tar.gz.part-*

tar czf "$STAGE/certifier-nets.tar.gz"                    models/neural_net_*.txt
tar czf "$STAGE/certifier-inputs-higgs-sweep-10k.tar.gz"  tests/inputs_higgs_w128_n10000 \
                                                          tests/inputs_higgs_w256_n10000 \
                                                          tests/inputs_higgs_w512_n10000 \
                                                          tests/inputs_higgs_w1024_n10000
tar czf "$STAGE/certifier-inputs-higgs-1024-500k.tar.gz"  tests/inputs_higgs_w1024_n500000
tar czf "$STAGE/certifier-inputs-emnist-balanced.tar.gz"  tests/inputs_emnist_balanced_full
tar czf "$STAGE/certifier-inputs-emnist-byclass.tar.gz"   tests/inputs_emnist_byclass_full

for f in "$STAGE"/*.tar.gz; do
  split -b 45m "$f" "$HERE/$(basename "$f").part-"
done
echo "Repacked $(ls "$HERE"/*.tar.gz.part-* | wc -l) parts into $HERE"
