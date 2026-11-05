#!/usr/bin/env bash
# Bash loader for models/registry.tsv (the model registry; see its header for the columns).
#   source models/registry.sh
#   registry_get KEY COLUMN     -> prints the cell ("-" = not applicable); exit 1 unknown key, 2 unknown column
#   registry_keys               -> prints every model key, one per line, in file order
# Override the file with REGISTRY_TSV=/path/to/registry.tsv.
REGISTRY_TSV="${REGISTRY_TSV:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/registry.tsv}"

registry_get() {
  awk -F'\t' -v key="$1" -v col="$2" '
    /^#/ || NF == 0 { next }
    !hdr { hdr = 1; for (i = 1; i <= NF; i++) c[$i] = i
           if (!(col in c)) { printf "registry_get: unknown column %s\n", col > "/dev/stderr"; bad = 2; exit }
           next }
    $1 == key { print $c[col]; found = 1; exit }
    END { if (bad) exit bad
          if (!found) { printf "registry_get: unknown key %s\n", key > "/dev/stderr"; exit 1 } }' "$REGISTRY_TSV"
}

registry_keys() {
  awk -F'\t' '/^#/ || NF == 0 { next } !hdr { hdr = 1; next } { print $1 }' "$REGISTRY_TSV"
}
