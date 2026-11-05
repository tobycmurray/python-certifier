"""Python loader for models/registry.tsv (the model registry; see its header for the columns).

    from registry import load_registry, REGISTRY_TSV
    reg = load_registry()                 # {key: {column: value}}; "-" cells become None
    reg["mnist"]["eps"]                   # '0.3'
    reg["mnist"]["cex_formats"]           # ['float32', 'float16', 'float64']  (list; None when "-")

Values are kept as strings (the scripts pass them through verbatim); only "-" -> None and
cex_formats -> list are converted. Run as a script to check the file against the inputs it names.
"""
import os
import sys

REGISTRY_TSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "registry.tsv")
NA = "-"


def load_registry(path=REGISTRY_TSV):
    reg = {}
    header = None
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            cells = line.split("\t")
            if header is None:
                header = cells
                continue
            if len(cells) != len(header):
                raise ValueError(f"{path}: row {cells[0]!r} has {len(cells)} cells, header has {len(header)}")
            row = {k: (None if v == NA else v) for k, v in zip(header, cells)}
            if row["cex_formats"] is not None:
                row["cex_formats"] = row["cex_formats"].split(",")
            if row["key"] in reg:
                raise ValueError(f"{path}: duplicate key {row['key']!r}")
            reg[row["key"]] = row
    if header is None:
        raise ValueError(f"{path}: no header row")
    return reg


def _check(reg, root):
    """Every path the registry names exists (provisioned nets/inputs may be absent: reported, not fatal),
    and eps equals the max_eps of the first record of the model's inputs_all file."""
    import json
    ok = True
    for key, r in reg.items():
        for col, base in (("net", "models"), ("ref_rq1", "models/precomputed"), ("ref_all", "models/precomputed"),
                          ("inputs_all", "tests"), ("inputs_dir", "tests"), ("biases_txt", "tests")):
            if r[col] is None:
                continue
            p = os.path.join(root, base, r[col])
            if not os.path.exists(p):
                print(f"{key}: {col} missing: {p}")
                ok = False if col not in ("net", "inputs_all", "inputs_dir") else ok
        p = os.path.join(root, "tests", r["inputs_all"])
        if os.path.exists(p):
            with open(p) as f:
                head = f.read(4096)
            i = head.find('"max_eps"')
            got = head[i:].split(":", 1)[1].split(",", 1)[0].strip().strip('"') if i >= 0 else None
            if got != r["eps"]:
                print(f"{key}: eps {r['eps']} != max_eps {got} of {p}")
                ok = False
            else:
                print(f"{key}: eps {r['eps']} == max_eps of {r['inputs_all']}")
    return ok


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reg = load_registry()
    print(f"{len(reg)} models: {' '.join(reg)}")
    sys.exit(0 if _check(reg, root) else 1)
