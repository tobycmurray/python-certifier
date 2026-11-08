#!/usr/bin/env python
"""Platform variants of the numpy-execution certification input sets, and their provenance records.

The hybrid modes (Hybrid-Centre, Measured-Radii) certify the execution they measure: the certifier
re-runs the compliant numpy forward pass at the centre and REFUSES any instance whose recorded logits
y1 that pass does not reproduce bit for bit (robust_certifier.logits_reproduced). The forward pass is
np.einsum's own sum of products, whose summation order depends on the architecture numpy was built
for, so a set of y1 produced on arm64 (the authors' machine) is reproduced on arm64 hosts and refused
on every x86_64 host (measured 2026-11-08: 10000/10000 MNIST logits differ by 1-7 ulps). Hence each
set carries a provenance record naming the platform that produced its y1, and a set is provisioned
per platform: tests/run_tests.sh selects `<set>_<platform>.json` when INPUT_PLATFORM names one.

Two modes, both running compliant_forward (the function the certifier measures with) on every input:

  relabel   (default) for every numpy-execution set the test matrix certifies in a hybrid mode, write
            the sibling <set>_<platform>.json: y1 recomputed on THIS platform, every other field (x1_file,
            label, max_eps, ...) copied, a provenance record first. Sets whose y1 this platform
            reproduces exactly get no sibling (run_tests.sh then uses the original). Prints, per set, how
            many instances' logits differ, by how many ulps of the largest logit, and whether any argmax
            (predicted label) changed.
  annotate  verify that this platform reproduces every y1 of the original sets, then insert the
            provenance record in place, textually, leaving every other byte of the file unchanged
            (for the authors' arm64 sets, which predate the record). Aborts on a set that is not
            reproduced unless --force.

Usage (from python-certifier/, venv active, no BLAS threads needed):
  python models/provision_platform_inputs.py [--mode relabel|annotate] [--sets PATTERN ...]
         [--out-root DIR] [--jobs N] [--force] [--generator NAME]
  --out-root  write the variants under DIR/<path relative to tests/> instead of beside the originals
              (the artifact image runs this with /results as the root and the outputs are then committed)
  --sets      restrict to sets whose path contains one of the patterns
The provenance record is {"_metadata": {"provenance": {...}}}, a record without x1_file, which every
reader of these files already skips (the counterexample suites' bias metadata uses the same shape).
"""
import argparse
import datetime
import json
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))        # python-certifier/models
ROOT = os.path.dirname(HERE)                              # python-certifier
TESTS = os.path.join(ROOT, "tests")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import compliant_forward as cf                            # noqa: E402
from parsing import load_network_from_file, load_biases_from_file  # noqa: E402
from registry import load_registry                        # noqa: E402

CHUNK = 500


def enumerate_sets(reg):
    """Every numpy-execution input set a hybrid mode certifies (tests/run_tests.sh's matrix under
    NUMPY_EXEC=1): per model the float32 "all" set (_numpy), its float64 and float16 siblings
    (_numpy64, _numpy16) and first-100 slice when present, the counterexample suites at each format
    (cex_<dataset>_<format>[_biased_...]_numpy/counter_examples.json), and for the biased models the
    biased "all" set and the natgrid set. Each entry: (json path, net path, biases path or None, format)."""
    out = []
    for key, r in reg.items():
        net = os.path.join(HERE, r["net"])
        base = os.path.join(TESTS, r["inputs_all"])[:-len(".json")]
        dataset = r["dataset"]
        if r["bias"] is None:
            for sfx, fmt in (("_numpy", "float32"), ("_numpy64", "float64"), ("_numpy16", "float16")):
                out.append((f"{base}{sfx}.json", net, None, fmt))
            # first-100 slices (RQ3's same-100-point ERAN comparison; natively run in every mode)
            for cand in (f"{base}_first100_numpy.json", os.path.join(TESTS, r["inputs_dir"], "inputs_first100_numpy.json")):
                if os.path.exists(cand) and (cand, net, None, "float32") not in out:
                    out.append((cand, net, None, "float32"))
                    break
            for fmt in (r["cex_formats"] or []):
                out.append((os.path.join(TESTS, f"cex_{dataset}_{fmt}_numpy", "counter_examples.json"), net, None, fmt))
        else:
            biases = os.path.join(TESTS, r["biases_txt"])
            sfx = f"_biased_{r['bias']}_{r['bias_pos']}"
            out.append((f"{base}_numpy{sfx}.json", net, biases, "float32"))
            out.append((os.path.join(TESTS, r["inputs_dir"], f"natgrid_numpy{sfx}.json"), net, biases, "float32"))
            for fmt in (r["cex_formats"] or []):
                out.append((os.path.join(TESTS, f"cex_{dataset}_{fmt}{sfx}_numpy", "counter_examples.json"), net, biases, fmt))
    return out


# --- the forward pass, parallel over chunks of records (fork: the prepared model is inherited) --------
_MODEL = None
_DIR = None


def _init(model, d):
    global _MODEL, _DIR
    _MODEL, _DIR = model, d


def _logits(recs):
    out = []
    for r in recs:
        x = np.load(os.path.join(_DIR, os.path.basename(r["x1_file"]))).reshape(-1)
        y = cf.forward_logits(_MODEL, x)
        out.append([float(v) for v in y])                 # exact: the format's values as doubles
    return out


def recompute(records, model, d, jobs):
    chunks = [records[i:i + CHUNK] for i in range(0, len(records), CHUNK)]
    if jobs <= 1 or len(chunks) == 1:
        _init(model, d)
        res = [_logits(c) for c in chunks]
    else:
        with mp.get_context("fork").Pool(jobs, initializer=_init, initargs=(model, d)) as pool:
            res = pool.map(_logits, chunks)
    return [y for c in res for y in c]


def compare(records, ys, fmt):
    """(n differing, max ulps of the largest recorded logit, indices whose argmax changed)."""
    dt = cf._DT[fmt]
    n_diff, max_ulps, argmax_changed = 0, 0.0, []
    for r, y in zip(records, ys):
        y1 = np.asarray([float(v) for v in r["y1"]], dtype=np.float64).astype(dt)
        yn = np.asarray(y, dtype=np.float64).astype(dt)
        if not np.array_equal(y1, yn):
            n_diff += 1
            scale = np.spacing(np.abs(y1).max())
            max_ulps = max(max_ulps, float(np.abs(yn.astype(np.float64) - y1.astype(np.float64)).max() / float(scale)))
        if int(np.argmax(yn)) != int(np.argmax(y1)):
            argmax_changed.append(r.get("index"))
    return n_diff, max_ulps, argmax_changed


def provenance(fmt, biases, src, generator, verified=None):
    p = dict(cf.execution_platform())
    p["execution"] = f"compliant_forward {fmt}" + (" with biases" if biases else "")
    p["generator"] = generator
    p["date"] = datetime.date.today().isoformat()
    if src is not None:
        p["derived_from"] = os.path.basename(src)
        p["note"] = "y1 of every instance recomputed on this platform; x1_file, label and every other field as in derived_from"
    if verified is not None:
        p["note"] = verified
    return {"_metadata": {"provenance": p}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("relabel", "annotate"), default="relabel")
    ap.add_argument("--sets", nargs="*", default=[])
    ap.add_argument("--out-root")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--generator", default="models/provision_platform_inputs.py")
    a = ap.parse_args()
    here = cf.execution_platform()
    print(f"this platform: {json.dumps(here)}; mode {a.mode}; jobs {a.jobs}")
    sets = enumerate_sets(load_registry())
    if a.sets:
        sets = [s for s in sets if any(p in s[0] for p in a.sets)]
    nets = {}
    n_written = 0
    failures = []
    for src, net_path, biases_path, fmt in sets:
        rel = os.path.relpath(src, TESTS)
        if not os.path.exists(src):
            print(f"-- {rel}: absent (not provisioned here), skipped")
            continue
        with open(src) as f:
            data = json.load(f)
        meta = [r for r in data if "x1_file" not in r]
        records = [r for r in data if "x1_file" in r]
        if any("provenance" in (m.get("_metadata") or {}) for m in meta) and a.mode == "annotate" and not a.force:
            print(f"-- {rel}: already carries a provenance record, skipped")
            continue
        key = (net_path, biases_path, fmt)
        if key not in nets:
            net = load_network_from_file(net_path, validate=True)
            biases = load_biases_from_file(biases_path) if biases_path else None
            nets[key] = cf.prepare(net, biases, fmt)
        ys = recompute(records, nets[key], os.path.dirname(src), a.jobs)
        n_diff, max_ulps, changed = compare(records, ys, fmt)
        print(f"-- {rel}: {len(records)} instances at {fmt}; logits differ on {n_diff}; "
              f"max {max_ulps:.1f} ulps of the largest logit; argmax changed on {len(changed)}"
              + (f" {changed[:20]}" if changed else ""))
        if a.mode == "annotate":
            if n_diff and not a.force:
                failures.append(rel)
                print(f"   NOT annotated: this platform does not reproduce the set")
                continue
            note = (f"provenance record added {datetime.date.today().isoformat()} to the authors' set; logits unchanged, "
                    f"verified: the compliant forward pass on this platform reproduces every y1")
            rec = provenance(fmt, biases_path, None, a.generator, verified=note)
            with open(src) as f:
                text = f.read()
            i = text.index("[")
            ins = json.dumps(rec, indent=2)
            text = text[:i + 1] + "\n" + ins + "," + text[i + 1:]
            with open(src, "w") as f:
                f.write(text)
            n_written += 1
            print(f"   annotated in place")
            continue
        if n_diff == 0:
            print(f"   identical to the source: no {here['platform']} variant needed")
            continue
        dst = f"{src[:-len('.json')]}_{here['platform']}.json"
        if a.out_root:
            dst = os.path.join(a.out_root, os.path.relpath(dst, TESTS))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
        out = [provenance(fmt, biases_path, src, a.generator)] + [m for m in meta if "provenance" not in (m.get("_metadata") or {})]
        for r, y in zip(records, ys):
            out.append(dict(r, y1=y))
        with open(dst, "w") as f:
            json.dump(out, f, indent=2)
        n_written += 1
        print(f"   -> {os.path.relpath(dst, a.out_root) if a.out_root else os.path.relpath(dst, TESTS)}")
    print(f"done: {n_written} files written" + (f"; NOT reproduced: {failures}" if failures else ""))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
