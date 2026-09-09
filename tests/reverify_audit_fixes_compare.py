#!/usr/bin/env python3
"""Compare the re-verification outputs (json_results_numpy_reverify/) with the
pre-fix baseline (json_results_numpy/): summary counts before/after for every
run, per-instance verdict differences (certified / certified_real), and the
new per-instance fields (overflow_ok; logits_reproduced for the hybrid modes).

Usage: reverify_audit_fixes_compare.py BASELINE_DIR NEW_DIR LOG_DIR
"""
import glob
import json
import os
import sys


def load(path):
    with open(path) as f:
        d = json.load(f)
    return d[0], d[1:]   # header, records


def counts(recs):
    return (len(recs),
            sum(1 for r in recs if r["certified"]),
            sum(1 for r in recs if r["certified_real"]))


def main():
    base_dir, new_dir, log_dir = sys.argv[1:4]
    rows = []
    problems = []
    for new_path in sorted(glob.glob(os.path.join(new_dir, "*.json"))):
        name = os.path.basename(new_path)
        base_path = os.path.join(base_dir, name)
        _, new = load(new_path)
        n_new, ok_new, real_new = counts(new)
        hybrid = "hybrid" in name
        is_cex = name.endswith("_cex.json")
        ovf_present = all("overflow_ok" in r for r in new)
        ovf_refused = sum(1 for r in new if not r.get("overflow_ok", True))
        lg_present = (not hybrid) or all("logits_reproduced" in r for r in new)
        lg_refused = sum(1 for r in new if not r.get("logits_reproduced", True))
        if not ovf_present:
            problems.append(f"{name}: overflow_ok field missing on some records")
        if not lg_present:
            problems.append(f"{name}: logits_reproduced field missing on some records")
        if ovf_refused:
            problems.append(f"{name}: {ovf_refused} instances refused for overflow")
        if lg_refused:
            problems.append(f"{name}: {lg_refused} instances refused (logits not reproduced)")
        if is_cex:
            if ok_new != 0:
                problems.append(f"{name}: FP-sound certifier certified {ok_new} counter-examples")
            if real_new != n_new:
                problems.append(f"{name}: real certifier certified only {real_new}/{n_new} counter-examples")
        if "biased" in name and not is_cex and ok_new != 0:
            problems.append(f"{name}: biased model 'all' certified {ok_new} (expected 0)")

        if os.path.exists(base_path):
            _, base = load(base_path)
            n_b, ok_b, real_b = counts(base)
            diffs = []
            if len(base) != len(new):
                problems.append(f"{name}: record count differs (baseline {len(base)}, new {len(new)})")
            for i, (rb, rn) in enumerate(zip(base, new)):
                if rb["certified"] != rn["certified"] or rb["certified_real"] != rn["certified_real"]:
                    diffs.append((i, rb["certified"], rn["certified"], rb["certified_real"], rn["certified_real"],
                                  rb.get("float_conservatism"), rn.get("float_conservatism")))
            for d in diffs:
                problems.append(f"{name}: instance {d[0]}: certified {d[1]}->{d[2]}, certified_real {d[3]}->{d[4]}, "
                                f"float_conservatism {d[5]}->{d[6]}")
            # largest relative change in the (mean per-pair) float conservatism, informational
            max_rel = 0.0
            for rb, rn in zip(base, new):
                fb, fn = rb.get("float_conservatism"), rn.get("float_conservatism")
                if fb and fn:
                    max_rel = max(max_rel, abs(fn - fb) / abs(fb))
            base_str = f"{n_b}/{ok_b}/{real_b}"
            diff_str = str(len(diffs))
        else:
            base_str, diff_str, max_rel = "(no baseline)", "-", float("nan")
        rows.append((name, base_str, f"{n_new}/{ok_new}/{real_new}", diff_str, ovf_refused,
                     (lg_refused if hybrid else "-"), max_rel))

    print("run\tbaseline N/certified/real\tnew N/certified/real\tverdict diffs\trefused(overflow)\trefused(logits)\tmax rel dFC")
    for r in rows:
        print("\t".join(str(x) if not isinstance(x, float) else f"{x:.2e}" for x in r))
    print()
    if problems:
        print("PROBLEMS / DIFFERENCES:")
        for p in problems:
            print("  " + p)
    else:
        print("OK: no verdict differences, no refusals, all cex rejected by FP and accepted by real, "
              "biased 'all' = 0 certified, overflow_ok (and logits_reproduced) fields present.")


if __name__ == "__main__":
    main()
