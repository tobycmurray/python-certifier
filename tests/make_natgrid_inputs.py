#!/usr/bin/env python
"""Build the RQ1 grid-instance input sets for run_tests.sh kind "natgrid".

The instances are the certified biased-model test images that the k/255-grid search
(verified-certified-robustness/scripts/natural_grid_cexs.py, run_natural_grid_cexs.sh)
refutes at the paper's standard radii: MNIST 24, Fashion-MNIST 204, CIFAR-10 370.
Each is a test image x (float32, on the grid) at the standard eps, with the biased
model's compliant-numpy float32 logits y1 -- exactly the record the biased numpy
"all" sets already carry for that index. So each set here is the subset of the
corresponding *_numpy_biased_*.json restricted to the grid search's found indices,
written next to it so the x npy paths resolve. Every record is cross-checked
against the grid search's own artefacts before it is written:
  * x1 npy == natgrid_<tag>_biased_points/idx<i>_x.npy, bit for bit (float32);
  * y1 == the logits the search wrote to natgrid_<tag>_biased.certifier_input.txt
    (150-digit decimal, lossless for float32 values) and argmax y1 == the label
    the search certified at eps;
  * max_eps == the search's eps; the search record says found / certified_at_eps /
    on grid / dist <= eps / labels differ.
The extra "natgrid" field records the refuting grid point's distance and label.
"""
import json, os, sys
import numpy as np

VCR = os.path.expanduser("~/git/verified-certified-robustness/scripts/results/natural_grid_cex")
HERE = os.path.dirname(os.path.abspath(__file__))
SETS = [  # natgrid tag, biased numpy all-set, output, eps, expected count
    ("mnist",   "all_mnist_test_inputs/test_inputs_epsilon_0.3_numpy_biased_1e6_end.json",
                "all_mnist_test_inputs/natgrid_numpy_biased_1e6_end.json", "0.3", 24),
    ("fashion", "all_fashion_mnist_test_inputs/test_inputs_epsilon_0.25_numpy_biased_3e6_end.json",
                "all_fashion_mnist_test_inputs/natgrid_numpy_biased_3e6_end.json", "0.25", 204),
    ("cifar",   "all_cifar10_test_inputs/all_test_inputs_numpy_biased_4e6_end.json",
                "all_cifar10_test_inputs/natgrid_numpy_biased_4e6_end.json", "0.141", 370),
]

def main():
    for tag, allf, outf, eps, n_expected in SETS:
        allf = os.path.join(HERE, allf); outf = os.path.join(HERE, outf)
        with open(f"{VCR}/natgrid_{tag}_biased.json") as f:
            g = json.load(f)
        summ = g["summary"]
        found = summ["found_indices"]
        assert len(found) == summ["n_found"] == n_expected, (tag, len(found), summ["n_found"])
        assert summ["eps"] == eps and summ["format"] == "float32", summ
        recs = g["records"]
        recs = ({int(k): v for k, v in recs.items()} if isinstance(recs, dict)
                else {r["index"]: r for r in recs})
        with open(allf) as f:
            alls = {r["index"]: r for r in json.load(f)}
        with open(f"{VCR}/natgrid_{tag}_biased.certifier_input.txt") as f:
            certin = [line.split() for line in f if line.strip()]
        assert len(certin) == len(found), (tag, len(certin), len(found))
        out = []
        for k, idx in enumerate(found):
            r, rec = alls[idx], recs[idx]
            assert rec["found"] and rec["certified_at_eps"] and rec["x_on_grid"] and rec["xadv_on_grid"] \
                and rec["dist_le_eps"] and rec["labels_differ"] and rec["label_x_as_given"], (tag, idx, rec)
            assert r["max_eps"] == eps, (tag, idx, r["max_eps"])
            x_all = np.load(os.path.join(os.path.dirname(allf), os.path.basename(r["x1_file"]))).reshape(-1)
            x_grid = np.load(f"{VCR}/natgrid_{tag}_biased_points/idx{idx}_x.npy").reshape(-1)
            assert x_all.dtype == np.float32 and x_grid.dtype == np.float32, (x_all.dtype, x_grid.dtype)
            assert np.array_equal(x_all, x_grid), (tag, idx)
            assert certin[k][1] == eps, (tag, k, certin[k][1])
            y_cert = [float(v) for v in certin[k][0].split(",")]
            y1 = [float(v) for v in r["y1"]]
            assert y_cert == y1, (tag, idx, y_cert, y1)
            assert int(np.argmax(y1)) == rec["label"] == rec["label_x"], (tag, idx)
            out.append(dict(r, natgrid={"dist": rec["dist"], "label_xadv": rec["label_xadv"],
                                        "R": rec["R"], "pixels_changed": rec["pixels_changed"]}))
        with open(outf, "w") as f:
            json.dump(out, f, indent=1)
        print(f"{tag}: {len(out)} grid instances (eps {eps}) -> {os.path.relpath(outf, HERE)}")

if __name__ == "__main__":
    main()
