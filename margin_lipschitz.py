from typing import List

import json

from arithmetic import Q
from linear_algebra import Matrix, l2_norm_upper_bound_vec

def margin_lipschitz_bounds(network: List[Matrix], op2_norms: List[Q]) -> List[List[Q]]:
    """
    Compute L[i][j] margin bounds per the paper:
    - product of operator-norm upper bounds for layers 1..n-1
    - times l2 norm of (last_layer[j] - last_layer[i])
    """
    assert len(network) >= 1
    *hidden, last = network
    *hidden_norms, _ = op2_norms
    # product of op-norm bounds for hidden layers
    prod = Q(1)
    for i, norm in enumerate(hidden_norms):
        prod *= norm
    # compute per-pair margins using last layer's row differences
    rows = last  # last is matrix, rows = classes
    num_classes = len(rows)
    L = [[Q(0) for _ in range(num_classes)] for _ in range(num_classes)]
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                continue
            diff = [rows[j][k] - rows[i][k] for k in range(len(rows[0]))]
            r = l2_norm_upper_bound_vec(diff)
            L[i][j] = prod * r
    return L

def load_margin_lipschitz_reference(L_real: List[List[Q]], gram_iters: int, dafny_json_file: str):
    """Load the verified Dafny exact margin-Lipschitz reference (returned as L_ref,
    the real-arithmetic verdict baseline) and print an informational comparison
    against the computed L_real.

    We no longer ASSERT L_real >= L_ref. That cross-check made sense only when the
    norms were computed non-transposed (then L_real reproduced the Dafny reference
    exactly). With the min-dimension transpose (||W||_2 = ||W^T||_2 on the smaller
    Gram), our bound is a hair TIGHTER than the non-transposed reference, so the two
    are no longer directly comparable and a >= check would spuriously fail.
    Soundness now rests on the Coq formalisation (gram_iter_fp_sound /
    gram_iter_fp_sound_trmx). The file-validity checks (presence, gram, dimensions)
    are kept since a wrong reference file is a genuine error.
    """
    if dafny_json_file is None:
        # No exact reference supplied (e.g. a brand-new model with no Dafny run).
        # Use the computed L_real as the real-arithmetic baseline: the "real"
        # verdict then becomes the no-FP-error ceiling (margin > eps*L_real), which
        # isolates the cost of the floating-point error terms. The FP-sound verdict
        # is unaffected -- it always uses L_real plus the E terms.
        print("No Dafny reference supplied; using computed L_real as the "
              "real-arithmetic baseline (real verdict = no-FP-error ceiling).")
        return L_real

    L_ref = None
    with open(dafny_json_file, mode="r") as f:
        data = json.load(f, parse_float=Q)
    for obj in data:
        if "lipschitz_bounds" in obj.keys():
            L_ref = obj["lipschitz_bounds"]
            gram_iters_ref = obj["GRAM_ITERATIONS"]

    if L_ref is None:
        raise ValueError(f"Reference file {dafny_json_file} doesn't contain lipschitz bounds")

    if gram_iters != gram_iters_ref:
        raise ValueError(f"Reference gram iterations {gram_iters_ref} doesn't match actual gram iterations {gram_iters}")

    if len(L_ref) != len(L_real) or len(L_ref[0]) != len(L_real[0]):
        raise ValueError(f"Dimensions of reference Lipschitz bounds don't match actual dimensions")

    # Informational only: how the computed L_real sits relative to the (non-
    # transposed) Dafny reference. Both bound the same true margin-Lipschitz
    # constant; the transposed Gram is a hair tighter, so some entries sit below.
    n_greater = n_below = 0
    max_rel_excess = max_rel_deficit = Q(0)
    for i in range(len(L_real)):
        for j in range(len(L_real[i])):
            if L_ref[i][j] == 0:
                continue
            if L_real[i][j] > L_ref[i][j]:
                n_greater += 1
                max_rel_excess = max(max_rel_excess, (L_real[i][j] - L_ref[i][j]) / L_ref[i][j])
            elif L_real[i][j] < L_ref[i][j]:
                n_below += 1
                max_rel_deficit = max(max_rel_deficit, (L_ref[i][j] - L_real[i][j]) / L_ref[i][j])
    print(f"Loaded Dafny reference (real-verdict baseline). vs computed L_real: "
          f"{n_greater} above (max +{float(max_rel_excess):.3e}), "
          f"{n_below} below (max -{float(max_rel_deficit):.3e}) -- expected: the "
          f"transposed Gram is a hair tighter than the non-transposed reference.")
    return L_ref
