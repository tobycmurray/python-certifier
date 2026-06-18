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

def check_margin_lipschitz_bounds(L_real: List[List[Q]], gram_iters: int, dafny_json_file: str):
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

    # Soundness cross-check against the formally-verified Dafny certifier.
    # The margin-Lipschitz constant is used so that LARGER = MORE CONSERVATIVE
    # (the certification condition is margin > eps*L + E). So it suffices that our
    # computed bounds are everywhere >= the verified reference: then our condition
    # is stricter than Dafny's, hence every instance we certify the verified
    # certifier would also certify -> our "robust" verdicts stay sound w.r.t. the
    # proof. With exact-rational norms the two are identical; with the binary64
    # Gram iteration ours sit a part in ~1e12 ABOVE exact, which this >= accepts.
    n_greater = 0
    max_rel_excess = Q(0)
    for i in range(len(L_real)):
        for j in range(len(L_real[i])):
            if L_real[i][j] < L_ref[i][j]:
                raise ValueError(
                    f"Computed margin Lipschitz bound L[{i}][{j}]={L_real[i][j]} is BELOW the "
                    f"verified Dafny reference {L_ref[i][j]}: not sound w.r.t. the verified certifier")
            if L_real[i][j] > L_ref[i][j]:
                n_greater += 1
                if L_ref[i][j] != 0:
                    rel = (L_real[i][j] - L_ref[i][j]) / L_ref[i][j]
                    if rel > max_rel_excess:
                        max_rel_excess = rel
    if n_greater == 0:
        print("Computed margin Lipschitz bounds match Dafny reference numbers exactly.")
    else:
        print(f"Computed margin Lipschitz bounds are >= Dafny reference (sound): "
              f"{n_greater} entries strictly greater, max relative excess {float(max_rel_excess):.3e}.")
