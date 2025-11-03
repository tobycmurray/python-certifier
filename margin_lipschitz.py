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

    if L_ref != L_real:
        raise ValueError(f"Reference Lipschitz constants differ from computed ones")
