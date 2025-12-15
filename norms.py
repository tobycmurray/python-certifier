from dataclasses import dataclass
from typing import List, Dict
import hashlib
import json

from arithmetic import Q, qstr
from linear_algebra import layer_infinity_norm, layer_opnorm_upper_bound, abs_matrix, max_row_l2_norm, Matrix
import time

@dataclass(frozen=True)
class Norms:
  max_row_inf_norms: List[Q]   # max_r ||W_r||_∞ = max absolute entry per layer (for M_layer overflow check)
  op2_norms: List[Q]           # spectral norm per layer (for radii + deviation alpha)
  op2_abs_norms: List[Q]       # spectral norm of |W| per layer (for deviation alpha/beta)
  max_row_l2_norms: List[Q]    # max_r ||W_r||_2 = max row L2 norm per layer (for S_layer overflow check)
  times: Dict[str,float]

def compute_norms(net: List[Matrix], gram_iters: int) -> Norms:
    max_row_inf_norms_list = []
    op2_norms = []
    op2_abs_norms = []
    max_row_l2_norms_list = []

    times={}
    times["max_row_inf"] = 0.0
    times["op2"] = 0.0
    times["op2_abs"] = 0.0
    times["max_row_l2"] = 0.0

    for i, W in enumerate(net):
        print(f"Computing norms for layer {i}, ...")
        print(f"  max row inf norm (for M_layer)...")
        start = time.perf_counter()
        max_row_inf = layer_infinity_norm(W)
        time_max_row_inf = time.perf_counter() - start

        print(f"  max row L2 norm (for S_layer)...")
        start = time.perf_counter()
        max_row_l2 = max_row_l2_norm(W)
        time_max_row_l2 = time.perf_counter() - start

        print(f"  operator norm...")
        start = time.perf_counter()
        op2_norm = layer_opnorm_upper_bound(W, gram_iters)
        time_op2 = time.perf_counter() - start

        print(f"  operator norm of abs (for deviation)...")
        start = time.perf_counter()
        op2_abs_norm = layer_opnorm_upper_bound(abs_matrix(W), gram_iters)
        time_op2_abs = time.perf_counter() - start

        times["max_row_inf"] += time_max_row_inf
        times["op2"] += time_op2
        times["op2_abs"] += time_op2_abs
        times["max_row_l2"] += time_max_row_l2

        max_row_inf_norms_list.append(max_row_inf)
        op2_norms.append(op2_norm)
        op2_abs_norms.append(op2_abs_norm)
        max_row_l2_norms_list.append(max_row_l2)

    return Norms(max_row_inf_norms=max_row_inf_norms_list, op2_norms=op2_norms, op2_abs_norms=op2_abs_norms,
                 max_row_l2_norms=max_row_l2_norms_list, times=times)

class QEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Q):
            return qstr(obj)
        return super().default(obj)

def save_norms(hsh: str, gram_iters: int, norms: Norms, filename: str):
    summary = {}
    summary["hash"] = hsh
    summary["gram_iters"] = gram_iters
    summary["times_secs"] = norms.times
    summary["max_row_inf_norms"] = norms.max_row_inf_norms
    summary["op2_norms"] = norms.op2_norms
    summary["op2_abs_norms"] = norms.op2_abs_norms
    summary["max_row_l2_norms"] = norms.max_row_l2_norms

    with open(filename, "w") as f:
        f.write(json.dumps(summary, indent=2, cls=QEncoder))
        f.flush()


def load_norms(hsh: str, gram_iters: int, filename: str) -> Norms:
    with open(filename, "r") as f:
        summary = json.load(f)
    if hsh != summary["hash"]:
        raise Exception(f"hash value {summary['hash']} doesn't match expected hash {hsh}")
    if gram_iters != summary["gram_iters"]:
        raise Exception(f"gram iters {summary['gram_iters']} doesn't match expected gram iters {gram_iters}")

    op2_norms = [Q(n) for n in summary["op2_norms"]]
    op2_abs_norms = [Q(n) for n in summary["op2_abs_norms"]]

    # Handle backwards compatibility for max_row_inf_norms:
    # - New files have "max_row_inf_norms" (max absolute entry = max_r ||W_r||_∞)
    # - Old files have "inf_norms" (max row sum = ||W||_∞), which is a conservative upper bound
    if "max_row_inf_norms" in summary:
        max_row_inf_norms = [Q(n) for n in summary["max_row_inf_norms"]]
    elif "inf_norms" in summary:
        print(f"WARNING: Norm file {filename} uses old 'inf_norms' format (max row sum).")
        print(f"WARNING: This is a conservative upper bound on 'max_row_inf_norms' (max absolute entry).")
        print(f"WARNING: Consider regenerating norm file for tighter overflow bounds.")
        max_row_inf_norms = [Q(n) for n in summary["inf_norms"]]
    else:
        raise Exception(f"Norm file {filename} missing both 'max_row_inf_norms' and 'inf_norms'")

    # Handle backwards compatibility for max_row_l2_norms:
    # - New files have "max_row_l2_norms" (max_r ||W_r||_2)
    # - Old files don't have it; fall back to op2_abs_norms (|||W|||_2), which is a conservative upper bound
    if "max_row_l2_norms" in summary:
        max_row_l2_norms = [Q(n) for n in summary["max_row_l2_norms"]]
    else:
        print(f"WARNING: Norm file {filename} missing 'max_row_l2_norms'.")
        print(f"         Falling back to 'op2_abs_norms' (spectral norm of |W|) as conservative upper bound.")
        print(f"         Consider regenerating norm file for tighter overflow bounds.")
        max_row_l2_norms = op2_abs_norms

    return Norms(max_row_inf_norms=max_row_inf_norms, op2_norms=op2_norms, op2_abs_norms=op2_abs_norms,
                 max_row_l2_norms=max_row_l2_norms, times={})

def hash_file_contents(filename: str) -> str:
    return hashlib.sha256(open(filename,'rb').read()).hexdigest()
