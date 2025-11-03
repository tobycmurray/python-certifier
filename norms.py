from dataclasses import dataclass
from typing import List, Dict
import hashlib
import json

from arithmetic import Q, qstr
from linear_algebra import layer_infinity_norm, layer_opnorm_upper_bound, abs_matrix, Matrix
import time

@dataclass(frozen=True)
class Norms:
  inf_norms: List[Q]
  op2_norms: List[Q]
  op2_abs_norms: List[Q]
  times: Dict[str,float]

def compute_norms(net: List[Matrix], gram_iters: int) -> Norms:
    inf_norms = []
    op2_norms = []
    op2_abs_norms = []

    times={}
    times["inf"] = 0.0
    times["op2"] = 0.0
    times["op2_abs"] = 0.0

    for i, W in enumerate(net):
        print(f"Computing norms for layer {i}, ...")
        print(f"  infinity norm...")
        start = time.perf_counter()
        inf_norm = layer_infinity_norm(W)
        time_inf = time.perf_counter() - start

        print(f"  operator norm...")
        start = time.perf_counter()
        op2_norm = layer_opnorm_upper_bound(W, gram_iters)
        time_op2 = time.perf_counter() - start

        print(f"  operator norm of abs...")
        start = time.perf_counter()
        op2_abs_norm = layer_opnorm_upper_bound(abs_matrix(W), gram_iters)
        time_op2_abs = time.perf_counter() - start

        times["inf"] += time_inf
        times["op2"] += time_op2
        times["op2_abs"] += time_op2_abs

        inf_norms.append(inf_norm)
        op2_norms.append(op2_norm)
        op2_abs_norms.append(op2_abs_norm)

    return Norms(inf_norms=inf_norms, op2_norms=op2_norms, op2_abs_norms=op2_abs_norms, times=times)

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
    summary["inf_norms"] = norms.inf_norms
    summary["op2_norms"] = norms.op2_norms
    summary["op2_abs_norms"] = norms.op2_abs_norms

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
    inf_norms = [Q(n) for n in summary["inf_norms"]]
    op2_norms = [Q(n) for n in summary["op2_norms"]]
    op2_abs_norms = [Q(n) for n in summary["op2_abs_norms"]]
    return Norms(inf_norms=inf_norms, op2_norms=op2_norms, op2_abs_norms=op2_abs_norms, times={})

def hash_file_contents(filename: str) -> str:
    return hashlib.sha256(open(filename,'rb').read()).hexdigest()
