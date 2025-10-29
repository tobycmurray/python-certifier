from dataclasses import dataclass
from typing import List, Optional, Tuple
from arithmetic import Q
from formats import FloatFormat

@dataclass(frozen=True)
class OverflowLayerStats:
    norm2_abs: Q
    norminf: Q
    slack_2: Q
    slack_inf: Q
    ok: bool

@dataclass(frozen=True)
class OverflowReport:
    layers: List[OverflowLayerStats]
    ok: bool
    first_failure: Optional[Tuple[int, str]]  # (layer_idx, "2-norm"|"inf-norm")

def certify_no_overflow_normwise(
    op2_abs: List[Q],          # precomputed ‖|W_l|‖_2 bounds
    inf_norm: List[Q],         # precomputed ‖W_l‖_∞
    radii_prev: List[Q],       # r_{l-1} per layer (same length as layers)
    fmt: FloatFormat,
) -> OverflowReport:
    Fmax_q = Q(str(fmt.Fmax))

    layers: List[OverflowLayerStats] = []
    for l, (a2, ainf, rprev) in enumerate(zip(op2_abs, inf_norm, radii_prev)):
        A = a2 * rprev
        B = ainf * rprev
        s2 = Fmax_q - A
        si = Fmax_q - B
        ok = (s2 > 0) and (si > 0)
        layers.append(OverflowLayerStats(a2, ainf, s2, si, ok))
        if not ok:
            return OverflowReport(layers, False, (l, "2-norm" if s2 <= 0 else "inf-norm"))
    return OverflowReport(layers, True, None)
