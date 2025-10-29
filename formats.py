from dataclasses import dataclass
from math import ldexp

@dataclass(frozen=True)
class FloatFormat:
    name: str
    p: int              # precision in bits (incl. implicit 1)
    emin: int           # minimum normal exponent (unbiased)
    emax: int           # maximum finite exponent (unbiased)
    # Derived (IEEE-754, round-to-nearest-even):
    Fmax: float         # largest finite positive
    eps: float          # machine epsilon (nextafter(1,2) - 1) = 2^(1-p)
    u: float            # unit roundoff = eps/2 = 2^(-p)
    min_normal: float   # smallest positive normal = 2^emin
    denorm_min: float   # smallest positive subnormal = 2^(emin - (p-1))

def _derive(name: str, p: int, emin: int, emax: int) -> FloatFormat:
    # eps  = 2^(1-p)
    eps = ldexp(1.0, 1 - p)
    # u    = 2^(-p)
    u = ldexp(1.0, -p)
    # min_normal = 2^emin
    min_normal = ldexp(1.0, emin)
    # denorm_min = 2^(emin - (p-1)) = min_normal * eps
    denorm_min = ldexp(1.0, emin - (p - 1))
    # Fmax = (2 - 2^(1-p)) * 2^emax
    Fmax = (2.0 - ldexp(1.0, 1 - p)) * ldexp(1.0, emax)
    return FloatFormat(
        name=name, p=p, emin=emin, emax=emax,
        Fmax=Fmax, eps=eps, u=u,
        min_normal=min_normal, denorm_min=denorm_min
    )

# Authoritative IEEE-754 binary formats (unbiased exponent bounds):
# - binary32:   p=24,  emin = -126, emax = 127
# - bfloat16:   p=8,   emin = -126, emax = 127   (same exponent range as fp32)
# - binary64:   p=53,  emin = -1022, emax = 1023
_REGISTRY = {
    "float32":  (24, -126, 127),
    "fp32":     (24, -126, 127),
    "bfloat16": (8,  -126, 127),
    "bf16":     (8,  -126, 127),
    "float64":  (53, -1022, 1023),
    "fp64":     (53, -1022, 1023),
    "double":   (53, -1022, 1023),
}

def get_float_format(name: str) -> FloatFormat:
    key = (name or "float32").lower()
    try:
        p, emin, emax = _REGISTRY[key]
    except KeyError:
        raise NotImplementedError(f"Format '{name}' not implemented.")
    return _derive(key, p, emin, emax)
