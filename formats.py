from dataclasses import dataclass
from math import ldexp
from arithmetic import Q

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
# - binary16:   p=11,  emin = -14, emax = 15
# - bfloat16:   p=8,   emin = -126, emax = 127   (same exponent range as fp32)
# - binary64:   p=53,  emin = -1022, emax = 1023
_REGISTRY = {
    "float32":   (24, -126, 127),
    "fp32":      (24, -126, 127),
    "binary32":  (24, -126, 127),
    "single":    (24, -126, 127),

    "float16":   (11,  -14, 15),
    "fp16":      (11,  -14, 15),
    "binary16":  (11,  -14, 15),
    "half":      (11,  -14, 15),

    "bfloat16":  (8,  -126, 127),
    "bf16":      (8,  -126, 127),

    "float64":   (53, -1022, 1023),
    "fp64":      (53, -1022, 1023),
    "binary64":  (53, -1022, 1023),
    "double":    (53, -1022, 1023),
}

def get_float_format(name: str) -> FloatFormat:
    key = (name or "float32").lower()
    try:
        p, emin, emax = _REGISTRY[key]
    except KeyError:
        raise NotImplementedError(f"Format '{name}' not implemented. Supported formats {_REGISTRY.keys()}")
    return _derive(key, p, emin, emax)

# ==============================================================================
# Floating-point error accumulation factors
# ==============================================================================

def gamma_n(n: int, u: Q) -> Q:
    """Relative error accumulation factor for n operations.

    γ_n = (n·u) / (1 - n·u)

    This is the standard forward error analysis term (1+u)^n - 1 ≈ n·u,
    used to bound accumulated roundoff error in dot products and matrix operations.

    Args:
        n: Number of operations (e.g., layer input dimension)
        u: Unit roundoff (e.g., 2^(-p) for p-bit precision)

    Returns:
        γ_n as a rational

    Raises:
        ValueError: If n·u >= 1 (error model breaks down)
    """
    nu = Q(n) * u
    if nu >= 1:
        raise ValueError(
            f"n·u = {float(nu)} >= 1. Error model requires n·u < 1 (n={n}, u={float(u)})"
        )
    return nu / (1 - nu)


def a_dot(n: int, u: Q, a_mul: Q) -> Q:
    """Absolute error bound for dot products with potential subnormals.

    a_dot(n) = (1 + γ_{n-1}) · n · a_mul

    where a_mul is the rounding error for a single subnormal product.
    For round-to-nearest: a_mul = denorm_min / 2.

    This bounds the accumulated absolute error from subnormal intermediate
    results in a length-n dot product.

    Args:
        n: Number of products in dot product
        u: Unit roundoff
        a_mul: Subnormal rounding error (typically denorm_min / 2)

    Returns:
        a_dot(n) as a rational
    """
    if n <= 1:
        # For n=1, no accumulation
        return Q(n) * a_mul
    # For n>1, include accumulation factor
    gamma_nm1 = gamma_n(n - 1, u) if n > 1 else Q(0)
    return (Q(1) + gamma_nm1) * Q(n) * a_mul
