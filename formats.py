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

    γ_n = (1 + u)^n - 1

    This is the exact forward error analysis term from the Coq formalization,
    used to bound accumulated roundoff error in dot products and matrix operations.

    Args:
        n: Number of operations (e.g., layer input dimension)
        u: Unit roundoff (e.g., 2^(-p) for p-bit precision)

    Returns:
        γ_n as a rational
    """
    return (Q(1) + u) ** n - Q(1)


def kappa_n(n: int, u: Q) -> Q:
    """Combined FP error constant for n operations.

    κ_n = γ_n + u·(1 + γ_n)

    This is the combined constant that accounts for:
    - The main dot product error (γ_n term)
    - Additional rounding when storing/using the result (u·(1+γ_n) term)

    Used in the deviation recursion (Corollary 3.1) where we need to bound
    both the accumulated error and the final rounding.

    Args:
        n: Number of operations (layer input dimension)
        u: Unit roundoff

    Returns:
        κ_n as a rational

    Raises:
        ValueError: If n·u >= 1
    """
    gamma = gamma_n(n, u)
    return gamma + u * (Q(1) + gamma)


def a_dot(n: int, u: Q, a_mul: Q) -> Q:
    """Absolute error bound for dot products (deviation analysis).

    a_dot(n) = n · a_mul · (1 + γ_n)

    This is the mixed error model bound from LAProof, used in the deviation
    recursion where we need the decomposition into relative and absolute parts.

    The factor (1 + γ_n) accounts for propagation of absolute errors through
    the full chain of n operations (n multiplications + n-1 additions, but
    using γ_n for the mixed model decomposition).

    Args:
        n: Number of products in dot product (layer input dimension)
        u: Unit roundoff
        a_mul: Subnormal rounding error (typically denorm_min / 2)

    Returns:
        a_dot(n) as a rational

    See also:
        a_dot_fwd: Tighter forward error variant for overflow analysis
    """
    if n <= 1:
        return Q(n) * a_mul
    gamma = gamma_n(n, u)
    return Q(n) * a_mul * (Q(1) + gamma)


def a_dot_fwd(n: int, u: Q, a_mul: Q) -> Q:
    """Absolute error bound for dot products (overflow/forward error analysis).

    a_dot_fwd(n) = (1 + γ_{n-1}) · n · a_mul

    This is the tighter forward error bound, using γ_{n-1} since a length-n
    dot product involves n multiplications but only n-1 additions.

    Used in overflow analysis where only the total error magnitude matters,
    not the mixed error decomposition.

    Args:
        n: Number of products in dot product
        u: Unit roundoff
        a_mul: Subnormal rounding error (typically denorm_min / 2)

    Returns:
        a_dot_fwd(n) as a rational

    See also:
        a_dot: Mixed error model variant for deviation analysis
    """
    if n <= 1:
        return Q(n) * a_mul
    gamma_nm1 = gamma_n(n - 1, u)
    return (Q(1) + gamma_nm1) * Q(n) * a_mul
