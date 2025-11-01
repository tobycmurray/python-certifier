from __future__ import annotations
from fractions import Fraction

Q = Fraction  # rational type alias

def qstr(q: Fraction) -> str:
    """
    Exact decimal expansion of a Fraction without float rounding.
    - If it terminates, returns all digits.
    - If it repeats, returns a string with the repeating part in parentheses, e.g. "0.(3)".
    """
    if q == 0:
        return "0"

    sign = '-' if q < 0 else ''
    n = abs(q.numerator)
    d = q.denominator

    # Integer part
    int_part = n // d
    rem = n % d
    if rem == 0:
        return f"{sign}{int_part}"

    # Long division for fractional part with cycle detection
    digits = []
    seen = {}  # remainder -> index in digits
    idx_repeat_start = None

    while rem != 0 and rem not in seen:
        seen[rem] = len(digits)
        rem *= 10
        digit = rem // d
        rem = rem % d
        digits.append(str(digit))

    if rem == 0:
        # Terminates
        frac = ''.join(digits)
        return f"{sign}{int_part}.{frac}"
    else:
        # Repeats
        start = seen[rem]
        nonrep = ''.join(digits[:start])
        rep = ''.join(digits[start:])
        if nonrep:
            return f"{sign}{int_part}.{nonrep}({rep})"
        else:
            return f"{sign}{int_part}.({rep})"

DECIMALS = 16
DEC_SCALE = 10 ** DECIMALS
ROUNDING_PRECISION = DECIMALS

# Parameters from Fig. 7
SQRT_ERR = Q(1, 10**11)          # 1e-11 as a rational
SQRT_ITERATIONS = 2_000_000

def to_q(x: int | float | str | Fraction) -> Q:
    """Convert to rational Q safely (floats go through string to avoid binary artifacts)."""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, float):
        return Q(str(x))  # avoid binary float rounding noise
    return Q(x)

def trunc16_scalar(x: Q) -> Q:
    """
    Truncate a rational number to 16 decimal places (towards zero), per the paper's
    "truncate to 16 decimal places" step.
    """
    y = x * DEC_SCALE
    num = y.numerator // y.denominator   # Python's // is floor for ints, including negatives
    return Q(num, DEC_SCALE)

def round_down(x: Q) -> Q:
    """
    Dafny-accurate RoundDown (requires x <= 0):
      r := x
      i := 0
      while r != floor(r) and i < ROUNDING_PRECISION:
          r := r * 10
          i := i + 1
      if r != floor(r): r := r - 1
      r := floor(r)
      while i > 0:
          r := r / 10
          i := i - 1
    Returns a rational on the 10^{-k} grid (k <= ROUNDING_PRECISION) with r <= x.
    """
    if x > 0:
        raise ValueError("round_down requires x <= 0")
    r = x
    i = 0
    def is_integer(q: Q) -> bool:
        return (q.denominator == 1)
    while (not is_integer(r)) and (i < ROUNDING_PRECISION):
        r = r * 10
        i += 1
    if not is_integer(r):
        r = r - 1
    # floor to be safe (integer already if branch above didn’t fire)
    r = Q(r.numerator // r.denominator, 1)
    while i > 0:
        r = r / 10
        i -= 1
    return r

def round_up(x: Q) -> Q:
    """
    Dafny-accurate RoundUp (requires x >= 0):
      r := x
      i := 0
      while r != floor(r) and i < ROUNDING_PRECISION:
          r := r * 10
          i := i + 1
      if r != floor(r): r := r + 1
      r := floor(r)
      while i > 0:
          r := r / 10
          i := i - 1
    Returns a rational on the 10^{-k} grid (k <= ROUNDING_PRECISION) with r >= x.
    """
    if x < 0:
        raise ValueError(f"round_up requires x >= 0. got x = {x}")
    r = x
    i = 0
    # helper: is_integer
    def is_integer(q: Q) -> bool:
        return (q.denominator == 1)
    # scale until integer or reach precision
    while (not is_integer(r)) and (i < ROUNDING_PRECISION):
        r = r * 10
        i += 1
    if not is_integer(r):
        r = r + 1
    # floor r (ensure integer)
    r = Q(r.numerator // r.denominator, 1)
    # scale back
    while i > 0:
        r = r / 10
        i -= 1
    return r

def sqrt_upper_bound(x: Q) -> Q:
    """
    Dafny-accurate Heron's method with per-iteration RoundUp to 16 dp:
      r := (x < 1) ? 1 : x
      while i < SQRT_ITERATIONS:
        old_r := r
        r := RoundUp( (r + x/r) / 2 )
        if old_r - r < SQRT_ERR: return r
    Ensures r >= sqrt(x).
    """
    if x < 0:
        raise ValueError("sqrt_upper_bound: negative input")
    if x == 0:
        return Q(0)
    r = Q(1) if x < 1 else x
    i = 0
    while i < SQRT_ITERATIONS:
        old_r = r
        r = round_up((r + x / r) / 2)
        i += 1
        if old_r - r < SQRT_ERR:
            return r
    return r
        
