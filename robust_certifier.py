# robust_certifier.py
# Unverified Python implementation of the verified certifier described in:
# "A Formally Verified Robustness Certifier for Neural Networks" (CAV 2025).
# - Follows Gram iteration (Fig. 6) and SqrtUpperBound (Fig. 7).
# - Arithmetic uses exact rationals (fractions.Fraction).
#
# NOTE: This mirrors the algorithmic structure; it is not a formally verified artifact.

from __future__ import annotations
from fractions import Fraction
from typing import List, Tuple

import datetime

Q = Fraction  # rational type alias

from fractions import Fraction

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
        raise ValueError("round_up requires x >= 0")
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

# -------- Linear algebra over rationals --------

Matrix = List[List[Q]]
Vector = List[Q]

def dims(M: Matrix) -> Tuple[int, int]:
    return len(M), len(M[0]) if M else (0, 0)

def zeros(m: int, n: int) -> Matrix:
    return [[Q(0) for _ in range(n)] for _ in range(m)]

def is_zero_matrix(M: Matrix) -> bool:
    return all(x == 0 for row in M for x in row)

def transpose(M: Matrix) -> Matrix:
    m, n = dims(M)
    return [[M[i][j] for i in range(m)] for j in range(n)]

def mv_product(M: Matrix, v: Vector) -> Vector:
    m, n = dims(M)
    assert len(v) == n
    out: Vector = []
    for i in range(m):
        s = Q(0)
        row = M[i]
        for j in range(n):
            s += row[j] * v[j]
        out.append(s)
    return out

def mm_product(A: Matrix, B: Matrix) -> Matrix:
    m, k = dims(A)
    k2, n = dims(B)
    assert k == k2
    Bt = transpose(B)
    out = zeros(m, n)
    for i in range(m):
        Ai = A[i]
        for j in range(n):
            s = Q(0)
            Bj = Bt[j]
            for t in range(k):
                s += Ai[t] * Bj[t]
            out[i][j] = s
    return out

def mtm(M: Matrix) -> Matrix:
    """
    Specialised MTM: returns M^T * M using symmetry, row-wise scanning.
    """
    m, n = dims(M)  # m rows, n cols
    out = zeros(n, n)
    # compute upper triangle, reuse symmetry
    for i in range(n):
        if i%10==0:
            print(f"[DBG] mtm i={i}")
        for j in range(i, n):
            s = Q(0)
            for r in range(m):
                s += M[r][i] * M[r][j]
            out[i][j] = s
            if j != i:
                out[j][i] = s
    return out

def matrix_div_scalar(M: Matrix, r: Q) -> Matrix:
    if r == 0:
        raise ZeroDivisionError("division by zero in matrix_div_scalar")
    m, n = dims(M)
    out = zeros(m, n)
    for i in range(m):
        for j in range(n):
            out[i][j] = M[i][j] / r
    return out

def matrix_sub(A: Matrix, B: Matrix) -> Matrix:
    m, n = dims(A)
    assert dims(B) == (m, n)
    out = zeros(m, n)
    for i in range(m):
        for j in range(n):
            out[i][j] = A[i][j] - B[i][j]
    return out

def truncate_with_error(M: Matrix) -> Tuple[Matrix, Q]:
    m, n = dims(M)
    T = zeros(m, n)
    sq_sum = Q(0)
    for i in range(m):
        for j in range(n):
            x = M[i][j]
            if x > 0:
                t = round_up(x)
            elif x < 0:
                t = round_down(x)
            else:
                t = Q(0)
            T[i][j] = t
            e = t - x
            sq_sum += e * e
    e_frob = sqrt_upper_bound(sq_sum)
    return T, e_frob

def frobenius_norm_upper_bound(M: Matrix) -> Q:
    sq_sum = Q(0)
    for row in M:
        for x in row:
            sq_sum += x * x
    return sqrt_upper_bound(sq_sum)

# -------- Fig. 6: Gram iteration + Expand --------

def gram_iteration(M: Matrix, n: int) -> Q:
    M_cur = [row[:] for row in M]
    a: List[Tuple[Q, Q]] = []
    i = 0
    while i != n:
        print(f"[DBG] gram iteration {i}")
        Mp = mtm(M_cur)
        r = Q(1) if is_zero_matrix(Mp) else frobenius_norm_upper_bound(Mp)
        Mn = matrix_div_scalar(Mp, r)
        M_trunc, e = truncate_with_error(Mn)
        a = [(r, e)] + a
        M_cur = M_trunc
        i += 1

    s0 = frobenius_norm_upper_bound(M_cur)

    ret = s0
    for (r, e) in a:
        arg = r * (ret + e)
        ret = sqrt_upper_bound(arg)

    return ret

# -------- Network representation & bounds --------

def relu_vec(v: Vector) -> Vector:
    return [x if x > 0 else Q(0) for x in v]

def forward(network: List[Matrix], x: Vector) -> Vector:
    """
    Forward pass with ReLU after each non-final layer; no biases
    """
    v = x[:]
    L = len(network)
    for idx, W in enumerate(network):
        v = mv_product(transpose(W), v)
        if idx < L - 1:
            v = relu_vec(v)
    return v

def layer_opnorm_upper_bound(W: Matrix, gram_iters: int) -> Q:
    return gram_iteration(W, gram_iters)

def l2_norm_upper_bound_vec(v: Vector) -> Q:
    # Exact sum of squares, then sqrt_upper_bound
    sq_sum = Q(0)
    for x in v:
        sq_sum += x * x
    return sqrt_upper_bound(sq_sum)

def margin_lipschitz_bounds(network: List[Matrix], gram_iters: int) -> List[List[Q]]:
    """
    Compute L[i][j] margin bounds per the paper:
    - product of operator-norm upper bounds for layers 1..n-1
    - times l2 norm of (last_layer[j] - last_layer[i])
    """
    assert len(network) >= 1
    *hidden, last = network
    # product of op-norm bounds for hidden layers
    prod = Q(1)
    for i, W in enumerate(hidden):
        print(f"[DBG] Computing operator norm for hidden layer {i}...")
        prod *= layer_opnorm_upper_bound(W, gram_iters)
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

def certify(v_prime: Vector, epsilon: Q, L: List[List[Q]]) -> bool:
    """
    Implements the certification rule (Fig. 4):
    - x = argmax v'
    - for all i != x: require v'[x] - v'[i] > epsilon * L[i][x]
    """
    # argmax index
    x = max(range(len(v_prime)), key=lambda k: v_prime[k])
    vx = v_prime[x]
    for i in range(len(v_prime)):
        if i == x:
            continue
        if epsilon * L[i][x] >= vx - v_prime[i]:
            return False
    return True

# Convenience: build matrix/vector from nested numbers
def make_matrix(data) -> Matrix:
    return [[to_q(x) for x in row] for row in data]

def make_vector(data) -> Vector:
    return [to_q(x) for x in data]

############################## code to parse input ########################

ALLOWED_CHARS = set("0123456789.,[]-")

class ParseError(ValueError):
    pass

def load_text_strict(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        s = f.read().strip()
    if any(c not in ALLOWED_CHARS for c in s):
        bad = sorted(set(c for c in s if c not in ALLOWED_CHARS))
        raise ParseError(f"Illegal character(s) found: {bad}. Allowed are only 0-9 . , [ ] -")
    if not s:
        raise ParseError("Empty file.")
    return s

def parse_number(s: str, i: int):
    n = len(s)
    start = i
    # optional leading '-'
    if i < n and s[i] == '-':
        i += 1
        if i >= n or not s[i].isdigit():
            raise ParseError(f"'-' must be followed by digits at position {i}")
    # first block of digits
    if i >= n or not s[i].isdigit():
        raise ParseError(f"Expected digit at position {i}")
    while i < n and s[i].isdigit():
        i += 1
    # dot
    if i >= n or s[i] != '.':
        raise ParseError(f"Expected '.' in real number at position {i}")
    i += 1
    # second block of digits
    if i >= n or not s[i].isdigit():
        raise ParseError(f"Expected digit(s) after '.' at position {i}")
    while i < n and s[i].isdigit():
        i += 1
    num_str = s[start:i]
    return Q(num_str), i

def expect_char(s: str, i: int, ch: str):
    if i >= len(s) or s[i] != ch:
        raise ParseError(f"Expected '{ch}' at position {i}")
    return i + 1

def parse_vector(s: str, i: int):
    i = expect_char(s, i, '[')
    vec = []
    x, i = parse_number(s, i)
    vec.append(x)
    while i < len(s) and s[i] == ',':
        i += 1
        x, i = parse_number(s, i)
        vec.append(x)
    i = expect_char(s, i, ']')
    return vec, i

def parse_matrix(s: str, i: int):
    i = expect_char(s, i, '[')
    rows = []
    v, i = parse_vector(s, i)
    rows.append(v)
    row_len = len(v)
    while i < len(s) and s[i] == ',':
        i += 1
        v, i = parse_vector(s, i)
        if len(v) != row_len:
            raise ParseError(f"Row length mismatch in matrix at position {i}: expected {row_len}, got {len(v)}")
        rows.append(v)
    i = expect_char(s, i, ']')
    return rows, i

def parse_network(s: str):
    i = 0
    matrices = []
    M, i = parse_matrix(s, i)
    matrices.append(M)
    while i < len(s) and s[i] == ',':
        i += 1
        M, i = parse_matrix(s, i)
        matrices.append(M)
    if i != len(s):
        raise ParseError(f"Unexpected trailing content at position {i}")
    return matrices

def check_chain_dimensions(network):
    def dims(M): return (len(M), len(M[0]) if M else 0)
    for idx in range(len(network) - 1):
        r, c = dims(network[idx])
        r2, c2 = dims(network[idx + 1])
        if c != r2:
            raise ParseError(
                f"Layer dimension mismatch between layer {idx} and {idx+1}: "
                f"W[{idx}] is {r}x{c}, W[{idx+1}] is {r2}x{c2}. Expected cols(current) == rows(next)."
            )

def load_network_from_file(path: str, validate: bool = True):
    s = load_text_strict(path)
    net = parse_network(s)
    if validate:
        check_chain_dimensions(net)
    return [transpose(M) for M in net]

import sys

def main():
    if len(sys.argv) != 3:
        print("Usage: main <neural_network_input.txt> <GRAM_ITERATIONS>")
        sys.exit(1)

    network_file = sys.argv[1]
    try:
        gram_iters = int(sys.argv[2])
    except ValueError:
        print("Error: <GRAM_ITERATIONS> must be an integer.")
        sys.exit(1)

    try:
        net = load_network_from_file(network_file, validate=True)
    except ParseError as e:
        print(f"Error parsing network file: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: file not found: {network_file}")
        sys.exit(1)

    print(f"Loaded network with {len(net)} layers; running {gram_iters} Gram iterations per layer...")

    L = margin_lipschitz_bounds(net, gram_iters)

    print("\nMargin Lipschitz Bounds (L[i][j]):")
    for i, row in enumerate(L):
        formatted = [f"{qstr(v)}" for v in row]
        print(f"{i}: [{', '.join(formatted)}]")

if __name__ == "__main__":
    main()

