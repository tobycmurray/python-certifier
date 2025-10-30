import numpy as np

from arithmetic import Q
from linear_algebra import transpose

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

def load_vector_from_file(path: str):
    s = load_text_strict(path)
    vec, i = parse_vector(s, 0)
    if i != len(s):
        raise ParseError(f"Unexpected trailing content at position {i}: {s[i:]}")
    return vec    

def load_vector_from_npy_file(path: str):
    arr = np.load(path)
    arr = arr.flatten()
    # Use repr(element) to preserve all available digits — unlike str() which can round
    return [Q(repr(x.item())) for x in arr]    
