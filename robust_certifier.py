# robust_certifier.py
# Unverified Python implementation of the verified certifier described in:
# "A Formally Verified Robustness Certifier for Neural Networks" (CAV 2025).
# - Follows Gram iteration (Fig. 6) and SqrtUpperBound (Fig. 7).
# - Arithmetic uses exact rationals (fractions.Fraction).
#
# NOTE: This mirrors the algorithmic structure; it is not a formally verified artifact.

from typing import List, Tuple

import sys

from parsing import load_network_from_file, ParseError
from arithmetic import Q, qstr, sqrt_upper_bound
from linear_algebra import Matrix, Vector, mtm, is_zero_matrix, frobenius_norm_upper_bound, matrix_div_scalar, \
    truncate_with_error, l2_norm_upper_bound_vec, layer_opnorm_upper_bound, layer_infinity_norm, abs_matrix

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

    inf_norms = []
    op2_norms = []
    op2_abs_norms = []

    for i, W in enumerate(net):
        print(f"Computing norms for layer {i}, ...")
        print(f"  infinity norm...")
        inf_norm = layer_infinity_norm(W)
        print(f"  operator norm...") 
        op2_norm = layer_opnorm_upper_bound(W, gram_iters)
        print(f"  operator norm of abs...")
        op2_abs_norm = layer_opnorm_upper_bound(abs_matrix(W), gram_iters)

        inf_norms.append(inf_norm)
        op2_norms.append(op2_norm)
        op2_abs_norms.append(op2_abs_norm)

    
    L = margin_lipschitz_bounds(net, op2_norms)

    print("\nMargin Lipschitz Bounds (L[i][j]):")
    for i, row in enumerate(L):
        formatted = [f"{qstr(v)}" for v in row]
        print(f"{i}: [{', '.join(formatted)}]")

if __name__ == "__main__":
    main()

