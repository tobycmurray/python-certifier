

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
