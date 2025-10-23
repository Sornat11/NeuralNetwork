"""
Layer initialization utilities for neural networks.
"""

import numpy as np

def init_weights(sizes, hidden_act="relu", seed=42):
    """
    Initialize weights and biases for each layer in the network.
    Args:
        sizes (list): List of layer sizes, e.g. [input_dim, h1, ..., output_dim].
        hidden_act (str): Activation function for hidden layers ('relu' or 'sigmoid').
        seed (int): Random seed for reproducibility.
    Returns:
        list: List of parameter dicts (with 'W', 'b').
    """
    rng = np.random.default_rng(seed)
    params = []
    for i in range(len(sizes) - 1):
        # He initialization for ReLU, Xavier for sigmoid
        if hidden_act == "relu":
            W = rng.normal(0, 1, (sizes[i+1], sizes[i])) * np.sqrt(2 / sizes[i])
        else:
            W = rng.normal(0, 1, (sizes[i+1], sizes[i])) * np.sqrt(1 / sizes[i])
        b = np.zeros((sizes[i+1], 1))
        params.append({"W": W, "b": b})
    return params