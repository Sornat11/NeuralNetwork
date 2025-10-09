 # Layers and activations
import numpy as np

def init_weights(sizes, hidden_act="relu", seed=42):
    rng = np.random.default_rng(seed)
    params = []
    for i in range(len(sizes) - 1):
        W = rng.normal(0, 1, (sizes[i+1], sizes[i])) * np.sqrt(2 / sizes[i])
        b = np.zeros((sizes[i+1], 1))
        params.append({"W": W, "b": b})
    return params