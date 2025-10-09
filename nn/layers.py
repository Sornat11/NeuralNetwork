 # Layers and activations
import numpy as np

def relu(x): return np.maximum(0.0, x)
def drelu(y):
    g = np.zeros_like(y)
    g[y > 0] = 1.0
    return g

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def tanh(x): return np.tanh(x)

def dtanh(y):
    return 1.0 - np.square(y)

def init_weights(sizes, hidden_act="relu", seed=42):
    rng = np.random.default_rng(seed)
    params = []
    for i in range(len(sizes)-1):
        fan_in, fan_out = sizes[i], sizes[i+1]
        if hidden_act.lower() == "relu" and i < len(sizes)-2:
            scale = np.sqrt(2.0 / fan_in)  # He initialization
        else:
            scale = np.sqrt(1.0 / fan_in)  # Xavier initialization
        W = rng.normal(0.0, scale, size=(fan_out, fan_in))
        b = np.zeros((fan_out, 1))
        params.append({"W": W, "b": b})
    return params
