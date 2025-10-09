import numpy as np

def generate_xor(reps=1, noise=0.0, seed=42):
    rng = np.random.default_rng(seed)
    X = np.array([[0,0,1,1],[0,1,0,1]], dtype=np.float32)
    y = np.array([[0,1,1,0]], dtype=np.float32)
    X = np.tile(X, reps)
    y = np.tile(y, reps)
    if noise > 0.0:
        X = X + rng.normal(0, noise, X.shape)
    return X, y