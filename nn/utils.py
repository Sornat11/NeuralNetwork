 # Utility functions
import numpy as np

def iterate_minibatches(X, y, batch_size, rng):
    m = X.shape[1]
    perm = rng.permutation(m)
    for i in range(0, m, batch_size):
        idx = perm[i:i+batch_size]
        yield X[:, idx], y[:, idx]
