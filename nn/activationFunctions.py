import numpy as np

def relu(Z):
    return np.maximum(0, Z)

def drelu(Z):
    return (Z > 0).astype(float)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def dsigmoid(A):
    return A * (1 - A)