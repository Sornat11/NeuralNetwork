import numpy as np

"""
Activation functions and their derivatives for neural networks.
Includes: ReLU, sigmoid, tanh, softmax.
All functions use NumPy arrays as input.
"""

def relu(Z):
    """
    Rectified Linear Unit activation function.
    Args:
        Z (np.ndarray): Input array.
    Returns:
        np.ndarray: Output after applying ReLU.
    """
    return np.maximum(0, Z)

def drelu(Z):
    """
    Derivative of ReLU activation function.
    Args:
        Z (np.ndarray): Input array.
    Returns:
        np.ndarray: Derivative (0 or 1).
    """
    return (Z > 0).astype(float)

def sigmoid(Z):
    """
    Sigmoid activation function.
    Args:
        Z (np.ndarray): Input array.
    Returns:
        np.ndarray: Output after applying sigmoid.
    """
    return 1 / (1 + np.exp(-Z))

def dsigmoid(A):
    """
    Derivative of sigmoid activation function.
    Args:
        A (np.ndarray): Output of sigmoid(Z).
    Returns:
        np.ndarray: Derivative.
    """
    return A * (1 - A)

def tanh(Z):
    """
    Hyperbolic tangent activation function.
    Args:
        Z (np.ndarray): Input array.
    Returns:
        np.ndarray: Output after applying tanh.
    """
    return np.tanh(Z)

def dtanh(A):
    """
    Derivative of tanh activation function.
    Args:
        A (np.ndarray): Output of tanh(Z).
    Returns:
        np.ndarray: Derivative.
    """
    return 1 - np.square(A)

def softmax(Z):
    """
    Softmax activation function (for multi-class classification).
    Args:
        Z (np.ndarray): Input array (2D: samples x classes).
    Returns:
        np.ndarray: Probabilities for each class.
    """
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def dsoftmax(A):
    """
    Derivative of softmax activation function (returns Jacobian for each sample).
    Args:
        A (np.ndarray): Output of softmax(Z), shape (n_samples, n_classes).
    Returns:
        np.ndarray: Array of Jacobian matrices, shape (n_samples, n_classes, n_classes).
    """
    n = A.shape[0]
    jacobians = np.zeros((n, A.shape[1], A.shape[1]))
    for i in range(n):
        a = A[i].reshape(-1, 1)
        jacobians[i] = np.diagflat(a) - np.dot(a, a.T)
    return jacobians