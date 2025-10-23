"""
Loss functions and their derivatives for neural networks.
Includes: MSE, binary cross-entropy, categorical cross-entropy.
All functions use NumPy arrays as input.
"""

import numpy as np

def mse_loss(y_hat, y_true):
    """
    Mean Squared Error loss (regression).
    Args:
        y_hat (np.ndarray): Predicted values.
        y_true (np.ndarray): True values.
    Returns:
        float: MSE loss value.
    """
    return np.mean((y_true - y_hat) ** 2)

def dmse_loss(y_hat, y_true):
    """
    Derivative of MSE loss with respect to predictions.
    Args:
        y_hat (np.ndarray): Predicted values.
        y_true (np.ndarray): True values.
    Returns:
        np.ndarray: Gradient of loss.
    """
    return 2 * (y_hat - y_true) / y_true.size

def bce_loss(y_hat, y_true, eps=1e-7):
    """
    Binary Cross-Entropy loss (binary classification).
    Args:
        y_hat (np.ndarray): Predicted probabilities.
        y_true (np.ndarray): True binary labels (0 or 1).
        eps (float): Small value to avoid log(0).
    Returns:
        float: BCE loss value.
    """
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat))

def dbce_loss(y_hat, y_true, eps=1e-7):
    """
    Derivative of BCE loss with respect to predictions.
    Args:
        y_hat (np.ndarray): Predicted probabilities.
        y_true (np.ndarray): True binary labels (0 or 1).
        eps (float): Small value to avoid division by zero.
    Returns:
        np.ndarray: Gradient of loss.
    """
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return (y_hat - y_true) / (y_hat * (1 - y_hat) * y_true.size)

def cce_loss(y_hat, y_true, eps=1e-7):
    """
    Categorical Cross-Entropy loss (multi-class classification).
    Args:
        y_hat (np.ndarray): Predicted probabilities (n_samples, n_classes).
        y_true (np.ndarray): True one-hot labels (n_samples, n_classes).
        eps (float): Small value to avoid log(0).
    Returns:
        float: CCE loss value.
    """
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_hat), axis=1))

def dcce_loss(y_hat, y_true, eps=1e-7):
    """
    Derivative of CCE loss with respect to predictions.
    Args:
        y_hat (np.ndarray): Predicted probabilities (n_samples, n_classes).
        y_true (np.ndarray): True one-hot labels (n_samples, n_classes).
        eps (float): Small value to avoid division by zero.
    Returns:
        np.ndarray: Gradient of loss.
    """
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -y_true / y_hat / y_true.shape[0]
