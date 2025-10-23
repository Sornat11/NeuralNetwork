"""
Unit tests for loss functions in nn/losses.py
"""
import numpy as np
from nn import losses

def test_mse_loss():
    y_hat = np.array([1.0, 2.0, 3.0])
    y_true = np.array([1.0, 2.0, 2.0])
    loss = losses.mse_loss(y_hat, y_true)
    assert np.isclose(loss, 1/3)

def test_dmse_loss():
    y_hat = np.array([1.0, 2.0, 3.0])
    y_true = np.array([1.0, 2.0, 2.0])
    grad = losses.dmse_loss(y_hat, y_true)
    assert np.allclose(grad, [0, 0, 2/3])

def test_bce_loss():
    y_hat = np.array([0.9, 0.1])
    y_true = np.array([1, 0])
    loss = losses.bce_loss(y_hat, y_true)
    assert loss > 0

def test_cce_loss():
    y_hat = np.array([[0.7, 0.2, 0.1]])
    y_true = np.array([[1, 0, 0]])
    loss = losses.cce_loss(y_hat, y_true)
    assert loss > 0
