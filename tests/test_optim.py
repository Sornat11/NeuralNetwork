"""
Unit tests for optimizers in nn/optim.py
"""
import numpy as np
from nn import optim

def test_sgd_step():
    params = [{"W": np.array([[1.0, 2.0], [3.0, 4.0]]), "b": np.array([[0.5, 0.5]])}]
    grads = [{"dW": np.array([[0.1, 0.2], [0.3, 0.4]]), "db": np.array([[0.05, 0.05]])}]
    optim.sgd_step(params, grads, lr=0.1)
    assert np.allclose(params[0]["W"], [[0.99, 1.98], [2.97, 3.96]])

def test_init_adam_state():
    params = [{"W": np.array([[1.0, 2.0], [3.0, 4.0]]), "b": np.array([[0.5, 0.5]])}]
    state = optim.init_adam_state(params)
    assert "mW" in state[0] and "mb" in state[0] and "vW" in state[0] and "vb" in state[0]

def test_adam_step():
    params = [{"W": np.array([[1.0, 2.0], [3.0, 4.0]]), "b": np.array([[0.5, 0.5]])}]
    grads = [{"dW": np.array([[0.1, 0.2], [0.3, 0.4]]), "db": np.array([[0.05, 0.05]])}]
    state = optim.init_adam_state(params)
    optim.adam_step(params, grads, state, t=1, lr=0.1)
    assert params[0]["W"].shape == (2, 2)
