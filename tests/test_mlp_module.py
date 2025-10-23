"""
Unit tests for MLP in nn/mlp.py
"""
import numpy as np
from nn.mlp import MLP

def test_mlp_forward_shape():
    model = MLP(sizes=(2, 4, 1), hidden_act="relu", seed=42)
    X = np.random.randn(2, 5)  # (features, samples)
    _, As = model.forward(X)
    out = As[-1]
    assert out.shape == (1, 5) or out.shape == (5, 1) or out.shape == (5,)

def test_mlp_predict_shape():
    model = MLP(sizes=(2, 4, 1), hidden_act="relu", seed=42)
    X = np.random.randn(2, 5)
    y_hat = model.predict(X)
    assert y_hat.shape == (1, 5) or y_hat.shape == (5, 1) or y_hat.shape == (5,)
