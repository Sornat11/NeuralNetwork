"""
Unit tests for layers in nn/layers.py
"""
import numpy as np
from nn import layers

def test_init_weights_shape():
    sizes = [2, 4, 1]
    params = layers.init_weights(sizes, hidden_act="relu", seed=42)
    assert len(params) == 2
    assert params[0]["W"].shape == (4, 2)
    assert params[1]["W"].shape == (1, 4)
    assert params[0]["b"].shape == (4, 1)
    assert params[1]["b"].shape == (1, 1)
