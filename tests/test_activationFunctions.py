"""
Unit tests for activation functions in nn/activationFunctions.py
"""
import numpy as np
import pytest
from nn import activationFunctions as af

def test_relu():
    x = np.array([-1, 0, 2])
    out = af.relu(x)
    assert np.allclose(out, [0, 0, 2])

def test_drelu():
    x = np.array([-1, 0, 2])
    out = af.drelu(x)
    assert np.allclose(out, [0, 0, 1])

def test_sigmoid():
    x = np.array([0])
    out = af.sigmoid(x)
    assert np.allclose(out, [0.5])

def test_dsigmoid():
    a = np.array([0.5])
    out = af.dsigmoid(a)
    assert np.allclose(out, [0.25])

def test_tanh():
    x = np.array([0])
    out = af.tanh(x)
    assert np.allclose(out, [0])

def test_dtanh():
    a = np.array([0])
    out = af.dtanh(a)
    assert np.allclose(out, [1])

def test_softmax():
    x = np.array([[1.0, 2.0, 3.0]])  # 2D input
    out = af.softmax(x)
    assert np.allclose(np.sum(out), 1.0)

def test_dsoftmax():
    a = np.array([[0.2, 0.5, 0.3]])  # 2D input
    out = af.dsoftmax(a)
    assert out.shape == (1, 3, 3)
