"""
Unit tests for utility functions in nn/utils.py
"""
import numpy as np
from nn import utils

def test_set_seed():
    utils.set_seed(42)
    a = np.random.rand()
    utils.set_seed(42)
    b = np.random.rand()
    assert a == b

def test_log(capsys):
    utils.log("Hello")
    captured = capsys.readouterr()
    assert "Hello" in captured.out

def test_accuracy():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])
    acc = utils.accuracy(y_true, y_pred)
    assert np.isclose(acc, 0.75)
