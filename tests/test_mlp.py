 # Unit tests for MLP
import numpy as np
from nn.mlp import MLP
from data.xor import generate_xor

def test_xor_learn():
    X, y = generate_xor(reps=10, noise=0.0)
    model = MLP(sizes=(2, 8, 1), hidden_act="relu", seed=42)
    model.train(X, y, epochs=3000, batch_size=1, lr=1e-2)
    y_hat = model.predict(X)
    acc = np.mean(y_hat == y)
    assert acc >= 0.75, f"Accuracy too low: {acc}"