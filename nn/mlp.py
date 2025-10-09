import numpy as np
from .layers import init_weights
from .activationFunctions import  relu, drelu, sigmoid, dsigmoid

class MLP:
    def __init__(self, sizes, hidden_act="relu", seed=42):
        self.sizes = sizes
        self.hidden_act = hidden_act
        self.params = init_weights(sizes, hidden_act, seed)
        self.rng = np.random.default_rng(seed)

    def forward(self, X):
        As = [X]
        Zs = []
        for i, p in enumerate(self.params):
            Z = p["W"] @ As[-1] + p["b"]
            Zs.append(Z)
            if i < len(self.params) - 1:
                if self.hidden_act == "relu":
                    A = relu(Z)
                else:
                    A = sigmoid(Z)
            else:
                A = sigmoid(Z)
            As.append(A)
        return Zs, As

    def predict(self, X):
        _, As = self.forward(X)
        return (As[-1] > 0.5).astype(float)

    def train(self, X, y, epochs=2000, batch_size=32, lr=0.01):
        m = X.shape[1]
        for epoch in range(epochs):
            Zs, As = self.forward(X)
            grads = [None] * len(self.params)
            # Output
            dA = As[-1] - y
            for i in reversed(range(len(self.params))):
                Z = Zs[i]
                A_prev = As[i]
                if i == len(self.params) - 1:
                    dZ = dA * dsigmoid(As[-1])
                else:
                    if self.hidden_act == "relu":
                        dZ = dA * drelu(Z)
                    else:
                        dZ = dA * dsigmoid(As[i+1])
                dW = dZ @ A_prev.T / m
                db = np.sum(dZ, axis=1, keepdims=True) / m
                grads[i] = {"dW": dW, "db": db}
                if i > 0:
                    dA = self.params[i]["W"].T @ dZ
            # Update weights
            for p, g in zip(self.params, grads):
                p["W"] -= lr * g["dW"]
                p["b"] -= lr * g["db"]
            if epoch % 100 == 0:
                loss = np.mean(-(y * np.log(As[-1] + 1e-7) + (1 - y) * np.log(1 - As[-1] + 1e-7)))