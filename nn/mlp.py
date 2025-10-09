 # MLP class/model
import numpy as np
from .layers import relu, drelu, sigmoid, tanh, dtanh, init_weights
from .losses import bce_loss
from .optim import init_adam_state, adam_step
from .utils import iterate_minibatches

class MLP:
    def __init__(self, sizes, hidden_act="relu", seed=42):
        self.sizes = sizes
        self.hidden_act = hidden_act
        self.params = init_weights(sizes, hidden_act, seed)
        self.state = init_adam_state(self.params)
        self.t = 0

    def forward(self, X):
        A = X
        Zs, As = [], [X]
        L = len(self.params)
        for l, p in enumerate(self.params):
            Z = p["W"] @ A + p["b"]
            if l < L-1:
                A = relu(Z) if self.hidden_act == "relu" else tanh(Z)
            else:
                A = sigmoid(Z)
            Zs.append(Z); As.append(A)
        return Zs, As

    def backward(self, Zs, As, y_true):
        L = len(self.params)
        m = y_true.shape[1]
        grads = [None] * L
        dA = (As[-1] - y_true)
        for l in reversed(range(L)):
            A_prev = As[l]
            Z = Zs[l]
            W = self.params[l]["W"]
            if l < L-1:
                if self.hidden_act == "relu":
                    dZ = dA * drelu(As[l+1])
                else:
                    dZ = dA * dtanh(As[l+1])
            else:
                dZ = dA
            dW = (dZ @ A_prev.T) / m
            db = np.mean(dZ, axis=1, keepdims=True)
            dA = (W.T @ dZ)
            grads[l] = {"dW": dW, "db": db}
        return grads

    def train(self, X, y, epochs=2000, batch_size=32, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, seed=123):
        rng = np.random.default_rng(seed)
        for epoch in range(1, epochs+1):
            epoch_loss = 0.0
            batches = 0
            for Xb, yb in iterate_minibatches(X, y, batch_size, rng):
                Zs, As = self.forward(Xb)
                loss = bce_loss(As[-1], yb)
                grads = self.backward(Zs, As, yb)
                self.t += 1
                self.params, self.state = adam_step(self.params, grads, self.state, self.t, lr, beta1, beta2, eps, weight_decay)
                epoch_loss += loss; batches += 1
            if epoch % max(1, epochs // 10) == 0:
                print(f"epoch {epoch:4d} | loss {epoch_loss / max(1,batches):.4f}")

    def predict(self, X, threshold=0.5):
        _, As = self.forward(X)
        p = As[-1]
        return (p >= threshold).astype(np.float32), p
