import numpy as np


class ActivationReLU:
    def __init__(self):
        self.output: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        # Gradient przechodzi tylko tam, gdzie wejście > 0
        drelu = dvalues.copy()
        drelu[self.output <= 0] = 0
        return drelu


class ActivationSoftmax:
    def __init__(self):
        self.output: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output
