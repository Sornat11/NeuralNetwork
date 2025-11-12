import numpy as np

from utils.seed import set_seed

set_seed(0)


class LayerDense:

    def __init__(
        self,
        n_inputs: int,
        n_neurons: int,
        weight_scale: float = 0.1,
        learning_rate: float = 0.01,
        l1_reg: float = 0.0,
        l2_reg: float = 0.0,
    ):
        self.weights = weight_scale * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output: np.ndarray | None = None
        self.learning_rate = learning_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, dvalues: np.ndarray, inputs: np.ndarray) -> tuple:
        # Gradients w.r.t weights
        dweights = np.dot(inputs.T, dvalues)
        # Add L1 regularization gradient
        if self.l1_reg > 0:
            dweights += self.l1_reg * np.sign(self.weights)
        # Add L2 regularization gradient
        if self.l2_reg > 0:
            dweights += 2 * self.l2_reg * self.weights
        # Gradients w.r.t biases
        dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradients w.r.t inputs (for previous layer)
        dinputs = np.dot(dvalues, self.weights.T)
        # SGD update with regularization
        self.weights -= self.learning_rate * dweights
        self.biases -= self.learning_rate * dbiases
        return dinputs
