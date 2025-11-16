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
        momentum: float = 0.0,  # Momentum coefficient (0.0 = bez momentum)
    ):
        self.weights = weight_scale * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output: np.ndarray | None = None
        self.learning_rate = learning_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.momentum = momentum

        # Velocity dla momentum (inicjalizowane na 0)
        self.weight_velocity = np.zeros_like(self.weights)
        self.bias_velocity = np.zeros_like(self.biases)

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

        # SGD update with momentum
        if self.momentum > 0:
            # Momentum: velocity = momentum * velocity - learning_rate * gradient
            self.weight_velocity = self.momentum * self.weight_velocity - self.learning_rate * dweights
            self.bias_velocity = self.momentum * self.bias_velocity - self.learning_rate * dbiases

            # Update weights with velocity
            self.weights += self.weight_velocity
            self.biases += self.bias_velocity
        else:
            # Standard SGD (bez momentum)
            self.weights -= self.learning_rate * dweights
            self.biases -= self.learning_rate * dbiases

        return dinputs
