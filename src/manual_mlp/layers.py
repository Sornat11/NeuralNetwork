import numpy as np

from utils.seed import set_seed

set_seed(0)


class LayerDense:

    def __init__(self, n_inputs: int, n_neurons: int, weight_scale: float = 0.1, learning_rate: float = 0.01):
        self.weights = weight_scale * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output: np.ndarray | None = None
        self.learning_rate = learning_rate

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, dvalues: np.ndarray, inputs: np.ndarray) -> tuple:
        # Gradienty względem wag
        dweights = np.dot(inputs.T, dvalues)
        # Gradienty względem biasów
        dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradienty względem wejść (do przekazania do poprzedniej warstwy)
        dinputs = np.dot(dvalues, self.weights.T)
        # Prosta aktualizacja wag (SGD) z parametrem learning_rate
        self.weights -= self.learning_rate * dweights
        self.biases -= self.learning_rate * dbiases
        return dinputs
