import numpy as np

from utils.seed import set_seed

set_seed(0)


class LayerDense:

    def __init__(self, n_inputs: int, n_neurons: int, weight_scale: float = 0.1):
        self.weights = weight_scale * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output: np.ndarray | None = None
        self.inputs: np.ndarray | None = None

        # Gradienty
        self.dweights: np.ndarray | None = None
        self.dbiases: np.ndarray | None = None
        self.dinputs: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, dvalues: np.ndarray) -> None:
        """
        Backpropagation przez warstwę Dense.

        Args:
            dvalues: Gradient z następnej warstwy (dL/doutput)
        """
        # Gradienty dla wag
        self.dweights = np.dot(self.inputs.T, dvalues)

        # Gradienty dla biasów
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradienty dla wejść (propagowane do poprzedniej warstwy)
        self.dinputs = np.dot(dvalues, self.weights.T)
