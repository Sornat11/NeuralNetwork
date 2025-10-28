import numpy as np

from utils.seed import set_seed

set_seed(0)


class LayerDense:

    def __init__(self, n_inputs: int, n_neurons: int, weight_scale: float = 0.1):
        self.weights = weight_scale * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output
