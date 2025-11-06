import numpy as np

class ActivationReLU:
    def __init__(self):
        self.output: np.ndarray | None = None
        self.inputs: np.ndarray | None = None
        self.dinputs: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, dvalues: np.ndarray) -> None:
        """
        Backpropagation przez ReLU.
        Gradient = 1 gdzie inputs > 0, inaczej 0
        """
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class ActivationSoftmax:
    def __init__(self):
        self.output: np.ndarray | None = None
        self.dinputs: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output

    def backward(self, dvalues: np.ndarray) -> None:
        """
        Backpropagation przez Softmax.
        Używa pełnej macierzy Jacobian.
        """
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class ActivationSigmoid:
    """Aktywacja sigmoid dla regresji binarnej lub wartości [0,1]"""

    def __init__(self):
        self.output: np.ndarray | None = None
        self.dinputs: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output

    def backward(self, dvalues: np.ndarray) -> None:
        """
        Backpropagation przez Sigmoid.
        Gradient = output * (1 - output)
        """
        self.dinputs = dvalues * self.output * (1 - self.output)


class ActivationLinear:
    """Aktywacja liniowa dla regresji"""

    def __init__(self):
        self.output: np.ndarray | None = None
        self.dinputs: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.output = inputs
        return self.output

    def backward(self, dvalues: np.ndarray) -> None:
        """
        Backpropagation przez Linear.
        Gradient = 1 (przepuszcza gradient bez zmian)
        """
        self.dinputs = dvalues.copy()