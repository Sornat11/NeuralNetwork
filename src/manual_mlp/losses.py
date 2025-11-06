import numpy as np

class Loss:
    def __init__(self):
        self.dinputs: np.ndarray | None = None

    def calculate(self, output: np.ndarray, y: np.ndarray) -> float:
        sample_losses = self.forward(output, y)
        return float(np.mean(sample_losses))


class LossCategoricalCrossentropy(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if y_true.ndim == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        else:  # one-hot
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues: np.ndarray, y_true: np.ndarray) -> None:
        """
        Backpropagation przez Categorical Crossentropy.

        Args:
            dvalues: Predykcje modelu (y_pred)
            y_true: Prawdziwe etykiety
        """
        samples = len(dvalues)
        labels = len(dvalues[0])

        # Konwersja sparse labels na one-hot
        if y_true.ndim == 1:
            y_true_one_hot = np.eye(labels)[y_true]
        else:
            y_true_one_hot = y_true

        # Gradient
        self.dinputs = -y_true_one_hot / dvalues
        # Normalizacja przez liczbę sampli
        self.dinputs = self.dinputs / samples


class LossMeanSquaredError(Loss):
    """Mean Squared Error dla regresji"""

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Oblicza MSE dla każdego sampla.
        """
        return np.mean((y_true - y_pred) ** 2, axis=-1)

    def backward(self, dvalues: np.ndarray, y_true: np.ndarray) -> None:
        """
        Backpropagation przez MSE.

        Args:
            dvalues: Predykcje modelu (y_pred)
            y_true: Prawdziwe wartości
        """
        samples = len(dvalues)
        outputs = dvalues.shape[1] if dvalues.ndim > 1 else 1

        # Gradient MSE: 2 * (y_pred - y_true) / n
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # Normalizacja przez liczbę sampli
        self.dinputs = self.dinputs / samples


class LossMeanAbsoluteError(Loss):
    """Mean Absolute Error dla regresji"""

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Oblicza MAE dla każdego sampla.
        """
        return np.mean(np.abs(y_true - y_pred), axis=-1)

    def backward(self, dvalues: np.ndarray, y_true: np.ndarray) -> None:
        """
        Backpropagation przez MAE.

        Args:
            dvalues: Predykcje modelu (y_pred)
            y_true: Prawdziwe wartości
        """
        samples = len(dvalues)
        outputs = dvalues.shape[1] if dvalues.ndim > 1 else 1

        # Gradient MAE: sign(y_pred - y_true)
        self.dinputs = np.sign(dvalues - y_true) / outputs
        # Normalizacja przez liczbę sampli
        self.dinputs = self.dinputs / samples


class SoftmaxCategoricalCrossentropy:
    """
    Zoptymalizowana kombinacja Softmax + Categorical Crossentropy.
    Używana w klasyfikacji wieloklasowej.
    """

    def __init__(self):
        self.dinputs: np.ndarray | None = None
        self.output: np.ndarray | None = None

    def forward(self, inputs: np.ndarray, y_true: np.ndarray) -> float:
        """
        Forward pass przez Softmax + CCE.

        Args:
            inputs: Wyjście z ostatniej warstwy (logits)
            y_true: Prawdziwe etykiety
        """
        # Softmax
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

        # Categorical Crossentropy
        samples = len(probabilities)
        y_pred_clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)

        if y_true.ndim == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        else:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return float(np.mean(negative_log_likelihoods))

    def backward(self, dvalues: np.ndarray, y_true: np.ndarray) -> None:
        """
        Zoptymalizowany backward pass przez Softmax + CCE.
        Gradient = y_pred - y_true

        Args:
            dvalues: Wyjście z softmax (prawdopodobieństwa)
            y_true: Prawdziwe etykiety
        """
        samples = len(dvalues)

        # Konwersja sparse labels na one-hot jeśli potrzeba
        if y_true.ndim == 1:
            y_true_one_hot = np.eye(dvalues.shape[1])[y_true]
        else:
            y_true_one_hot = y_true

        # Gradient: softmax_output - y_true
        self.dinputs = dvalues - y_true_one_hot
        # Normalizacja przez liczbę sampli
        self.dinputs = self.dinputs / samples
