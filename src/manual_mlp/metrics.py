import numpy as np


class ModelMetrics:
    def __init__(self):
        pass

    def crossentropy_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if y_true.ndim == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        else:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return float(np.mean(negative_log_likelihoods))

    def accuracy(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # Przewidywana klasa to ta z największym prawdopodobieństwem
        predictions = np.argmax(y_pred, axis=1)
        if y_true.ndim != 1:
            y_true = np.argmax(y_true, axis=1)
        return np.mean(predictions == y_true)
