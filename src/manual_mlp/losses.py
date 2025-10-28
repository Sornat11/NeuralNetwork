import numpy as np

class Loss:
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
