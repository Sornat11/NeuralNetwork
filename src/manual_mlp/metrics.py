import numpy as np


class ModelMetrics:
    def mse(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Mean Squared Error (MSE)
        MSE = mean((y_pred - y_true)^2)
        """
        return float(np.mean((y_pred - y_true) ** 2))

    def mae(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Mean Absolute Error (MAE)
        MAE = mean(|y_pred - y_true|)
        """
        return float(np.mean(np.abs(y_pred - y_true)))

    def r2_score(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        R^2 (coefficient of determination)
        R^2 = 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    def __init__(self):
        pass

    def crossentropy_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Categorical Crossentropy Loss
        L = -mean(log(p_true_class))
        """
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if y_true.ndim == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        else:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return float(np.mean(negative_log_likelihoods))

    def accuracy(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Accuracy (classification)
        accuracy = (number of correct predictions) / (total predictions)
        """
        predictions = np.argmax(y_pred, axis=1)
        if y_true.ndim != 1:
            y_true = np.argmax(y_true, axis=1)
        return np.mean(predictions == y_true)

    def precision(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Precision (classification, macro)
        Precision = TP / (TP + FP) for each class, macro-average (mean over classes)
        """
        predictions = np.argmax(y_pred, axis=1)
        if y_true.ndim != 1:
            y_true = np.argmax(y_true, axis=1)
        num_classes = np.max(y_true) + 1
        precisions = []
        for c in range(num_classes):
            tp = np.sum((predictions == c) & (y_true == c))
            fp = np.sum((predictions == c) & (y_true != c))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            precisions.append(precision)
        return float(np.mean(precisions))

    def recall(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Recall (classification, macro)
        Recall = TP / (TP + FN) for each class, macro-average (mean over classes)
        """
        predictions = np.argmax(y_pred, axis=1)
        if y_true.ndim != 1:
            y_true = np.argmax(y_true, axis=1)
        num_classes = np.max(y_true) + 1
        recalls = []
        for c in range(num_classes):
            tp = np.sum((predictions == c) & (y_true == c))
            fn = np.sum((predictions != c) & (y_true == c))
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recalls.append(recall)
        return float(np.mean(recalls))

    def f1_score(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        F1-score (classification, macro)
        F1 = 2 * (precision * recall) / (precision + recall), macro-average
        """
        prec = self.precision(y_pred, y_true)
        rec = self.recall(y_pred, y_true)
        if (prec + rec) == 0:
            return 0.0
        return 2 * (prec * rec) / (prec + rec)
