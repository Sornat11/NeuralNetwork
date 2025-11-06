"""
Metryki ewaluacji dla klasyfikacji i regresji.
"""

import numpy as np
from typing import Dict, Optional


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Oblicza accuracy dla klasyfikacji.

    Args:
        y_true: Prawdziwe etykiety
        y_pred: Predykcje (prawdopodobieństwa lub klasy)

    Returns:
        Accuracy (0-1)
    """
    # Jeśli y_pred to prawdopodobieństwa, weź argmax
    if y_pred.ndim > 1:
        predictions = np.argmax(y_pred, axis=1)
    else:
        predictions = y_pred

    # Jeśli y_true to one-hot, weź argmax
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)

    return float(np.mean(predictions == y_true))


def precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
    """
    Oblicza precision dla klasyfikacji.

    Args:
        y_true: Prawdziwe etykiety
        y_pred: Predykcje
        average: 'macro', 'micro', lub 'weighted'

    Returns:
        Precision (0-1)
    """
    # Konwersja do klas
    if y_pred.ndim > 1:
        predictions = np.argmax(y_pred, axis=1)
    else:
        predictions = y_pred

    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)

    classes = np.unique(y_true)
    precisions = []

    for cls in classes:
        true_positives = np.sum((predictions == cls) & (y_true == cls))
        predicted_positives = np.sum(predictions == cls)

        if predicted_positives == 0:
            precisions.append(0.0)
        else:
            precisions.append(true_positives / predicted_positives)

    if average == "macro":
        return float(np.mean(precisions))
    elif average == "micro":
        # Micro averaging
        tp = np.sum(predictions == y_true)
        return float(tp / len(y_true))
    elif average == "weighted":
        # Weighted by support
        weights = [np.sum(y_true == cls) for cls in classes]
        return float(np.average(precisions, weights=weights))
    else:
        raise ValueError(f"Unknown average: {average}")


def recall(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
    """
    Oblicza recall dla klasyfikacji.

    Args:
        y_true: Prawdziwe etykiety
        y_pred: Predykcje
        average: 'macro', 'micro', lub 'weighted'

    Returns:
        Recall (0-1)
    """
    # Konwersja do klas
    if y_pred.ndim > 1:
        predictions = np.argmax(y_pred, axis=1)
    else:
        predictions = y_pred

    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)

    classes = np.unique(y_true)
    recalls = []

    for cls in classes:
        true_positives = np.sum((predictions == cls) & (y_true == cls))
        actual_positives = np.sum(y_true == cls)

        if actual_positives == 0:
            recalls.append(0.0)
        else:
            recalls.append(true_positives / actual_positives)

    if average == "macro":
        return float(np.mean(recalls))
    elif average == "micro":
        # Micro averaging
        tp = np.sum(predictions == y_true)
        return float(tp / len(y_true))
    elif average == "weighted":
        # Weighted by support
        weights = [np.sum(y_true == cls) for cls in classes]
        return float(np.average(recalls, weights=weights))
    else:
        raise ValueError(f"Unknown average: {average}")


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
    """
    Oblicza F1 score dla klasyfikacji.

    Args:
        y_true: Prawdziwe etykiety
        y_pred: Predykcje
        average: 'macro', 'micro', lub 'weighted'

    Returns:
        F1 score (0-1)
    """
    prec = precision(y_true, y_pred, average=average)
    rec = recall(y_true, y_pred, average=average)

    if prec + rec == 0:
        return 0.0

    return float(2 * (prec * rec) / (prec + rec))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Oblicza macierz pomyłek (confusion matrix).

    Args:
        y_true: Prawdziwe etykiety
        y_pred: Predykcje

    Returns:
        Macierz pomyłek (n_classes x n_classes)
    """
    # Konwersja do klas
    if y_pred.ndim > 1:
        predictions = np.argmax(y_pred, axis=1)
    else:
        predictions = y_pred

    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)

    classes = np.unique(np.concatenate([y_true, predictions]))
    n_classes = len(classes)

    cm = np.zeros((n_classes, n_classes), dtype=int)

    for true_cls, pred_cls in zip(y_true, predictions):
        cm[int(true_cls), int(pred_cls)] += 1

    return cm


# ==================== METRYKI REGRESJI ====================


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Oblicza Mean Squared Error (MSE).

    Args:
        y_true: Prawdziwe wartości
        y_pred: Predykcje

    Returns:
        MSE
    """
    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Oblicza Mean Absolute Error (MAE).

    Args:
        y_true: Prawdziwe wartości
        y_pred: Predykcje

    Returns:
        MAE
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Oblicza Root Mean Squared Error (RMSE).

    Args:
        y_true: Prawdziwe wartości
        y_pred: Predykcje

    Returns:
        RMSE
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Oblicza R² (coefficient of determination).

    Args:
        y_true: Prawdziwe wartości
        y_pred: Predykcje

    Returns:
        R² score
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return float(1 - (ss_res / ss_tot))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Oblicza Mean Absolute Percentage Error (MAPE).

    Args:
        y_true: Prawdziwe wartości
        y_pred: Predykcje

    Returns:
        MAPE (w procentach)
    """
    # Unikaj dzielenia przez zero
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ==================== FUNKCJE POMOCNICZE ====================


def evaluate_classification(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro"
) -> Dict[str, float]:
    """
    Oblicza wszystkie metryki klasyfikacji.

    Args:
        y_true: Prawdziwe etykiety
        y_pred: Predykcje
        average: Typ uśredniania

    Returns:
        Dict z metrykami
    """
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred, average=average),
        "recall": recall(y_true, y_pred, average=average),
        "f1_score": f1_score(y_true, y_pred, average=average),
    }


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Oblicza wszystkie metryki regresji.

    Args:
        y_true: Prawdziwe wartości
        y_pred: Predykcje

    Returns:
        Dict z metrykami
    """
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
