"""
Moduł do ładowania i przygotowywania datasetów.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    split_type: str = "80_20",
    stratify: bool = True,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Dzieli dane na zbiory treningowe, walidacyjne i testowe.

    Args:
        X: Cechy
        y: Etykiety
        split_type: '80_20' lub '70_15_15'
        stratify: Czy zachować proporcje klas (dla klasyfikacji)
        random_state: Seed dla reproducibility

    Returns:
        Dict z kluczami: X_train, X_val, X_test, y_train, y_val, y_test
    """
    stratify_param = y if stratify and len(np.unique(y)) > 1 else None

    if split_type == "80_20":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=stratify_param
        )
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "X_val": None,
            "y_val": None,
        }

    elif split_type == "70_15_15":
        # Najpierw 70% train, 30% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=random_state, stratify=stratify_param
        )

        # Następnie temp dzielimy 50/50 na val i test (czyli 15% każdy)
        stratify_temp = y_temp if stratify and len(np.unique(y_temp)) > 1 else None
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=stratify_temp
        )

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        }
    else:
        raise ValueError(f"Unknown split_type: {split_type}")


def split_time_series(
    X: np.ndarray, y: np.ndarray, split_type: str = "80_20"
) -> Dict[str, np.ndarray]:
    """
    Dzieli szereg czasowy na zbiory (bez shuffling, chronologicznie).

    Args:
        X: Cechy
        y: Wartości
        split_type: '80_20' lub '70_15_15'

    Returns:
        Dict z kluczami: X_train, X_val, X_test, y_train, y_val, y_test
    """
    n = len(X)

    if split_type == "80_20":
        train_size = int(n * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "X_val": None,
            "y_val": None,
        }

    elif split_type == "70_15_15":
        train_size = int(n * 0.7)
        val_size = int(n * 0.15)

        X_train = X[:train_size]
        X_val = X[train_size : train_size + val_size]
        X_test = X[train_size + val_size :]

        y_train = y[:train_size]
        y_val = y[train_size : train_size + val_size]
        y_test = y[train_size + val_size :]

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        }
    else:
        raise ValueError(f"Unknown split_type: {split_type}")


# ==================== DATASETY KLASYFIKACYJNE ====================


def load_iris(split_type: str = "80_20", normalize: bool = True) -> Dict:
    """
    Ładuje dataset Iris.

    Args:
        split_type: '80_20' lub '70_15_15'
        normalize: Czy normalizować cechy

    Returns:
        Dict z danymi i metadanymi
    """
    from sklearn.datasets import load_iris as sklearn_load_iris

    data = sklearn_load_iris()
    X, y = data.data, data.target

    # Normalizacja
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Podział danych
    splits = split_data(X, y, split_type=split_type, stratify=True)

    return {
        **splits,
        "feature_names": data.feature_names,
        "target_names": data.target_names,
        "n_classes": len(data.target_names),
        "n_features": X.shape[1],
    }


def load_wine(split_type: str = "80_20", normalize: bool = True) -> Dict:
    """
    Ładuje dataset Wine.

    Args:
        split_type: '80_20' lub '70_15_15'
        normalize: Czy normalizować cechy

    Returns:
        Dict z danymi i metadanymi
    """
    from sklearn.datasets import load_wine as sklearn_load_wine

    data = sklearn_load_wine()
    X, y = data.data, data.target

    # Normalizacja
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Podział danych
    splits = split_data(X, y, split_type=split_type, stratify=True)

    return {
        **splits,
        "feature_names": data.feature_names,
        "target_names": data.target_names,
        "n_classes": len(data.target_names),
        "n_features": X.shape[1],
    }


def load_fashion_mnist(split_type: str = "80_20", normalize: bool = True) -> Dict:
    """
    Ładuje dataset Fashion MNIST.

    Args:
        split_type: '80_20' lub '70_15_15'
        normalize: Czy normalizować piksele do [0, 1]

    Returns:
        Dict z danymi i metadanymi
    """
    try:
        from tensorflow.keras.datasets import fashion_mnist

        (X_train_full, y_train_full), (X_test_full, y_test_full) = fashion_mnist.load_data()

        # Połącz train i test żeby potem podzielić zgodnie z wymaganiami
        X = np.concatenate([X_train_full, X_test_full], axis=0)
        y = np.concatenate([y_train_full, y_test_full], axis=0)

        # Reshape do (n_samples, 784) dla MLP
        X_flat = X.reshape(X.shape[0], -1)

        # Normalizacja
        if normalize:
            X_flat = X_flat.astype("float32") / 255.0

        # Podział danych
        splits = split_data(X_flat, y, split_type=split_type, stratify=True)

        # Zachowaj również oryginalny kształt (28, 28) dla CNN
        if splits["X_val"] is not None:
            splits_cnn = {
                "X_train_cnn": splits["X_train"].reshape(-1, 28, 28, 1),
                "X_val_cnn": splits["X_val"].reshape(-1, 28, 28, 1),
                "X_test_cnn": splits["X_test"].reshape(-1, 28, 28, 1),
            }
        else:
            splits_cnn = {
                "X_train_cnn": splits["X_train"].reshape(-1, 28, 28, 1),
                "X_test_cnn": splits["X_test"].reshape(-1, 28, 28, 1),
                "X_val_cnn": None,
            }

        class_names = [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]

        return {
            **splits,
            **splits_cnn,
            "target_names": class_names,
            "n_classes": 10,
            "n_features": 784,
            "image_shape": (28, 28),
        }
    except Exception as e:
        print(f"Error loading Fashion MNIST: {e}")
        print("Tworzę syntetyczne dane jako fallback...")
        # Fallback - syntetyczne dane
        X = np.random.randn(1000, 784)
        y = np.random.randint(0, 10, 1000)
        splits = split_data(X, y, split_type=split_type, stratify=True)
        return {**splits, "n_classes": 10, "n_features": 784}


# ==================== DATASETY REGRESYJNE ====================


def load_airline_passengers(
    split_type: str = "80_20", normalize: bool = True, lookback: int = 12
) -> Dict:
    """
    Ładuje dataset Airline Passengers (szereg czasowy).

    Args:
        split_type: '80_20' lub '70_15_15'
        normalize: Czy normalizować wartości
        lookback: Ile poprzednich wartości używać do predykcji

    Returns:
        Dict z danymi i metadanymi
    """
    # Dane Airline Passengers (liczba pasażerów miesięcznie 1949-1960)
    data = np.array(
        [
            112,
            118,
            132,
            129,
            121,
            135,
            148,
            148,
            136,
            119,
            104,
            118,
            115,
            126,
            141,
            135,
            125,
            149,
            170,
            170,
            158,
            133,
            114,
            140,
            145,
            150,
            178,
            163,
            172,
            178,
            199,
            199,
            184,
            162,
            146,
            166,
            171,
            180,
            193,
            181,
            183,
            218,
            230,
            242,
            209,
            191,
            172,
            194,
            196,
            196,
            236,
            235,
            229,
            243,
            264,
            272,
            237,
            211,
            180,
            201,
            204,
            188,
            235,
            227,
            234,
            264,
            302,
            293,
            259,
            229,
            203,
            229,
            242,
            233,
            267,
            269,
            270,
            315,
            364,
            347,
            312,
            274,
            237,
            278,
            284,
            277,
            317,
            313,
            318,
            374,
            413,
            405,
            355,
            306,
            271,
            306,
            315,
            301,
            356,
            348,
            355,
            422,
            465,
            467,
            404,
            347,
            305,
            336,
            340,
            318,
            362,
            348,
            363,
            435,
            491,
            505,
            404,
            359,
            310,
            337,
            360,
            342,
            406,
            396,
            420,
            472,
            548,
            559,
            463,
            407,
            362,
            405,
            417,
            391,
            419,
            461,
            472,
            535,
            622,
            606,
            508,
            461,
            390,
            432,
        ]
    )

    # Normalizacja
    if normalize:
        data_min, data_max = data.min(), data.max()
        data_normalized = (data - data_min) / (data_max - data_min)
    else:
        data_normalized = data.copy()

    # Tworzenie sekwencji (sliding window)
    X, y = [], []
    for i in range(len(data_normalized) - lookback):
        X.append(data_normalized[i : i + lookback])
        y.append(data_normalized[i + lookback])

    X = np.array(X)
    y = np.array(y)

    # Podział chronologiczny (szereg czasowy)
    splits = split_time_series(X, y, split_type=split_type)

    # Reshape dla RNN/LSTM (samples, timesteps, features)
    splits_rnn = {}
    for key in ["X_train", "X_val", "X_test"]:
        if splits[key] is not None:
            splits_rnn[f"{key}_rnn"] = splits[key].reshape(splits[key].shape[0], lookback, 1)
        else:
            splits_rnn[f"{key}_rnn"] = None

    return {
        **splits,
        **splits_rnn,
        "lookback": lookback,
        "n_features": lookback,
        "original_data": data,
        "data_min": data_min if normalize else None,
        "data_max": data_max if normalize else None,
    }


def load_synthetic_regression(
    n_samples: int = 1000,
    n_features: int = 10,
    noise: float = 0.1,
    split_type: str = "80_20",
    normalize: bool = True,
) -> Dict:
    """
    Tworzy syntetyczny dataset do regresji.

    Args:
        n_samples: Liczba próbek
        n_features: Liczba cech
        noise: Poziom szumu
        split_type: '80_20' lub '70_15_15'
        normalize: Czy normalizować cechy

    Returns:
        Dict z danymi i metadanymi
    """
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=42)

    # Normalizacja
    if normalize:
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X = scaler_X.fit_transform(X)
        y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    # Podział danych
    splits = split_data(X, y, split_type=split_type, stratify=False)

    return {
        **splits,
        "n_features": n_features,
        "n_samples": n_samples,
    }


# ==================== FUNKCJE POMOCNICZE ====================


def get_dataset(dataset_name: str, split_type: str = "80_20", **kwargs) -> Dict:
    """
    Uniwersalna funkcja do ładowania datasetów.

    Args:
        dataset_name: Nazwa datasetu ('iris', 'wine', 'fashion_mnist', 'airline', 'synthetic_reg')
        split_type: '80_20' lub '70_15_15'
        **kwargs: Dodatkowe argumenty dla konkretnego datasetu

    Returns:
        Dict z danymi
    """
    datasets = {
        "iris": load_iris,
        "wine": load_wine,
        "fashion_mnist": load_fashion_mnist,
        "airline": load_airline_passengers,
        "synthetic_reg": load_synthetic_regression,
    }

    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(datasets.keys())}")

    return datasets[dataset_name](split_type=split_type, **kwargs)
