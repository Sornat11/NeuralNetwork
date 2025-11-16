"""
Preprocessing dla Fashion MNIST dataset.

Fashion MNIST zawiera 70,000 obrazów (28x28 pikseli) odzieży w 10 kategoriach:
0: T-shirt/top
1: Trouser
2: Pullover
3: Dress
4: Coat
5: Sandal
6: Shirt
7: Sneaker
8: Bag
9: Ankle boot
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple


def load_fashion_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Wczytuje Fashion MNIST z plików CSV.

    Returns:
        X_train, y_train, X_test, y_test
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Wczytaj dane
    train_df = pd.read_csv(os.path.join(script_dir, "fashion-mnist_train.csv"))
    test_df = pd.read_csv(os.path.join(script_dir, "fashion-mnist_test.csv"))

    # Rozdziel X i y
    X_train = train_df.iloc[:, 1:].values  # Wszystkie kolumny poza 'label'
    y_train = train_df.iloc[:, 0].values   # Kolumna 'label'

    X_test = test_df.iloc[:, 1:].values
    y_test = test_df.iloc[:, 0].values

    print(f"Fashion MNIST loaded:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")

    return X_train, y_train, X_test, y_test


def normalize_images(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalizuje wartości pikseli do zakresu [0, 1].

    Args:
        X_train: Dane treningowe (60000, 784)
        X_test: Dane testowe (10000, 784)

    Returns:
        X_train_norm, X_test_norm
    """
    X_train_norm = X_train.astype('float32') / 255.0
    X_test_norm = X_test.astype('float32') / 255.0

    print(f"\nNormalized to [0, 1]:")
    print(f"  Train min: {X_train_norm.min():.4f}, max: {X_train_norm.max():.4f}")
    print(f"  Test min: {X_test_norm.min():.4f}, max: {X_test_norm.max():.4f}")

    return X_train_norm, X_test_norm


def create_validation_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    val_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Dzieli dane treningowe na train + validation.

    Args:
        X_train: Dane treningowe
        y_train: Etykiety treningowe
        val_ratio: Procent danych na walidację (domyślnie 15%)

    Returns:
        X_train_new, y_train_new, X_val, y_val
    """
    n_samples = len(X_train)
    n_val = int(n_samples * val_ratio)

    # Shuffle z zachowaniem proporcji klas
    from sklearn.model_selection import train_test_split

    X_train_new, X_val, y_train_new, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_ratio,
        stratify=y_train,  # Zachowaj proporcje klas
        random_state=42
    )

    print(f"\nValidation split created:")
    print(f"  Train: {X_train_new.shape[0]} samples")
    print(f"  Val: {X_val.shape[0]} samples")

    return X_train_new, y_train_new, X_val, y_val


def save_splits(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str
):
    """
    Zapisuje przetworzone dane do plików numpy.

    Args:
        X_train, y_train, X_val, y_val, X_test, y_test: Dane
        output_dir: Folder wyjściowy
    """
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    print(f"\nSaved preprocessed data to: {output_dir}")


def get_class_distribution(y: np.ndarray, name: str = "Dataset"):
    """Wyświetla rozkład klas."""
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n{name} class distribution:")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count:5d} samples ({count/len(y)*100:.1f}%)")


if __name__ == "__main__":
    print("="*80)
    print("FASHION MNIST PREPROCESSING")
    print("="*80)

    # 1. Wczytaj dane
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_fashion_mnist()

    # 2. Sprawdź rozkład klas
    get_class_distribution(y_train_raw, "Training set")
    get_class_distribution(y_test_raw, "Test set")

    # 3. Normalizacja
    X_train_norm, X_test_norm = normalize_images(X_train_raw, X_test_raw)

    # 4. Podział train/val/test (85/15 z train, czyli ~70/15/15 ogólnie)
    X_train, y_train, X_val, y_val = create_validation_split(
        X_train_norm,
        y_train_raw,
        val_ratio=0.15
    )

    # 5. Zapisz przetworzone dane
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_splits(
        X_train, y_train,
        X_val, y_val,
        X_test_norm, y_test_raw,
        output_dir=script_dir
    )

    print("\n" + "="*80)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*80)
    print("\nPlik zawiera:")
    print("  - X_train.npy: (51000, 784) - dane treningowe")
    print("  - y_train.npy: (51000,) - etykiety treningowe")
    print("  - X_val.npy: (9000, 784) - dane walidacyjne")
    print("  - y_val.npy: (9000,) - etykiety walidacyjne")
    print("  - X_test.npy: (10000, 784) - dane testowe")
    print("  - y_test.npy: (10000,) - etykiety testowe")
