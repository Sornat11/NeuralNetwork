"""
Implementacja CNN 1D dla regresji (szeregi czasowe).
Używa Conv1D do analizy wzorców czasowych.
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from typing import Dict, Tuple


class KerasCNN1DRegression:
    """
    Model CNN 1D dla regresji na szeregach czasowych.

    Architektura:
    - Conv1D → ReLU → MaxPooling (x N)
    - Flatten
    - Dense → ReLU (x M)
    - Dense → linear (output)
    """

    def __init__(
        self,
        n_features: int,
        n_conv_layers: int = 2,
        n_filters: int = 64,
        kernel_size: int = 3,
        pool_size: int = 2,
        n_dense_neurons: int = 64,
        learning_rate: float = 0.001,
        optimizer_name: str = "adam"
    ):
        """
        Args:
            n_features: Liczba cech wejściowych
            n_conv_layers: Liczba warstw Conv1D
            n_filters: Liczba filtrów w Conv1D
            kernel_size: Rozmiar kernela
            pool_size: Rozmiar poolingu
            n_dense_neurons: Liczba neuronów w Dense
            learning_rate: Learning rate
            optimizer_name: Nazwa optymalizatora
        """
        self.n_features = n_features
        self.n_conv_layers = n_conv_layers
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.n_dense_neurons = n_dense_neurons
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name

        self.model = self._build_model()
        self.history = None

    def _build_model(self) -> keras.Model:
        """Buduje model CNN 1D."""
        model = keras.Sequential()

        # Input: (n_samples, n_features) → reshape to (n_samples, n_features, 1)
        model.add(layers.Input(shape=(self.n_features,)))
        model.add(layers.Reshape((self.n_features, 1)))

        # Warstwy Conv1D + MaxPooling (ale tylko jeśli rozmiar pozwala)
        current_size = self.n_features
        for i in range(self.n_conv_layers):
            model.add(layers.Conv1D(
                filters=self.n_filters,
                kernel_size=self.kernel_size,
                activation='relu',
                padding='same',
                name=f'conv1d_{i+1}'
            ))

            # Dodaj pooling tylko jeśli rozmiar > 4 (żeby nie zredukować za bardzo)
            if current_size > 4:
                model.add(layers.MaxPooling1D(
                    pool_size=self.pool_size,
                    name=f'pool_{i+1}'
                ))
                current_size = current_size // self.pool_size
            # Jeśli za małe, pomiń pooling i zostaw tylko Conv

        # Flatten
        model.add(layers.Flatten())

        # Dense layer
        model.add(layers.Dense(
            self.n_dense_neurons,
            activation='relu',
            name='dense'
        ))

        # Output (regresja - linear activation)
        model.add(layers.Dense(1, activation='linear', name='output'))

        # Wybór optymalizatora
        if self.optimizer_name.lower() == "sgd":
            optimizer = optimizers.SGD(learning_rate=self.learning_rate)
        elif self.optimizer_name.lower() == "adam":
            optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer_name.lower() == "rmsprop":
            optimizer = optimizers.RMSprop(learning_rate=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )

        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 0
    ) -> Dict:
        """Trenuje model."""
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )

        return self.history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predykcja."""
        return self.model.predict(X, verbose=0)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Ewaluacja (MSE, MAE)."""
        results = self.model.evaluate(X, y, verbose=0)
        return results[0], results[1]

    def get_summary(self) -> None:
        """Podsumowanie architektury."""
        self.model.summary()
