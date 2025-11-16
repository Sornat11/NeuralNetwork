"""
Implementacja MLP przy użyciu TensorFlow/Keras.
Architektura analogiczna do ręcznej implementacji z src/manual_mlp/model.py
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from typing import Dict, Tuple


class KerasMLPModel:
    """
    Model MLP zbudowany w Keras, odwzorowujący architekturę z manual_mlp.

    Architektura:
    - N warstw ukrytych (Dense + ReLU)
    - Warstwa wyjściowa:
        * Classification: Dense(n_outputs) + Softmax
        * Regression: Dense(1), linear activation
    """

    def __init__(
        self,
        n_inputs: int,
        n_hidden_layers: int,
        n_neurons: int,
        n_outputs: int,
        learning_rate: float = 0.01,
        task_type: str = "classification",  # "classification" lub "regression"
        optimizer_name: str = "sgd"  # "sgd", "adam", "rmsprop"
    ):
        """
        Args:
            n_inputs: Liczba cech wejściowych
            n_hidden_layers: Liczba warstw ukrytych
            n_neurons: Liczba neuronów w każdej warstwie ukrytej
            n_outputs: Liczba klas (classification) lub 1 (regression)
            learning_rate: Learning rate dla SGD
            task_type: "classification" lub "regression"
        """
        self.n_inputs = n_inputs
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate
        self.task_type = task_type
        self.optimizer_name = optimizer_name

        # Budujemy model
        self.model = self._build_model()

        # Historia treningu
        self.history = None

    def _build_model(self) -> keras.Model:
        """
        Buduje sekwencyjny model Keras.

        Returns:
            Skompilowany model Keras
        """
        model = keras.Sequential()

        # Warstwa wejściowa (Input layer)
        model.add(layers.Input(shape=(self.n_inputs,)))

        # Warstwy ukryte z ReLU (analogicznie do manual_mlp)
        for i in range(self.n_hidden_layers):
            model.add(layers.Dense(
                self.n_neurons,
                activation='relu',
                kernel_initializer='glorot_uniform',  # Xavier initialization (jak w manual)
                name=f'hidden_{i+1}'
            ))

        # Warstwa wyjściowa
        if self.task_type == "classification":
            # Softmax dla klasyfikacji
            model.add(layers.Dense(
                self.n_outputs,
                activation='softmax',
                kernel_initializer='glorot_uniform',
                name='output'
            ))

            # Wybór optymalizatora
            optimizer = self._get_optimizer()

            # Kompilacja dla klasyfikacji
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',  # Cross-entropy jak w manual
                metrics=['accuracy']
            )
        else:
            # Linear dla regresji
            model.add(layers.Dense(
                1,
                activation='linear',
                kernel_initializer='glorot_uniform',
                name='output'
            ))

            # Wybór optymalizatora
            optimizer = self._get_optimizer()

            # Kompilacja dla regresji
            model.compile(
                optimizer=optimizer,
                loss='mse',  # Mean Squared Error
                metrics=['mae']  # Mean Absolute Error
            )

        return model

    def _get_optimizer(self):
        """
        Wybiera optymalizator na podstawie nazwy.

        Returns:
            Keras optimizer
        """
        if self.optimizer_name.lower() == "sgd":
            return optimizers.SGD(learning_rate=self.learning_rate)
        elif self.optimizer_name.lower() == "adam":
            return optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer_name.lower() == "rmsprop":
            return optimizers.RMSprop(learning_rate=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

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
        """
        Trenuje model.

        Args:
            X_train: Dane treningowe
            y_train: Etykiety treningowe
            X_val: Dane walidacyjne (opcjonalne)
            y_val: Etykiety walidacyjne (opcjonalne)
            epochs: Liczba epok
            batch_size: Rozmiar batcha
            verbose: Poziom logowania (0=cicho, 1=progress bar, 2=jedna linia na epokę)

        Returns:
            Historia treningu (loss, metrics przez epoki)
        """
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
        """
        Predykcja na danych.

        Args:
            X: Dane wejściowe

        Returns:
            Dla classification: prawdopodobieństwa klas (shape: [n_samples, n_classes])
            Dla regression: wartości ciągłe (shape: [n_samples, 1])
        """
        return self.model.predict(X, verbose=0)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, float]:
        """
        Ewaluacja modelu.

        Args:
            X: Dane
            y: Prawdziwe etykiety

        Returns:
            (loss, metric) - loss i accuracy/mae w zależności od task_type
        """
        results = self.model.evaluate(X, y, verbose=0)
        return results[0], results[1]  # loss, metric

    def get_summary(self) -> None:
        """Wyświetla podsumowanie architektury modelu."""
        self.model.summary()

    def save(self, filepath: str) -> None:
        """Zapisuje model do pliku."""
        self.model.save(filepath)

    @classmethod
    def load(cls, filepath: str) -> 'KerasMLPModel':
        """Wczytuje model z pliku."""
        loaded_model = keras.models.load_model(filepath)
        # Tworzymy wrapper (ale bez reinicjalizacji modelu)
        instance = cls.__new__(cls)
        instance.model = loaded_model
        return instance
