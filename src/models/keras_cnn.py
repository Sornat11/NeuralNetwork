"""
Implementacja CNN (Convolutional Neural Network) dla Fashion MNIST.
Używa TensorFlow/Keras.
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from typing import Dict, Tuple, List


class KerasCNNModel:
    """
    Model CNN zbudowany w Keras dla klasyfikacji obrazów.

    Architektura:
    - Conv2D → ReLU → MaxPooling (x N)
    - Flatten
    - Dense → ReLU (x M)
    - Dense → Softmax (output)
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (28, 28, 1),  # (height, width, channels)
        n_conv_layers: int = 2,
        n_filters: List[int] = None,  # Liczba filtrów w każdej warstwie conv
        kernel_size: int = 3,
        pool_size: int = 2,
        n_dense_layers: int = 1,
        n_dense_neurons: int = 128,
        n_outputs: int = 10,
        learning_rate: float = 0.001,
        optimizer_name: str = "adam"  # "sgd", "adam", "rmsprop"
    ):
        """
        Args:
            input_shape: Kształt obrazu wejściowego (H, W, C)
            n_conv_layers: Liczba warstw konwolucyjnych
            n_filters: Lista liczby filtrów dla każdej warstwy conv
            kernel_size: Rozmiar kernela (filtru)
            pool_size: Rozmiar poolingu
            n_dense_layers: Liczba warstw Dense (fully connected)
            n_dense_neurons: Liczba neuronów w warstwach Dense
            n_outputs: Liczba klas
            learning_rate: Learning rate
            optimizer_name: Nazwa optymalizatora ("sgd", "adam", "rmsprop")
        """
        self.input_shape = input_shape
        self.n_conv_layers = n_conv_layers
        self.n_filters = n_filters if n_filters else [32 * (2**i) for i in range(n_conv_layers)]
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.n_dense_layers = n_dense_layers
        self.n_dense_neurons = n_dense_neurons
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name

        # Budujemy model
        self.model = self._build_model()

        # Historia treningu
        self.history = None

    def _build_model(self) -> keras.Model:
        """
        Buduje sekwencyjny model CNN.

        Returns:
            Skompilowany model Keras
        """
        model = keras.Sequential()

        # Warstwa wejściowa
        model.add(layers.Input(shape=self.input_shape))

        # Warstwy konwolucyjne + pooling
        for i in range(self.n_conv_layers):
            model.add(layers.Conv2D(
                filters=self.n_filters[i],
                kernel_size=self.kernel_size,
                activation='relu',
                padding='same',
                kernel_initializer='glorot_uniform',
                name=f'conv_{i+1}'
            ))
            model.add(layers.MaxPooling2D(
                pool_size=self.pool_size,
                name=f'pool_{i+1}'
            ))

        # Flatten (spłaszczenie do wektora 1D)
        model.add(layers.Flatten())

        # Warstwy Dense (fully connected)
        for i in range(self.n_dense_layers):
            model.add(layers.Dense(
                self.n_dense_neurons,
                activation='relu',
                kernel_initializer='glorot_uniform',
                name=f'dense_{i+1}'
            ))

        # Warstwa wyjściowa (Softmax)
        model.add(layers.Dense(
            self.n_outputs,
            activation='softmax',
            kernel_initializer='glorot_uniform',
            name='output'
        ))

        # Wybór optymalizatora
        if self.optimizer_name.lower() == "sgd":
            optimizer = optimizers.SGD(learning_rate=self.learning_rate)
        elif self.optimizer_name.lower() == "adam":
            optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer_name.lower() == "rmsprop":
            optimizer = optimizers.RMSprop(learning_rate=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        # Kompilacja
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 20,
        batch_size: int = 128,
        verbose: int = 0
    ) -> Dict:
        """
        Trenuje model.

        Args:
            X_train: Dane treningowe (N, H, W, C) lub (N, H*W) - automatycznie reshape
            y_train: Etykiety treningowe
            X_val: Dane walidacyjne (opcjonalne)
            y_val: Etykiety walidacyjne (opcjonalne)
            epochs: Liczba epok
            batch_size: Rozmiar batcha
            verbose: Poziom logowania

        Returns:
            Historia treningu
        """
        # Reshape jeśli potrzeba (z (N, 784) na (N, 28, 28, 1))
        X_train = self._reshape_if_needed(X_train)
        if X_val is not None:
            X_val = self._reshape_if_needed(X_val)

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

    def _reshape_if_needed(self, X: np.ndarray) -> np.ndarray:
        """Reshape z (N, 784) na (N, 28, 28, 1) jeśli potrzeba."""
        if X.ndim == 2:
            # Zakładamy obrazy 28x28
            n_samples = X.shape[0]
            h, w, c = self.input_shape
            return X.reshape(n_samples, h, w, c)
        return X

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predykcja na danych.

        Args:
            X: Dane wejściowe (N, H, W, C) lub (N, H*W)

        Returns:
            Prawdopodobieństwa klas (N, n_classes)
        """
        X = self._reshape_if_needed(X)
        return self.model.predict(X, verbose=0)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, float]:
        """
        Ewaluacja modelu.

        Args:
            X: Dane (N, H, W, C) lub (N, H*W)
            y: Prawdziwe etykiety

        Returns:
            (loss, accuracy)
        """
        X = self._reshape_if_needed(X)
        results = self.model.evaluate(X, y, verbose=0)
        return results[0], results[1]  # loss, accuracy

    def get_summary(self) -> None:
        """Wyświetla podsumowanie architektury modelu."""
        self.model.summary()

    def save(self, filepath: str) -> None:
        """Zapisuje model do pliku."""
        self.model.save(filepath)

    @classmethod
    def load(cls, filepath: str) -> 'KerasCNNModel':
        """Wczytuje model z pliku."""
        loaded_model = keras.models.load_model(filepath)
        instance = cls.__new__(cls)
        instance.model = loaded_model
        return instance


if __name__ == "__main__":
    """Test modelu CNN."""
    print("="*80)
    print("TEST CNN MODEL")
    print("="*80)

    # Stwórz model
    model = KerasCNNModel(
        input_shape=(28, 28, 1),
        n_conv_layers=2,
        n_filters=[32, 64],
        n_dense_layers=1,
        n_dense_neurons=128,
        n_outputs=10,
        learning_rate=0.001,
        optimizer_name="adam"
    )

    print("\n📐 Architektura modelu:")
    model.get_summary()

    # Test na losowych danych
    print("\n🧪 Test na losowych danych:")
    X_dummy = np.random.rand(100, 784).astype('float32')
    y_dummy = np.random.randint(0, 10, size=100)

    print("  Trenowanie (3 epoki)...")
    history = model.train(X_dummy, y_dummy, epochs=3, verbose=1)

    print(f"\n  Loss po 3 epokach: {history['loss'][-1]:.4f}")
    print(f"  Accuracy po 3 epokach: {history['accuracy'][-1]:.4f}")

    # Test predykcji
    print("\n🔮 Test predykcji:")
    preds = model.predict(X_dummy[:5])
    print(f"  Shape predykcji: {preds.shape}")
    print(f"  Przewidziane klasy: {np.argmax(preds, axis=1)}")

    print("\n✅ Test zakończony pomyślnie!")
