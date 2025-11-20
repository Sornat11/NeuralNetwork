"""
Implementacja LSTM (Long Short-Term Memory) dla regresji (szeregi czasowe).
Sieć rekurencyjna do przetwarzania sekwencji czasowych.
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from typing import Dict, Tuple


class KerasLSTMRegression:
    """
    Model LSTM dla regresji na szeregach czasowych.

    Architektura:
    - LSTM → LSTM (x N)
    - Dense → ReLU
    - Dense → linear (output)
    """

    def __init__(
        self,
        n_features: int,
        n_lstm_layers: int = 2,
        n_lstm_units: int = 64,
        n_dense_neurons: int = 32,
        learning_rate: float = 0.001,
        optimizer_name: str = "adam",
        dropout: float = 0.0
    ):
        """
        Args:
            n_features: Liczba cech wejściowych
            n_lstm_layers: Liczba warstw LSTM
            n_lstm_units: Liczba jednostek w każdej warstwie LSTM
            n_dense_neurons: Liczba neuronów w warstwie Dense
            learning_rate: Learning rate
            optimizer_name: Nazwa optymalizatora
            dropout: Dropout rate (0.0 = bez dropout)
        """
        self.n_features = n_features
        self.n_lstm_layers = n_lstm_layers
        self.n_lstm_units = n_lstm_units
        self.n_dense_neurons = n_dense_neurons
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.dropout = dropout

        self.model = self._build_model()
        self.history = None

    def _build_model(self) -> keras.Model:
        """Buduje model LSTM."""
        model = keras.Sequential()

        # Input: (n_samples, n_features) → reshape to (n_samples, n_features, 1)
        # LSTM wymaga kształtu: (batch, timesteps, features)
        # Tutaj traktujemy każdą cechę jako timestep
        model.add(layers.Input(shape=(self.n_features,)))
        model.add(layers.Reshape((self.n_features, 1)))

        # Warstwy LSTM
        for i in range(self.n_lstm_layers):
            # return_sequences=True dla wszystkich warstw oprócz ostatniej
            return_sequences = (i < self.n_lstm_layers - 1)

            model.add(layers.LSTM(
                units=self.n_lstm_units,
                return_sequences=return_sequences,
                activation='tanh',
                recurrent_activation='sigmoid',
                kernel_initializer='glorot_uniform',
                name=f'lstm_{i+1}'
            ))

            # Dropout (jeśli włączony)
            if self.dropout > 0:
                model.add(layers.Dropout(self.dropout, name=f'dropout_{i+1}'))

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


if __name__ == "__main__":
    """Test modelu LSTM."""
    print("="*80)
    print("TEST LSTM MODEL")
    print("="*80)

    # Stwórz model
    model = KerasLSTMRegression(
        n_features=10,
        n_lstm_layers=2,
        n_lstm_units=64,
        n_dense_neurons=32,
        learning_rate=0.001,
        optimizer_name="adam"
    )

    print("\n📐 Architektura modelu:")
    model.get_summary()

    # Test na losowych danych
    print("\n🧪 Test na losowych danych:")
    X_dummy = np.random.rand(100, 10).astype('float32')
    y_dummy = np.random.rand(100, 1).astype('float32')

    print("  Trenowanie (3 epoki)...")
    history = model.train(X_dummy, y_dummy, epochs=3, verbose=1)

    print(f"\n  MSE po 3 epokach: {history['loss'][-1]:.4f}")
    print(f"  MAE po 3 epokach: {history['mae'][-1]:.4f}")

    print("\n✅ Test zakończony pomyślnie!")
