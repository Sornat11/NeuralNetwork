
import numpy as np
from typing import Optional, Tuple, List


class Model:
    """
    Model sieci neuronowej z własną implementacją forward i backward pass.
    """

    def __init__(self):
        self.layers: list = []
        self.loss = None
        self.optimizer = None
        self.trainable_layers: list = []

    def add(self, layer) -> None:
        """Dodaje warstwę do modelu"""
        self.layers.append(layer)

    def set(self, *, loss=None, optimizer=None) -> None:
        """
        Konfiguruje model z loss function i optimizer.

        Args:
            loss: Funkcja straty (np. LossCategoricalCrossentropy)
            optimizer: Optimizer (np. OptimizerAdam)
        """
        if loss:
            self.loss = loss
        if optimizer:
            self.optimizer = optimizer

    def finalize(self) -> None:
        """
        Finalizuje model - identyfikuje warstwy trenowalne.
        Wywołaj po dodaniu wszystkich warstw.
        """
        self.trainable_layers = []
        for layer in self.layers:
            if hasattr(layer, "weights"):
                self.trainable_layers.append(layer)

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass przez wszystkie warstwy.

        Args:
            X: Dane wejściowe
            training: Czy model jest w trybie treningu

        Returns:
            Wyjście z ostatniej warstwy
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, output: np.ndarray, y: np.ndarray) -> None:
        """
        Backward pass przez wszystkie warstwy.

        Args:
            output: Wyjście z forward pass
            y: Prawdziwe etykiety/wartości
        """
        # Gradient z loss function
        self.loss.backward(output, y)

        # Propagacja wsteczna przez warstwy (od końca do początku)
        dvalues = self.loss.dinputs
        for layer in reversed(self.layers):
            layer.backward(dvalues)
            dvalues = layer.dinputs

    def train_on_batch(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Trenuje model na jednym batch'u danych.

        Args:
            X: Dane wejściowe
            y: Prawdziwe etykiety/wartości

        Returns:
            Tuple (loss, predictions)
        """
        # Forward pass
        output = self.forward(X, training=True)

        # Oblicz loss
        loss = self.loss.calculate(output, y)

        # Backward pass
        self.backward(output, y)

        # Aktualizacja wag
        self.optimizer.pre_update_params()
        for layer in self.trainable_layers:
            self.optimizer.update_params(layer)
        self.optimizer.post_update_params()

        return loss, output

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 1,
        batch_size: Optional[int] = None,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: bool = True,
    ) -> dict:
        """
        Trenuje model na danych treningowych.

        Args:
            X: Dane treningowe
            y: Etykiety treningowe
            epochs: Liczba epok
            batch_size: Rozmiar batcha (None = pełny dataset)
            validation_data: Tuple (X_val, y_val) dla walidacji
            verbose: Czy wyświetlać postęp

        Returns:
            Dict z historią treningu (loss, val_loss, etc.)
        """
        # Finalizuj model jeśli nie został jeszcze
        if not self.trainable_layers:
            self.finalize()

        # Historia treningu
        history = {
            "loss": [],
            "val_loss": [],
        }

        train_steps = 1 if batch_size is None else len(X) // batch_size
        if batch_size is not None and len(X) % batch_size != 0:
            train_steps += 1

        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_steps = 0

            # Mini-batches
            for step in range(train_steps):
                # Przygotuj batch
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    start_idx = step * batch_size
                    end_idx = min(start_idx + batch_size, len(X))
                    batch_X = X[start_idx:end_idx]
                    batch_y = y[start_idx:end_idx]

                # Train na batchu
                loss, _ = self.train_on_batch(batch_X, batch_y)
                epoch_loss += loss
                epoch_steps += 1

            # Średni loss z epoki
            epoch_loss = epoch_loss / epoch_steps
            history["loss"].append(epoch_loss)

            # Walidacja
            if validation_data is not None:
                X_val, y_val = validation_data
                val_output = self.forward(X_val, training=False)
                val_loss = self.loss.calculate(val_output, y_val)
                history["val_loss"].append(val_loss)

                if verbose:
                    print(
                        f"Epoch {epoch + 1}/{epochs}, "
                        f"Loss: {epoch_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}"
                    )
            else:
                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predykcja dla danych wejściowych.

        Args:
            X: Dane wejściowe

        Returns:
            Predykcje
        """
        return self.forward(X, training=False)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Ewaluacja modelu na danych testowych.

        Args:
            X: Dane testowe
            y: Etykiety testowe

        Returns:
            Tuple (loss, predictions)
        """
        output = self.predict(X)
        loss = self.loss.calculate(output, y)
        return loss, output