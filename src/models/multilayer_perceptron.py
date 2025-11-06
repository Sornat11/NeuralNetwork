"""
Multilayer Perceptron (MLP) implementacja w Keras/TensorFlow.
U|ywane do porównania z wBasn implementacj.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple, Optional, Dict, Any


def create_mlp_classifier(
    input_dim: int,
    n_classes: int,
    n_hidden_layers: int = 2,
    hidden_units: Tuple[int, ...] = (128, 64),
    activation: str = "relu",
    dropout_rate: float = 0.0,
) -> keras.Model:
    """
    Tworzy model MLP dla klasyfikacji.

    Args:
        input_dim: Liczba cech wej[ciowych
        n_classes: Liczba klas
        n_hidden_layers: Liczba warstw ukrytych
        hidden_units: Liczba neuronów w ka|dej warstwie ukrytej
        activation: Funkcja aktywacji
        dropout_rate: WspóBczynnik dropout

    Returns:
        Model Keras
    """
    model = models.Sequential(name="MLP_Classifier")

    # Input layer
    model.add(layers.InputLayer(input_shape=(input_dim,)))

    # Hidden layers
    for i in range(n_hidden_layers):
        units = hidden_units[i] if i < len(hidden_units) else hidden_units[-1]
        model.add(layers.Dense(units, activation=activation))

        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    # Output layer
    model.add(layers.Dense(n_classes, activation="softmax"))

    return model


def create_mlp_regressor(
    input_dim: int,
    n_outputs: int = 1,
    n_hidden_layers: int = 2,
    hidden_units: Tuple[int, ...] = (128, 64),
    activation: str = "relu",
    dropout_rate: float = 0.0,
) -> keras.Model:
    """
    Tworzy model MLP dla regresji.

    Args:
        input_dim: Liczba cech wej[ciowych
        n_outputs: Liczba wyj[
        n_hidden_layers: Liczba warstw ukrytych
        hidden_units: Liczba neuronów w ka|dej warstwie ukrytej
        activation: Funkcja aktywacji
        dropout_rate: WspóBczynnik dropout

    Returns:
        Model Keras
    """
    model = models.Sequential(name="MLP_Regressor")

    # Input layer
    model.add(layers.InputLayer(input_shape=(input_dim,)))

    # Hidden layers
    for i in range(n_hidden_layers):
        units = hidden_units[i] if i < len(hidden_units) else hidden_units[-1]
        model.add(layers.Dense(units, activation=activation))

        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    # Output layer (linear for regression)
    model.add(layers.Dense(n_outputs, activation="linear"))

    return model


def compile_mlp(
    model: keras.Model,
    optimizer: str = "adam",
    learning_rate: float = 0.001,
    momentum: float = 0.0,
    loss: Optional[str] = None,
    metrics: Optional[list] = None,
) -> keras.Model:
    """
    Kompiluje model MLP.

    Args:
        model: Model Keras
        optimizer: Nazwa optimizera ('adam', 'sgd', 'rmsprop')
        learning_rate: WspóBczynnik uczenia
        momentum: Momentum (dla SGD)
        loss: Funkcja straty (None = auto-detect)
        metrics: Lista metryk

    Returns:
        Skompilowany model
    """
    # Wybierz optimizer
    if optimizer.lower() == "adam":
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer.lower() == "sgd":
        opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    elif optimizer.lower() == "rmsprop":
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        opt = optimizer

    # Auto-detect loss
    if loss is None:
        last_layer = model.layers[-1]
        if hasattr(last_layer, "activation"):
            if last_layer.activation.__name__ == "softmax":
                loss = "sparse_categorical_crossentropy"
            elif last_layer.activation.__name__ == "sigmoid":
                loss = "binary_crossentropy"
            else:  # linear
                loss = "mse"
        else:
            loss = "mse"

    # Default metrics
    if metrics is None:
        if loss in ["sparse_categorical_crossentropy", "categorical_crossentropy"]:
            metrics = ["accuracy"]
        else:
            metrics = ["mae"]

    model.compile(optimizer=opt, loss=loss, metrics=metrics)

    return model


def train_mlp(
    model: keras.Model,
    X_train: tf.Tensor,
    y_train: tf.Tensor,
    X_val: Optional[tf.Tensor] = None,
    y_val: Optional[tf.Tensor] = None,
    epochs: int = 100,
    batch_size: int = 32,
    verbose: int = 1,
    callbacks: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Trenuje model MLP.

    Args:
        model: Skompilowany model
        X_train: Dane treningowe
        y_train: Etykiety treningowe
        X_val: Dane walidacyjne
        y_val: Etykiety walidacyjne
        epochs: Liczba epok
        batch_size: Rozmiar batcha
        verbose: Poziom verbose
        callbacks: Lista callbacks

    Returns:
        Historia treningu
    """
    validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None

    history = model.fit(
        X_train,
        y_train,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks,
    )

    return history.history
