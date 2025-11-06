"""
Convolutional Neural Network (CNN) implementacja w Keras/TensorFlow.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple, Optional, Dict, Any


def create_cnn_classifier(
    input_shape: Tuple[int, ...],
    n_classes: int,
    n_conv_layers: int = 2,
    filters: Tuple[int, ...] = (32, 64),
    kernel_size: Tuple[int, int] = (3, 3),
    pool_size: Tuple[int, int] = (2, 2),
    n_dense_layers: int = 1,
    dense_units: Tuple[int, ...] = (128,),
    dropout_rate: float = 0.25,
    activation: str = "relu",
) -> keras.Model:
    """
    Tworzy model CNN dla klasyfikacji obrazów.

    Args:
        input_shape: KsztaBt wej[cia (height, width, channels)
        n_classes: Liczba klas
        n_conv_layers: Liczba warstw konwolucyjnych
        filters: Liczba filtrów w ka|dej warstwie konwolucyjnej
        kernel_size: Rozmiar kernela
        pool_size: Rozmiar poolingu
        n_dense_layers: Liczba warstw Dense po flatteningu
        dense_units: Liczba neuronów w warstwach Dense
        dropout_rate: WspóBczynnik dropout
        activation: Funkcja aktywacji

    Returns:
        Model Keras
    """
    model = models.Sequential(name="CNN_Classifier")

    # Pierwsza warstwa konwolucyjna
    model.add(
        layers.Conv2D(
            filters[0] if len(filters) > 0 else 32,
            kernel_size,
            activation=activation,
            input_shape=input_shape,
            padding="same",
        )
    )
    model.add(layers.MaxPooling2D(pool_size))

    # Dodatkowe warstwy konwolucyjne
    for i in range(1, n_conv_layers):
        filter_count = filters[i] if i < len(filters) else filters[-1] * 2
        model.add(layers.Conv2D(filter_count, kernel_size, activation=activation, padding="same"))
        model.add(layers.MaxPooling2D(pool_size))

    # Dropout po konwolucjach
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))

    # Flatten
    model.add(layers.Flatten())

    # Warstwy Dense
    for i in range(n_dense_layers):
        units = dense_units[i] if i < len(dense_units) else dense_units[-1]
        model.add(layers.Dense(units, activation=activation))

        # Dropout po Dense
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    # Output layer
    model.add(layers.Dense(n_classes, activation="softmax"))

    return model


def create_cnn_regressor(
    input_shape: Tuple[int, ...],
    n_outputs: int = 1,
    n_conv_layers: int = 2,
    filters: Tuple[int, ...] = (32, 64),
    kernel_size: Tuple[int, int] = (3, 3),
    pool_size: Tuple[int, int] = (2, 2),
    n_dense_layers: int = 1,
    dense_units: Tuple[int, ...] = (128,),
    dropout_rate: float = 0.25,
    activation: str = "relu",
) -> keras.Model:
    """
    Tworzy model CNN dla regresji.

    Args:
        input_shape: KsztaBt wej[cia
        n_outputs: Liczba wyj[ (warto[ci do predykcji)
        n_conv_layers: Liczba warstw konwolucyjnych
        filters: Liczba filtrów w ka|dej warstwie
        kernel_size: Rozmiar kernela
        pool_size: Rozmiar poolingu
        n_dense_layers: Liczba warstw Dense
        dense_units: Liczba neuronów w warstwach Dense
        dropout_rate: WspóBczynnik dropout
        activation: Funkcja aktywacji

    Returns:
        Model Keras
    """
    model = models.Sequential(name="CNN_Regressor")

    # Pierwsza warstwa konwolucyjna
    model.add(
        layers.Conv2D(
            filters[0] if len(filters) > 0 else 32,
            kernel_size,
            activation=activation,
            input_shape=input_shape,
            padding="same",
        )
    )
    model.add(layers.MaxPooling2D(pool_size))

    # Dodatkowe warstwy konwolucyjne
    for i in range(1, n_conv_layers):
        filter_count = filters[i] if i < len(filters) else filters[-1] * 2
        model.add(layers.Conv2D(filter_count, kernel_size, activation=activation, padding="same"))
        model.add(layers.MaxPooling2D(pool_size))

    # Dropout
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))

    # Flatten
    model.add(layers.Flatten())

    # Warstwy Dense
    for i in range(n_dense_layers):
        units = dense_units[i] if i < len(dense_units) else dense_units[-1]
        model.add(layers.Dense(units, activation=activation))

        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    # Output layer (linear activation for regression)
    model.add(layers.Dense(n_outputs, activation="linear"))

    return model


def create_simple_cnn(input_shape: Tuple[int, ...], n_classes: int) -> keras.Model:
    """
    Tworzy prosty model CNN (baseline).

    Args:
        input_shape: KsztaBt wej[cia
        n_classes: Liczba klas

    Returns:
        Model Keras
    """
    return create_cnn_classifier(
        input_shape=input_shape,
        n_classes=n_classes,
        n_conv_layers=2,
        filters=(32, 64),
        n_dense_layers=1,
        dense_units=(128,),
        dropout_rate=0.25,
    )


def compile_cnn(
    model: keras.Model,
    optimizer: str = "adam",
    learning_rate: float = 0.001,
    momentum: float = 0.0,
    loss: Optional[str] = None,
    metrics: Optional[list] = None,
) -> keras.Model:
    """
    Kompiluje model CNN.

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
        # Sprawdz ostatni warstw
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


def train_cnn(
    model: keras.Model,
    X_train: tf.Tensor,
    y_train: tf.Tensor,
    X_val: Optional[tf.Tensor] = None,
    y_val: Optional[tf.Tensor] = None,
    epochs: int = 10,
    batch_size: int = 32,
    verbose: int = 1,
    callbacks: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Trenuje model CNN.

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
