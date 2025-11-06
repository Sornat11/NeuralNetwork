"""
Recurrent Neural Networks (RNN, LSTM, GRU) implementacja w Keras/TensorFlow.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple, Optional, Dict, Any, Literal


def create_rnn_model(
    input_shape: Tuple[int, int],
    n_outputs: int = 1,
    rnn_type: Literal["rnn", "lstm", "gru"] = "lstm",
    n_rnn_layers: int = 2,
    units: Tuple[int, ...] = (64, 32),
    n_dense_layers: int = 1,
    dense_units: Tuple[int, ...] = (32,),
    dropout_rate: float = 0.2,
    activation: str = "relu",
    output_activation: Optional[str] = None,
    bidirectional: bool = False,
) -> keras.Model:
    """
    Tworzy model RNN (LSTM/GRU/SimpleRNN).

    Args:
        input_shape: KsztaBt wej[cia (timesteps, features)
        n_outputs: Liczba wyj[
        rnn_type: Typ warstwy RNN ('rnn', 'lstm', 'gru')
        n_rnn_layers: Liczba warstw RNN
        units: Liczba jednostek w ka|dej warstwie RNN
        n_dense_layers: Liczba warstw Dense po RNN
        dense_units: Liczba neuronów w warstwach Dense
        dropout_rate: WspóBczynnik dropout
        activation: Funkcja aktywacji dla Dense
        output_activation: Aktywacja output layer (None = linear dla regresji)
        bidirectional: Czy u|ywa bidirectional RNN

    Returns:
        Model Keras
    """
    model = models.Sequential(name=f"{rnn_type.upper()}_Model")

    # Wybierz typ RNN
    rnn_layer_class = {
        "rnn": layers.SimpleRNN,
        "lstm": layers.LSTM,
        "gru": layers.GRU,
    }[rnn_type.lower()]

    # Dodaj warstwy RNN
    for i in range(n_rnn_layers):
        units_count = units[i] if i < len(units) else units[-1]
        return_sequences = i < n_rnn_layers - 1  # Ostatnia warstwa nie zwraca sekwencji

        rnn_layer = rnn_layer_class(
            units_count,
            return_sequences=return_sequences,
            dropout=dropout_rate if dropout_rate > 0 else 0.0,
            recurrent_dropout=dropout_rate if dropout_rate > 0 else 0.0,
        )

        # Opcjonalnie owijamy w Bidirectional
        if bidirectional:
            rnn_layer = layers.Bidirectional(rnn_layer)

        # Pierwsza warstwa musi mie input_shape
        if i == 0:
            model.add(rnn_layer)
            model.add(layers.InputLayer(input_shape=input_shape))
            model.layers.pop()  # Remove InputLayer
            model.add(rnn_layer_class(
                units_count,
                return_sequences=return_sequences,
                dropout=dropout_rate if dropout_rate > 0 else 0.0,
                recurrent_dropout=dropout_rate if dropout_rate > 0 else 0.0,
                input_shape=input_shape,
            ) if not bidirectional else layers.Bidirectional(
                rnn_layer_class(
                    units_count,
                    return_sequences=return_sequences,
                    dropout=dropout_rate if dropout_rate > 0 else 0.0,
                    recurrent_dropout=dropout_rate if dropout_rate > 0 else 0.0,
                ),
                input_shape=input_shape,
            ))
        else:
            model.add(rnn_layer)

    # Warstwy Dense
    for i in range(n_dense_layers):
        units_count = dense_units[i] if i < len(dense_units) else dense_units[-1]
        model.add(layers.Dense(units_count, activation=activation))

        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    # Output layer
    model.add(layers.Dense(n_outputs, activation=output_activation))

    return model


def create_lstm_regressor(
    input_shape: Tuple[int, int],
    n_outputs: int = 1,
    lstm_units: Tuple[int, ...] = (64, 32),
    dense_units: Tuple[int, ...] = (32,),
    dropout_rate: float = 0.2,
) -> keras.Model:
    """
    Tworzy model LSTM dla regresji (szeregów czasowych).

    Args:
        input_shape: KsztaBt wej[cia (timesteps, features)
        n_outputs: Liczba wyj[
        lstm_units: Liczba jednostek LSTM w ka|dej warstwie
        dense_units: Liczba neuronów w warstwach Dense
        dropout_rate: WspóBczynnik dropout

    Returns:
        Model Keras
    """
    return create_rnn_model(
        input_shape=input_shape,
        n_outputs=n_outputs,
        rnn_type="lstm",
        n_rnn_layers=len(lstm_units),
        units=lstm_units,
        n_dense_layers=len(dense_units),
        dense_units=dense_units,
        dropout_rate=dropout_rate,
        output_activation="linear",
    )


def create_gru_regressor(
    input_shape: Tuple[int, int],
    n_outputs: int = 1,
    gru_units: Tuple[int, ...] = (64, 32),
    dense_units: Tuple[int, ...] = (32,),
    dropout_rate: float = 0.2,
) -> keras.Model:
    """
    Tworzy model GRU dla regresji (szeregów czasowych).

    Args:
        input_shape: KsztaBt wej[cia (timesteps, features)
        n_outputs: Liczba wyj[
        gru_units: Liczba jednostek GRU w ka|dej warstwie
        dense_units: Liczba neuronów w warstwach Dense
        dropout_rate: WspóBczynnik dropout

    Returns:
        Model Keras
    """
    return create_rnn_model(
        input_shape=input_shape,
        n_outputs=n_outputs,
        rnn_type="gru",
        n_rnn_layers=len(gru_units),
        units=gru_units,
        n_dense_layers=len(dense_units),
        dense_units=dense_units,
        dropout_rate=dropout_rate,
        output_activation="linear",
    )


def create_simple_lstm(input_shape: Tuple[int, int], n_outputs: int = 1) -> keras.Model:
    """
    Tworzy prosty model LSTM (baseline).

    Args:
        input_shape: KsztaBt wej[cia (timesteps, features)
        n_outputs: Liczba wyj[

    Returns:
        Model Keras
    """
    return create_lstm_regressor(
        input_shape=input_shape,
        n_outputs=n_outputs,
        lstm_units=(64, 32),
        dense_units=(32,),
        dropout_rate=0.2,
    )


def compile_rnn(
    model: keras.Model,
    optimizer: str = "adam",
    learning_rate: float = 0.001,
    momentum: float = 0.0,
    loss: str = "mse",
    metrics: Optional[list] = None,
) -> keras.Model:
    """
    Kompiluje model RNN.

    Args:
        model: Model Keras
        optimizer: Nazwa optimizera ('adam', 'sgd', 'rmsprop')
        learning_rate: WspóBczynnik uczenia
        momentum: Momentum (dla SGD)
        loss: Funkcja straty
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

    # Default metrics
    if metrics is None:
        if loss in ["mse", "mae", "huber"]:
            metrics = ["mae", "mse"]
        else:
            metrics = ["accuracy"]

    model.compile(optimizer=opt, loss=loss, metrics=metrics)

    return model


def train_rnn(
    model: keras.Model,
    X_train: tf.Tensor,
    y_train: tf.Tensor,
    X_val: Optional[tf.Tensor] = None,
    y_val: Optional[tf.Tensor] = None,
    epochs: int = 50,
    batch_size: int = 32,
    verbose: int = 1,
    callbacks: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Trenuje model RNN.

    Args:
        model: Skompilowany model
        X_train: Dane treningowe
        y_train: Warto[ci treningowe
        X_val: Dane walidacyjne
        y_val: Warto[ci walidacyjne
        epochs: Liczba epok
        batch_size: Rozmiar batcha
        verbose: Poziom verbose
        callbacks: Lista callbacks

    Returns:
        Historia treningu
    """
    validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None

    # Default callbacks - early stopping
    if callbacks is None:
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss" if validation_data else "loss",
                patience=10,
                restore_best_weights=True,
            )
        ]

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


def create_conv1d_lstm_hybrid(
    input_shape: Tuple[int, int],
    n_outputs: int = 1,
    n_conv_layers: int = 2,
    filters: Tuple[int, ...] = (64, 32),
    kernel_size: int = 3,
    lstm_units: Tuple[int, ...] = (50,),
    dense_units: Tuple[int, ...] = (25,),
    dropout_rate: float = 0.2,
) -> keras.Model:
    """
    Tworzy hybrydowy model Conv1D + LSTM dla szeregów czasowych.

    Args:
        input_shape: KsztaBt wej[cia (timesteps, features)
        n_outputs: Liczba wyj[
        n_conv_layers: Liczba warstw Conv1D
        filters: Liczba filtrów w ka|dej warstwie Conv1D
        kernel_size: Rozmiar kernela Conv1D
        lstm_units: Liczba jednostek LSTM
        dense_units: Liczba neuronów w warstwach Dense
        dropout_rate: WspóBczynnik dropout

    Returns:
        Model Keras
    """
    model = models.Sequential(name="Conv1D_LSTM_Hybrid")

    # Warstwy Conv1D
    for i in range(n_conv_layers):
        filter_count = filters[i] if i < len(filters) else filters[-1]

        if i == 0:
            model.add(
                layers.Conv1D(
                    filter_count,
                    kernel_size,
                    activation="relu",
                    padding="same",
                    input_shape=input_shape,
                )
            )
        else:
            model.add(layers.Conv1D(filter_count, kernel_size, activation="relu", padding="same"))

        model.add(layers.MaxPooling1D(pool_size=2))

        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    # Warstwy LSTM
    for i, units in enumerate(lstm_units):
        return_sequences = i < len(lstm_units) - 1
        model.add(
            layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=dropout_rate if dropout_rate > 0 else 0.0,
            )
        )

    # Warstwy Dense
    for units in dense_units:
        model.add(layers.Dense(units, activation="relu"))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    # Output
    model.add(layers.Dense(n_outputs, activation="linear"))

    return model
