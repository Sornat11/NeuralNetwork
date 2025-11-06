"""
Skrypt do uruchamiania eksperymentów regresyjnych na szeregach czasowych.
Testuje MLP, CNN i RNN/LSTM/GRU na datasecie Airline Passengers.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import tensorflow as tf
from data.datasets import get_dataset
from src.models.multilayer_perceptron import create_mlp_regressor, compile_mlp, train_mlp
from src.models.convolutional_nn import create_cnn_regressor, compile_cnn, train_cnn
from src.models.recurrent_nn import create_lstm_regressor, create_gru_regressor, compile_rnn, train_rnn
from src.manual_mlp.metrics import evaluate_regression
from experiments.experiment_runner import ExperimentRunner, create_param_grid
from experiments.visualizations import plot_model_comparison, plot_parameter_comparison
from utils.seed import set_seed


def eval_regression_model(model, data):
    """Ewaluuje model regresyjny"""
    predictions = model.predict(data["X"], verbose=0)
    if predictions.ndim > 1:
        predictions = predictions.flatten()
    metrics = evaluate_regression(data["y"], predictions)
    return metrics


def main():
    print("\n" + "=" * 80)
    print("EKSPERYMENTY REGRESYJNE - SZEREGI CZASOWE")
    print("=" * 80 + "\n")

    set_seed(42)

    # Załaduj dane Airline Passengers
    print("### Dataset: Airline Passengers ###\n")
    data = get_dataset("airline", split_type="70_15_15", normalize=True, lookback=12)

    print(f"Train: {data['X_train'].shape}")
    print(f"Val: {data['X_val'].shape}")
    print(f"Test: {data['X_test'].shape}")
    print(f"Lookback: {data['lookback']}")

    runner = ExperimentRunner(results_dir="../results")

    # ===== MLP dla szeregów czasowych =====
    print("\n### Eksperyment 1: MLP Regressor ###\n")

    def create_mlp_ts(learning_rate, optimizer_name, n_hidden_layers, hidden_units, **kwargs):
        model = create_mlp_regressor(
            input_dim=data["n_features"],
            n_outputs=1,
            n_hidden_layers=n_hidden_layers,
            hidden_units=(hidden_units,) * n_hidden_layers,
        )
        compile_mlp(model, optimizer=optimizer_name, learning_rate=learning_rate, loss="mse")
        return model

    def train_mlp_ts(model, data_dict, epochs, batch_size, **kwargs):
        return train_mlp(
            model, data_dict["X_train"], data_dict["y_train"],
            X_val=data_dict.get("X_val"), y_val=data_dict.get("y_val"),
            epochs=epochs, batch_size=batch_size, verbose=0
        )

    param_grid_mlp = create_param_grid(
        base_params={"epochs": 50, "batch_size": 16},
        variations={
            "n_hidden_layers": [1, 2],
            "hidden_units": [32, 64],
            "learning_rate": [0.001, 0.01],
            "optimizer_name": ["adam", "rmsprop"],
        },
    )

    results_mlp = runner.run_experiment(
        experiment_name="mlp_airline",
        model_fn=create_mlp_ts,
        train_fn=train_mlp_ts,
        eval_fn=eval_regression_model,
        data=data,
        param_grid=param_grid_mlp,
        n_repeats=5,
        save_results=True,
        verbose=True,
    )

    # ===== LSTM dla szeregów czasowych =====
    print("\n### Eksperyment 2: LSTM Regressor ###\n")

    def create_lstm_ts(learning_rate, optimizer_name, lstm_units, dense_units, **kwargs):
        model = create_lstm_regressor(
            input_shape=(data["lookback"], 1),
            n_outputs=1,
            lstm_units=(lstm_units,),
            dense_units=(dense_units,),
        )
        compile_rnn(model, optimizer=optimizer_name, learning_rate=learning_rate, loss="mse")
        return model

    def train_lstm_ts(model, data_dict, epochs, batch_size, **kwargs):
        return train_rnn(
            model, data_dict["X_train_rnn"], data_dict["y_train"],
            X_val=data_dict.get("X_val_rnn"), y_val=data_dict.get("y_val"),
            epochs=epochs, batch_size=batch_size, verbose=0
        )

    def eval_lstm_ts(model, data_dict):
        """Ewaluuje LSTM (używa X_rnn zamiast X)"""
        key = "X_rnn" if "X_rnn" in data_dict else "X_train_rnn"
        if key not in data_dict:
            # Reshape X do (samples, timesteps, features)
            X = data_dict["X"].reshape(data_dict["X"].shape[0], data_dict["X"].shape[1], 1)
        else:
            X = data_dict[key]

        predictions = model.predict(X, verbose=0).flatten()
        metrics = evaluate_regression(data_dict["y"], predictions)
        return metrics

    param_grid_lstm = create_param_grid(
        base_params={"epochs": 50, "batch_size": 16, "dense_units": 16},
        variations={
            "lstm_units": [32, 64],
            "learning_rate": [0.001, 0.01],
            "optimizer_name": ["adam"],
        },
    )

    # Przygotuj dane dla LSTM (z reshaped X)
    lstm_data = {
        **data,
        "X": data["X_train_rnn"],  # Użyj reshaped dla eval
        "y": data["y_train"]
    }
    # Dla eval na test set
    lstm_data_test = {
        "X_rnn": data["X_test_rnn"],
        "y": data["y_test"]
    }

    # Tymczasowo upraszczamy - użyj prostych danych
    results_lstm = runner.run_experiment(
        experiment_name="lstm_airline",
        model_fn=create_lstm_ts,
        train_fn=train_lstm_ts,
        eval_fn=eval_lstm_ts,
        data=data,
        param_grid=param_grid_lstm,
        n_repeats=3,  # Mniej powtórzeń bo LSTM jest wolniejszy
        save_results=True,
        verbose=True,
    )

    # ===== GRU dla szeregów czasowych =====
    print("\n### Eksperyment 3: GRU Regressor ###\n")

    def create_gru_ts(learning_rate, optimizer_name, gru_units, dense_units, **kwargs):
        model = create_gru_regressor(
            input_shape=(data["lookback"], 1),
            n_outputs=1,
            gru_units=(gru_units,),
            dense_units=(dense_units,),
        )
        compile_rnn(model, optimizer=optimizer_name, learning_rate=learning_rate, loss="mse")
        return model

    param_grid_gru = create_param_grid(
        base_params={"epochs": 50, "batch_size": 16, "dense_units": 16},
        variations={
            "gru_units": [32, 64],
            "learning_rate": [0.001, 0.01],
            "optimizer_name": ["adam"],
        },
    )

    results_gru = runner.run_experiment(
        experiment_name="gru_airline",
        model_fn=create_gru_ts,
        train_fn=train_lstm_ts,  # Używa tej samej funkcji trenowania
        eval_fn=eval_lstm_ts,
        data=data,
        param_grid=param_grid_gru,
        n_repeats=3,
        save_results=True,
        verbose=True,
    )

    # ===== PORÓWNANIE =====
    print("\n### Porównanie modeli ###\n")

    mean_mse = {
        "MLP": results_mlp["test_mse_mean"].mean(),
        "LSTM": results_lstm["test_mse_mean"].mean(),
        "GRU": results_gru["test_mse_mean"].mean(),
    }

    for model_name, mse in mean_mse.items():
        print(f"Średnie test MSE - {model_name}: {mse:.6f}")

    # Wizualizacje
    plot_model_comparison(
        {"MLP": results_mlp, "LSTM": results_lstm, "GRU": results_gru},
        metric="test_mse_mean",
        title="Porównanie modeli na Airline Passengers (MSE)",
        save_path="../results/airline_model_comparison.png",
    )

    print("\n" + "=" * 80)
    print("EKSPERYMENTY REGRESYJNE ZAKOŃCZONE!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
