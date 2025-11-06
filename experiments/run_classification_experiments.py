"""
Skrypt do uruchamiania eksperymentów klasyfikacyjnych.
Testuje własny MLP i gotowy MLP z Keras na datasetach Iris i Wine.
"""

import sys
from pathlib import Path

# Dodaj główny katalog do path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from data.datasets import get_dataset
from src.manual_mlp.model import Model
from src.manual_mlp.layers import LayerDense
from src.manual_mlp.activations import ActivationReLU
from src.manual_mlp.losses import SoftmaxCategoricalCrossentropy
from src.manual_mlp.optimizers import OptimizerSGD, OptimizerAdam, OptimizerRMSprop
from src.manual_mlp.metrics import evaluate_classification, confusion_matrix
from src.models.multilayer_perceptron import create_mlp_classifier, compile_mlp, train_mlp
from experiments.experiment_runner import ExperimentRunner, create_param_grid
from experiments.visualizations import (
    plot_parameter_comparison,
    plot_model_comparison,
    plot_confusion_matrix,
    create_results_summary_table,
)
from utils.seed import set_seed


def create_custom_mlp(n_inputs, n_classes, n_hidden_layers, hidden_units, learning_rate, optimizer_name, momentum):
    """Tworzy własny MLP"""
    model = Model()

    # Warstwy ukryte
    prev_units = n_inputs
    for i in range(n_hidden_layers):
        units = hidden_units if isinstance(hidden_units, int) else hidden_units[i] if i < len(hidden_units) else hidden_units[-1]
        model.add(LayerDense(prev_units, units))
        model.add(ActivationReLU())
        prev_units = units

    # Output layer (już z softmax w loss)
    model.add(LayerDense(prev_units, n_classes))

    # Loss + Optimizer
    loss = SoftmaxCategoricalCrossentropy()

    if optimizer_name == "adam":
        optimizer = OptimizerAdam(learning_rate=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = OptimizerSGD(learning_rate=learning_rate, momentum=momentum)
    elif optimizer_name == "rmsprop":
        optimizer = OptimizerRMSprop(learning_rate=learning_rate)
    else:
        optimizer = OptimizerAdam(learning_rate=learning_rate)

    model.set(loss=loss, optimizer=optimizer)
    model.finalize()

    return model


def train_custom_mlp(model, data, epochs, batch_size, **kwargs):
    """Trenuje własny MLP"""
    validation_data = None
    if data.get("X_val") is not None:
        validation_data = (data["X_val"], data["y_val"])

    history = model.fit(
        data["X_train"],
        data["y_train"],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        verbose=0,
    )
    return history


def eval_custom_mlp(model, data):
    """Ewaluuje własny MLP"""
    predictions = model.predict(data["X"])
    metrics = evaluate_classification(data["y"], predictions, average="macro")
    return metrics


def create_keras_mlp(n_inputs, n_classes, n_hidden_layers, hidden_units, learning_rate, optimizer_name, momentum, **kwargs):
    """Tworzy MLP w Keras"""
    # hidden_units jako tuple
    if isinstance(hidden_units, int):
        hidden_units = (hidden_units,) * n_hidden_layers
    elif isinstance(hidden_units, list):
        hidden_units = tuple(hidden_units)

    model = create_mlp_classifier(
        input_dim=n_inputs,
        n_classes=n_classes,
        n_hidden_layers=n_hidden_layers,
        hidden_units=hidden_units,
        activation="relu",
        dropout_rate=0.0,
    )

    compile_mlp(
        model,
        optimizer=optimizer_name,
        learning_rate=learning_rate,
        momentum=momentum,
    )

    return model


def train_keras_mlp(model, data, epochs, batch_size, **kwargs):
    """Trenuje MLP w Keras"""
    validation_data = None
    if data.get("X_val") is not None:
        validation_data = (data["X_val"], data["y_val"])

    history = train_mlp(
        model,
        data["X_train"],
        data["y_train"],
        X_val=data.get("X_val"),
        y_val=data.get("y_val"),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )
    return history


def eval_keras_mlp(model, data):
    """Ewaluuje MLP w Keras"""
    predictions = model.predict(data["X"], verbose=0)
    metrics = evaluate_classification(data["y"], predictions, average="macro")
    return metrics


def main():
    """Główna funkcja uruchamiająca eksperymenty"""

    print("\n" + "=" * 80)
    print("EKSPERYMENTY KLASYFIKACYJNE")
    print("=" * 80 + "\n")

    set_seed(42)

    # ========== DATASET: IRIS ==========
    print("\n### Dataset: IRIS ###\n")

    # Załaduj dane (70/15/15)
    iris_data = get_dataset("iris", split_type="70_15_15", normalize=True)

    print(f"Train: {iris_data['X_train'].shape}")
    print(f"Val: {iris_data['X_val'].shape}")
    print(f"Test: {iris_data['X_test'].shape}")
    print(f"Klasy: {iris_data['n_classes']}")

    # Parametry do przetestowania
    param_grid = create_param_grid(
        base_params={
            "n_inputs": iris_data["n_features"],
            "n_classes": iris_data["n_classes"],
            "epochs": 100,
            "batch_size": 16,
        },
        variations={
            "n_hidden_layers": [1, 2, 3],  # Liczba warstw
            "hidden_units": [32, 64, 128],  # Liczba neuronów
            "learning_rate": [0.001, 0.01, 0.1],  # Learning rate
            "optimizer_name": ["sgd", "adam", "rmsprop"],  # Optimizers
            "momentum": [0.0, 0.9],  # Momentum (dla SGD)
        },
    )

    runner = ExperimentRunner(results_dir="../results")

    # ===== Eksperyment 1: Własny MLP na Iris =====
    print("\n### Eksperyment 1: Własny MLP na Iris ###\n")

    # Uproszczony param grid (żeby nie trwało wieki)
    simple_param_grid = create_param_grid(
        base_params={
            "n_inputs": iris_data["n_features"],
            "n_classes": iris_data["n_classes"],
            "epochs": 100,
            "batch_size": 16,
            "momentum": 0.9,
        },
        variations={
            "n_hidden_layers": [1, 2, 3],
            "hidden_units": [32, 64],
            "learning_rate": [0.001, 0.01],
            "optimizer_name": ["adam", "sgd"],
        },
    )

    results_custom_iris = runner.run_experiment(
        experiment_name="custom_mlp_iris",
        model_fn=create_custom_mlp,
        train_fn=train_custom_mlp,
        eval_fn=eval_custom_mlp,
        data=iris_data,
        param_grid=simple_param_grid,
        n_repeats=5,
        save_results=True,
        verbose=True,
    )

    # Najlepsze parametry
    best_params = runner.get_best_params(results_custom_iris, metric="test_accuracy_mean")
    print(f"\nNajlepsze parametry (własny MLP): {best_params}")

    # ===== Eksperyment 2: Keras MLP na Iris =====
    print("\n### Eksperyment 2: Keras MLP na Iris ###\n")

    results_keras_iris = runner.run_experiment(
        experiment_name="keras_mlp_iris",
        model_fn=create_keras_mlp,
        train_fn=train_keras_mlp,
        eval_fn=eval_keras_mlp,
        data=iris_data,
        param_grid=simple_param_grid,
        n_repeats=5,
        save_results=True,
        verbose=True,
    )

    # Najlepsze parametry
    best_params_keras = runner.get_best_params(results_keras_iris, metric="test_accuracy_mean")
    print(f"\nNajlepsze parametry (Keras MLP): {best_params_keras}")

    # ========== PORÓWNANIE ==========
    print("\n### Porównanie modeli ###\n")

    # Porównaj średnie accuracy
    mean_custom = results_custom_iris["test_accuracy_mean"].mean()
    mean_keras = results_keras_iris["test_accuracy_mean"].mean()

    print(f"Średnie test accuracy - Własny MLP: {mean_custom:.4f}")
    print(f"Średnie test accuracy - Keras MLP: {mean_keras:.4f}")

    # Wizualizacje
    plot_model_comparison(
        {
            "Custom MLP": results_custom_iris,
            "Keras MLP": results_keras_iris,
        },
        metric="test_accuracy_mean",
        title="Porównanie MLP na Iris",
        save_path="../results/iris_model_comparison.png",
    )

    plot_parameter_comparison(
        results_custom_iris,
        param_name="learning_rate",
        metrics=["test_accuracy_mean", "test_f1_score_mean"],
        title="Wpływ learning rate (Własny MLP - Iris)",
        save_path="../results/iris_custom_learning_rate.png",
    )

    # ========== DATASET: WINE ==========
    print("\n### Dataset: WINE ###\n")

    wine_data = get_dataset("wine", split_type="70_15_15", normalize=True)

    simple_param_grid_wine = create_param_grid(
        base_params={
            "n_inputs": wine_data["n_features"],
            "n_classes": wine_data["n_classes"],
            "epochs": 100,
            "batch_size": 16,
            "momentum": 0.9,
        },
        variations={
            "n_hidden_layers": [1, 2],
            "hidden_units": [32, 64],
            "learning_rate": [0.001, 0.01],
            "optimizer_name": ["adam"],
        },
    )

    # Własny MLP na Wine
    print("\n### Własny MLP na Wine ###\n")
    results_custom_wine = runner.run_experiment(
        experiment_name="custom_mlp_wine",
        model_fn=create_custom_mlp,
        train_fn=train_custom_mlp,
        eval_fn=eval_custom_mlp,
        data=wine_data,
        param_grid=simple_param_grid_wine,
        n_repeats=5,
        save_results=True,
        verbose=True,
    )

    # Keras MLP na Wine
    print("\n### Keras MLP na Wine ###\n")
    results_keras_wine = runner.run_experiment(
        experiment_name="keras_mlp_wine",
        model_fn=create_keras_mlp,
        train_fn=train_keras_mlp,
        eval_fn=eval_keras_mlp,
        data=wine_data,
        param_grid=simple_param_grid_wine,
        n_repeats=5,
        save_results=True,
        verbose=True,
    )

    # Podsumowanie Wine
    mean_custom_wine = results_custom_wine["test_accuracy_mean"].mean()
    mean_keras_wine = results_keras_wine["test_accuracy_mean"].mean()

    print(f"\nŚrednie test accuracy na Wine - Własny MLP: {mean_custom_wine:.4f}")
    print(f"Średnie test accuracy na Wine - Keras MLP: {mean_keras_wine:.4f}")

    print("\n" + "=" * 80)
    print("EKSPERYMENTY ZAKOŃCZONE!")
    print("Wyniki zapisane w katalogu ../results/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
