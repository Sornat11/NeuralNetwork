"""
Skrypt do uruchamiania eksperymentów na obrazach.
Testuje własny MLP, Keras MLP i CNN na Fashion MNIST.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from data.datasets import get_dataset
from src.manual_mlp.model import Model
from src.manual_mlp.layers import LayerDense
from src.manual_mlp.activations import ActivationReLU
from src.manual_mlp.losses import SoftmaxCategoricalCrossentropy
from src.manual_mlp.optimizers import OptimizerAdam, OptimizerSGD
from src.manual_mlp.metrics import evaluate_classification, confusion_matrix
from src.models.multilayer_perceptron import create_mlp_classifier, compile_mlp, train_mlp
from src.models.convolutional_nn import create_cnn_classifier, compile_cnn, train_cnn
from experiments.experiment_runner import ExperimentRunner, create_param_grid
from experiments.visualizations import (
    plot_model_comparison,
    plot_confusion_matrix,
    plot_learning_curves,
)
from utils.seed import set_seed


def create_custom_mlp_image(n_inputs, n_classes, n_hidden_layers, hidden_units, learning_rate, optimizer_name, **kwargs):
    """Tworzy własny MLP dla obrazów"""
    model = Model()
    prev_units = n_inputs

    for i in range(n_hidden_layers):
        units = hidden_units if isinstance(hidden_units, int) else hidden_units[i]
        model.add(LayerDense(prev_units, units))
        model.add(ActivationReLU())
        prev_units = units

    model.add(LayerDense(prev_units, n_classes))

    loss = SoftmaxCategoricalCrossentropy()
    optimizer = OptimizerAdam(learning_rate=learning_rate) if optimizer_name == "adam" else OptimizerSGD(learning_rate=learning_rate)

    model.set(loss=loss, optimizer=optimizer)
    model.finalize()
    return model


def train_custom_mlp_image(model, data, epochs, batch_size, **kwargs):
    """Trenuje własny MLP"""
    validation_data = (data["X_val"], data["y_val"]) if data.get("X_val") is not None else None
    history = model.fit(
        data["X_train"], data["y_train"],
        epochs=epochs, batch_size=batch_size,
        validation_data=validation_data, verbose=0
    )
    return history


def eval_custom_mlp_image(model, data):
    """Ewaluuje własny MLP"""
    predictions = model.predict(data["X"])
    metrics = evaluate_classification(data["y"], predictions, average="macro")
    return metrics


def create_keras_mlp_image(n_inputs, n_classes, n_hidden_layers, hidden_units, learning_rate, optimizer_name, **kwargs):
    """Tworzy Keras MLP"""
    if isinstance(hidden_units, int):
        hidden_units = (hidden_units,) * n_hidden_layers

    model = create_mlp_classifier(
        input_dim=n_inputs, n_classes=n_classes,
        n_hidden_layers=n_hidden_layers, hidden_units=hidden_units,
        activation="relu", dropout_rate=0.2
    )
    compile_mlp(model, optimizer=optimizer_name, learning_rate=learning_rate)
    return model


def train_keras_mlp_image(model, data, epochs, batch_size, **kwargs):
    """Trenuje Keras MLP"""
    return train_mlp(
        model, data["X_train"], data["y_train"],
        X_val=data.get("X_val"), y_val=data.get("y_val"),
        epochs=epochs, batch_size=batch_size, verbose=0
    )


def eval_keras_mlp_image(model, data):
    """Ewaluuje Keras MLP"""
    predictions = model.predict(data["X"], verbose=0)
    metrics = evaluate_classification(data["y"], predictions, average="macro")
    return metrics


def create_cnn_image(n_classes, n_conv_layers, filters, learning_rate, optimizer_name, **kwargs):
    """Tworzy CNN dla obrazów"""
    if isinstance(filters, int):
        filters = (filters, filters * 2)[:n_conv_layers]

    model = create_cnn_classifier(
        input_shape=(28, 28, 1),
        n_classes=n_classes,
        n_conv_layers=n_conv_layers,
        filters=filters,
        n_dense_layers=1,
        dense_units=(128,),
        dropout_rate=0.25,
    )
    compile_cnn(model, optimizer=optimizer_name, learning_rate=learning_rate)
    return model


def train_cnn_image(model, data, epochs, batch_size, **kwargs):
    """Trenuje CNN"""
    return train_cnn(
        model, data["X_train_cnn"], data["y_train"],
        X_val=data.get("X_val_cnn"), y_val=data.get("y_val"),
        epochs=epochs, batch_size=batch_size, verbose=0
    )


def eval_cnn_image(model, data):
    """Ewaluuje CNN"""
    key = "X_cnn" if "X_cnn" in data else "X_test_cnn"
    X = data.get(key, data["X"])

    if X.ndim == 2:  # Jeśli płaski, reshape do obrazów
        X = X.reshape(-1, 28, 28, 1)

    predictions = model.predict(X, verbose=0)
    metrics = evaluate_classification(data["y"], predictions, average="macro")
    return metrics


def main():
    print("\n" + "=" * 80)
    print("EKSPERYMENTY NA OBRAZACH - FASHION MNIST")
    print("=" * 80 + "\n")

    set_seed(42)

    # Załaduj Fashion MNIST
    print("### Dataset: Fashion MNIST ###\n")
    print("Ładowanie danych (może potrwać chwilę)...\n")

    data = get_dataset("fashion_mnist", split_type="70_15_15", normalize=True)

    print(f"Train: {data['X_train'].shape}")
    print(f"Val: {data['X_val'].shape}")
    print(f"Test: {data['X_test'].shape}")
    print(f"Classes: {data['n_classes']}")
    print(f"Image shape: {data['image_shape']}")

    runner = ExperimentRunner(results_dir="../results")

    # ===== Eksperyment 1: Własny MLP =====
    print("\n### Eksperyment 1: Własny MLP na Fashion MNIST ###\n")

    param_grid_custom = create_param_grid(
        base_params={
            "n_inputs": data["n_features"],
            "n_classes": data["n_classes"],
            "epochs": 10,  # Mniej epok ze względu na rozmiar datasetu
            "batch_size": 128,
        },
        variations={
            "n_hidden_layers": [1, 2],
            "hidden_units": [128, 256],
            "learning_rate": [0.001, 0.01],
            "optimizer_name": ["adam"],
        },
    )

    results_custom = runner.run_experiment(
        experiment_name="custom_mlp_fashion_mnist",
        model_fn=create_custom_mlp_image,
        train_fn=train_custom_mlp_image,
        eval_fn=eval_custom_mlp_image,
        data=data,
        param_grid=param_grid_custom,
        n_repeats=3,  # Mniej powtórzeń ze względu na czas
        save_results=True,
        verbose=True,
    )

    # ===== Eksperyment 2: Keras MLP =====
    print("\n### Eksperyment 2: Keras MLP na Fashion MNIST ###\n")

    results_keras_mlp = runner.run_experiment(
        experiment_name="keras_mlp_fashion_mnist",
        model_fn=create_keras_mlp_image,
        train_fn=train_keras_mlp_image,
        eval_fn=eval_keras_mlp_image,
        data=data,
        param_grid=param_grid_custom,
        n_repeats=3,
        save_results=True,
        verbose=True,
    )

    # ===== Eksperyment 3: CNN =====
    print("\n### Eksperyment 3: CNN na Fashion MNIST ###\n")

    param_grid_cnn = create_param_grid(
        base_params={
            "n_classes": data["n_classes"],
            "epochs": 10,
            "batch_size": 128,
        },
        variations={
            "n_conv_layers": [2, 3],
            "filters": [32, 64],
            "learning_rate": [0.001],
            "optimizer_name": ["adam"],
        },
    )

    results_cnn = runner.run_experiment(
        experiment_name="cnn_fashion_mnist",
        model_fn=create_cnn_image,
        train_fn=train_cnn_image,
        eval_fn=eval_cnn_image,
        data=data,
        param_grid=param_grid_cnn,
        n_repeats=3,
        save_results=True,
        verbose=True,
    )

    # ===== PORÓWNANIE =====
    print("\n### Porównanie modeli ###\n")

    mean_acc = {
        "Custom MLP": results_custom["test_accuracy_mean"].mean(),
        "Keras MLP": results_keras_mlp["test_accuracy_mean"].mean(),
        "CNN": results_cnn["test_accuracy_mean"].mean(),
    }

    for model_name, acc in mean_acc.items():
        print(f"Średnie test accuracy - {model_name}: {acc:.4f}")

    # Wizualizacje
    plot_model_comparison(
        {
            "Custom MLP": results_custom,
            "Keras MLP": results_keras_mlp,
            "CNN": results_cnn,
        },
        metric="test_accuracy_mean",
        title="Porównanie modeli na Fashion MNIST",
        save_path="../results/fashion_mnist_comparison.png",
    )

    # Najlepszy model - wytrenuj i pokaż confusion matrix
    print("\n### Confusion Matrix dla najlepszego modelu ###\n")

    best_model_type = max(mean_acc, key=mean_acc.get)
    print(f"Najlepszy model: {best_model_type}")

    # Trenuj najlepszy model z najlepszymi parametrami
    if best_model_type == "CNN":
        best_params = runner.get_best_params(results_cnn, metric="test_accuracy_mean")
        model = create_cnn_image(**best_params)
        train_cnn_image(model, data, epochs=15, batch_size=128)

        # Predykcje na test set
        predictions = model.predict(data["X_test_cnn"], verbose=0)
        cm = confusion_matrix(data["y_test"], predictions)

        plot_confusion_matrix(
            cm,
            class_names=data["target_names"],
            title=f"Confusion Matrix - {best_model_type}",
            save_path="../results/fashion_mnist_confusion_matrix.png",
        )

    print("\n" + "=" * 80)
    print("EKSPERYMENTY NA OBRAZACH ZAKOŃCZONE!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
