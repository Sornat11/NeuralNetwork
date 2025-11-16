"""
Skrypt do generowania wszystkich wizualizacji dla raportu.

Generuje:
1. Learning curves dla najlepszych modeli (manual + Keras)
2. Confusion matrices dla klasyfikacji
3. Regression scatter plots
4. Wykresy porównawcze manual vs Keras
"""

import os
import numpy as np
import pandas as pd

from utils.visualization import ResultsVisualizer
from src.manual_mlp.model import Model as ManualModel
from src.models.keras_mlp import KerasMLPModel


def load_data(path: str):
    """Wczytuje dane z pliku CSV."""
    df = pd.read_csv(path)

    if df.columns[0].lower() == "datetime":
        df.set_index(df.columns[0], inplace=True)

    X = df.iloc[:, :-1].values
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    y = df.iloc[:, -1].values
    return X, y


def find_best_config(results_file: str, metric: str = "test_accuracy") -> dict:
    """
    Znajduje najlepszą konfigurację hiperparametrów z pliku Excel.

    Args:
        results_file: Ścieżka do pliku Excel
        metric: Metryka do optymalizacji

    Returns:
        Słownik z najlepszymi hiperparametrami
    """
    df = pd.read_excel(results_file, sheet_name="Results")

    # Wybierz najlepszy wiersz
    if 'loss' in metric or 'mse' in metric:
        best_idx = df[metric].idxmin()
    else:
        best_idx = df[metric].idxmax()

    best_row = df.loc[best_idx]

    config = {
        "n_hidden_layers": int(best_row["n_hidden_layers"]),
        "n_neurons": int(best_row["n_neurons"]),
        "learning_rate": float(best_row["learning_rate"]),
    }

    return config, best_row


def generate_learning_curves_manual(
    dataset_name: str,
    split_name: str,
    visualizer: ResultsVisualizer
):
    """Generuje learning curves dla ręcznego modelu."""
    print(f"\n📈 Generuję learning curves (Manual): {dataset_name} - {split_name}")

    # Określ ścieżki
    data_dir = os.path.join("data", dataset_name)
    results_file = os.path.join("results", f"manual_perceptron_wyniki_{dataset_name}_{split_name}.xlsx")

    if not os.path.exists(results_file):
        print(f"⚠️  Plik nie istnieje: {results_file}")
        return

    # Wczytaj dane
    if split_name == "train_test":
        train_file, test_file, val_file = "train80.csv", "test20.csv", None
    else:
        train_file, test_file, val_file = "train70.csv", "test15.csv", "validation15.csv"

    X_train, y_train = load_data(os.path.join(data_dir, train_file))
    X_test, y_test = load_data(os.path.join(data_dir, test_file))
    X_val, y_val = None, None
    if val_file:
        X_val, y_val = load_data(os.path.join(data_dir, val_file))

    # Znajdź najlepszą konfigurację
    is_regression = "regression" in dataset_name
    metric = "test_mse" if is_regression else "test_accuracy"
    config, best_row = find_best_config(results_file, metric)

    # Określ n_outputs
    if is_regression:
        n_outputs = 1
    else:
        n_outputs = len(np.unique(y_train))

    # Trenuj model i zbieraj historię
    model = ManualModel(
        n_inputs=X_train.shape[1],
        n_hidden_layers=config["n_hidden_layers"],
        n_neurons=config["n_neurons"],
        n_outputs=n_outputs,
        learning_rate=config["learning_rate"]
    )

    # Trenuj i zbieraj metryki co epokę
    history = {"loss": [], "val_loss": []}
    if is_regression:
        history.update({"mae": [], "val_mae": []})
    else:
        history.update({"accuracy": [], "val_accuracy": []})

    from src.manual_mlp.metrics import ModelMetrics
    metrics_calc = ModelMetrics()

    batch_size = 32
    epochs = 50

    print(f"   Trening modelu (może potrwać ~1 min)...")

    for epoch in range(epochs):
        epoch_losses = []
        epoch_metrics = []

        # Trening
        for start in range(0, len(X_train), batch_size):
            end = start + batch_size
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            if is_regression:
                # Regresja
                if y_batch.ndim == 1:
                    y_batch = y_batch.reshape(-1, 1)

                preds = model.forward(X_batch)
                loss = metrics_calc.mse(preds, y_batch)
                metric_val = metrics_calc.mae(preds, y_batch)

                dvalues = preds - y_batch
            else:
                # Klasyfikacja
                probs = model.forward(X_batch, classification=True)
                loss = metrics_calc.crossentropy_loss(probs, y_batch)
                metric_val = metrics_calc.accuracy(probs, y_batch)

                samples = len(probs)
                dvalues = probs.copy()
                dvalues[np.arange(samples), y_batch] -= 1
                dvalues = dvalues / samples

            epoch_losses.append(loss)
            epoch_metrics.append(metric_val)

            # Backward
            if len(model.hidden_layers) > 0:
                prev_hidden_out = model.activations[-1].output
            else:
                prev_hidden_out = X_batch

            d_out = model.output_layer.backward(dvalues, prev_hidden_out)

            for i in reversed(range(len(model.hidden_layers))):
                d_act = model.activations[i].backward(d_out)
                prev_input = X_batch if i == 0 else model.activations[i - 1].output
                d_out = model.hidden_layers[i].backward(d_act, prev_input)

        # Zapisz metryki treningowe
        history["loss"].append(np.mean(epoch_losses))
        metric_key = "mae" if is_regression else "accuracy"
        history[metric_key].append(np.mean(epoch_metrics))

        # Walidacja (jeśli dostępna)
        if X_val is not None:
            if is_regression:
                val_preds = model.forward(X_val)
                val_y = y_val.reshape(-1, 1) if y_val.ndim == 1 else y_val
                val_loss = metrics_calc.mse(val_preds, val_y)
                val_metric = metrics_calc.mae(val_preds, val_y)
            else:
                val_probs = model.forward(X_val, classification=True)
                val_loss = metrics_calc.crossentropy_loss(val_probs, y_val)
                val_metric = metrics_calc.accuracy(val_probs, y_val)

            history["val_loss"].append(val_loss)
            history[f"val_{metric_key}"].append(val_metric)

    # Generuj wykres
    title = f"Manual MLP - {dataset_name.replace('_', ' ').title()} ({split_name.replace('_', '/')})"
    visualizer.plot_learning_curves(history, title)

    # Dla klasyfikacji: confusion matrix
    if not is_regression:
        test_probs = model.forward(X_test, classification=True)
        test_preds = np.argmax(test_probs, axis=1)
        visualizer.plot_confusion_matrix(
            y_test,
            test_preds,
            title=f"Manual MLP - {dataset_name.replace('_', ' ').title()} (Test Set)"
        )

    # Dla regresji: scatter plot
    if is_regression:
        test_preds = model.forward(X_test)
        visualizer.plot_regression_scatter(
            y_test,
            test_preds,
            title=f"Manual MLP - {dataset_name.replace('_', ' ').title()} (Test Set)"
        )

    print(f"   ✅ Gotowe!")


def generate_learning_curves_keras(
    dataset_name: str,
    split_name: str,
    visualizer: ResultsVisualizer
):
    """Generuje learning curves dla modelu Keras."""
    print(f"\n📈 Generuję learning curves (Keras): {dataset_name} - {split_name}")

    # Określ ścieżki
    data_dir = os.path.join("data", dataset_name)
    results_file = os.path.join("results", f"keras_wyniki_{dataset_name}_{split_name}.xlsx")

    if not os.path.exists(results_file):
        print(f"⚠️  Plik nie istnieje: {results_file}")
        return

    # Wczytaj dane
    if split_name == "train_test":
        train_file, test_file, val_file = "train80.csv", "test20.csv", None
    else:
        train_file, test_file, val_file = "train70.csv", "test15.csv", "validation15.csv"

    X_train, y_train = load_data(os.path.join(data_dir, train_file))
    X_test, y_test = load_data(os.path.join(data_dir, test_file))
    X_val, y_val = None, None
    if val_file:
        X_val, y_val = load_data(os.path.join(data_dir, val_file))

    # Znajdź najlepszą konfigurację
    is_regression = "regression" in dataset_name
    metric = "test_mse" if is_regression else "test_accuracy"
    config, best_row = find_best_config(results_file, metric)

    # Określ task_type i n_outputs
    task_type = "regression" if is_regression else "classification"
    if is_regression:
        n_outputs = 1
    else:
        n_outputs = len(np.unique(y_train))

    # Trenuj model
    model = KerasMLPModel(
        n_inputs=X_train.shape[1],
        n_hidden_layers=config["n_hidden_layers"],
        n_neurons=config["n_neurons"],
        n_outputs=n_outputs,
        learning_rate=config["learning_rate"],
        task_type=task_type
    )

    print(f"   Trening modelu (może potrwać ~1 min)...")

    history = model.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=50,
        batch_size=32,
        verbose=0
    )

    # Generuj wykres
    title = f"Keras MLP - {dataset_name.replace('_', ' ').title()} ({split_name.replace('_', '/')})"
    visualizer.plot_learning_curves(history, title)

    # Dla klasyfikacji: confusion matrix
    if not is_regression:
        test_probs = model.predict(X_test)
        test_preds = np.argmax(test_probs, axis=1)
        visualizer.plot_confusion_matrix(
            y_test,
            test_preds,
            title=f"Keras MLP - {dataset_name.replace('_', ' ').title()} (Test Set)"
        )

    # Dla regresji: scatter plot
    if is_regression:
        test_preds = model.predict(X_test)
        visualizer.plot_regression_scatter(
            y_test,
            test_preds,
            title=f"Keras MLP - {dataset_name.replace('_', ' ').title()} (Test Set)"
        )

    print(f"   ✅ Gotowe!")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("GENEROWANIE WIZUALIZACJI DLA RAPORTU")
    print("="*80)

    visualizer = ResultsVisualizer()

    # Konfiguracja
    datasets = [
        "classification",
        "classification_our",
        "regression",
        "regression_our"
    ]

    splits = [
        "train_test",
        "train_val_test"
    ]

    # 1. Generuj learning curves dla manual
    print("\n" + "-"*80)
    print("CZĘŚĆ 1: Learning Curves - Manual MLP")
    print("-"*80)

    for dataset in datasets:
        for split in splits:
            try:
                generate_learning_curves_manual(dataset, split, visualizer)
            except Exception as e:
                print(f"❌ Błąd: {dataset} - {split}: {e}")

    # 2. Generuj learning curves dla Keras
    print("\n" + "-"*80)
    print("CZĘŚĆ 2: Learning Curves - Keras MLP")
    print("-"*80)

    for dataset in datasets:
        for split in splits:
            try:
                generate_learning_curves_keras(dataset, split, visualizer)
            except Exception as e:
                print(f"❌ Błąd: {dataset} - {split}: {e}")

    # 3. Generuj wykresy porównawcze
    print("\n" + "-"*80)
    print("CZĘŚĆ 3: Wykresy porównawcze Manual vs Keras")
    print("-"*80)

    visualizer.create_all_comparisons()

    print("\n" + "="*80)
    print("✅ WSZYSTKIE WIZUALIZACJE WYGENEROWANE!")
    print(f"📂 Zapisane w: {visualizer.output_dir}")
    print("="*80 + "\n")