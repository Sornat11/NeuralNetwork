"""
Główny skrypt do eksperymentów na Fashion MNIST.
Uruchamia 3 typy modeli:
1. Manual MLP (własna implementacja)
2. Keras MLP
3. Keras CNN
"""

import os
import time
import numpy as np

from src.manual_mlp.model import Model as ManualModel
from src.models.keras_mlp import KerasMLPModel
from src.models.keras_cnn import KerasCNNModel
from utils.experiment_runner import ExperimentRunner
from utils.keras_experiment_runner import KerasExperimentRunner


def load_fashion_mnist_data():
    """Wczytuje przetworzone dane Fashion MNIST."""
    data_dir = "data/fashion_mist"

    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    print(f"Fashion MNIST data loaded:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def run_fashion_mnist_experiments():
    print("\n" + "="*80)
    print("FASHION MNIST EXPERIMENTS")
    print("="*80)

    # Wczytaj dane
    X_train, y_train, X_val, y_val, X_test, y_test = load_fashion_mnist_data()

    # Grid hiperparametrów
    # Dla obrazów używamy mniejszego gridu bo CNN jest wolniejszy
    HIDDEN_LAYERS_GRID = [2, 3]  # Mniej warstw niż dla tabular data
    NEURONS_GRID = [64, 128]  # Więcej neuronów bo obrazy są bardziej złożone
    LEARNING_RATES_GRID = [0.001, 0.0001]  # Mniejsze LR dla obrazów
    OPTIMIZERS_GRID = ["sgd", "adam"]  # Testujemy różne optymalizatory

    # CNN-specific parameters
    CONV_LAYERS_GRID = [2, 3]
    FILTERS_GRID = [[32, 64, 128], [64, 128, 256]]  # 3 wartości (dla max 3 warstw)

    # ========================================================================
    # 1. MANUAL MLP
    # ========================================================================
    print("\n" + "-"*80)
    print("EXPERIMENT 1: Manual MLP")
    print("-"*80)

    param_grid_manual = {
        "n_hidden_layers": HIDDEN_LAYERS_GRID,
        "n_neurons": NEURONS_GRID,
        "learning_rate": LEARNING_RATES_GRID,
        "n_inputs": [784],  # 28x28 pixels
        "n_outputs": [10],  # 10 classes
    }

    runner_manual = ExperimentRunner(
        model_class=ManualModel,
        X=X_train,
        y=y_train,
        param_grid=param_grid_manual,
        n_runs=3,  # Zmniejszone z 5 dla szybkości
        batch_size=256,  # Większy batch = szybsze trenowanie
        epochs=15,  # Zmniejszone z 20
        description="Manual MLP - Fashion MNIST",
        regression=False,
        output_file="results/manual_perceptron_wyniki_fashion_mnist.xlsx",
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    print("\n🚀 Rozpoczynam eksperymenty Manual MLP...")
    start_time = time.time()

    try:
        runner_manual.run_all()
        elapsed = time.time() - start_time
        print(f"\n✅ Manual MLP zakończone w {elapsed/60:.1f} minut")
    except Exception as e:
        print(f"\n❌ Błąd Manual MLP: {e}")
        import traceback
        traceback.print_exc()

    # ========================================================================
    # 2. KERAS MLP
    # ========================================================================
    print("\n" + "-"*80)
    print("EXPERIMENT 2: Keras MLP")
    print("-"*80)

    param_grid_mlp = {
        "n_hidden_layers": HIDDEN_LAYERS_GRID,
        "n_neurons": NEURONS_GRID,
        "learning_rate": LEARNING_RATES_GRID,
        "optimizer_name": OPTIMIZERS_GRID,  # Testujemy różne optymalizatory
    }

    runner_mlp = KerasExperimentRunner(
        X_train=X_train,
        y_train=y_train,
        param_grid=param_grid_mlp,
        n_runs=3,  # Zmniejszone z 5
        batch_size=256,  # Większy batch
        epochs=15,  # Zmniejszone z 20
        description="Keras MLP - Fashion MNIST",
        task_type="classification",
        output_file="results/keras_mlp_wyniki_fashion_mnist.xlsx",
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    print("\n🚀 Rozpoczynam eksperymenty Keras MLP...")
    start_time = time.time()

    try:
        runner_mlp.run_all()
        elapsed = time.time() - start_time
        print(f"\n✅ Keras MLP zakończone w {elapsed/60:.1f} minut")
    except Exception as e:
        print(f"\n❌ Błąd Keras MLP: {e}")
        import traceback
        traceback.print_exc()

    # ========================================================================
    # 3. KERAS CNN
    # ========================================================================
    print("\n" + "-"*80)
    print("EXPERIMENT 3: Keras CNN")
    print("-"*80)

    # CNN potrzebuje własnego runnera bo ma inne parametry
    # Tworzymy uproszczoną wersję grid search dla CNN

    cnn_results = []
    total_combinations = len(CONV_LAYERS_GRID) * len(FILTERS_GRID) * len(LEARNING_RATES_GRID) * len(OPTIMIZERS_GRID)
    total_experiments = total_combinations * 3  # 3 runs (zmniejszone z 5)

    print(f"\nCNN Grid Search:")
    print(f"  Conv layers: {CONV_LAYERS_GRID}")
    print(f"  Filters: {FILTERS_GRID}")
    print(f"  Learning rates: {LEARNING_RATES_GRID}")
    print(f"  Optimizers: {OPTIMIZERS_GRID}")
    print(f"  Total: {total_combinations} combinations × 3 runs = {total_experiments} experiments")

    from tqdm import tqdm, trange

    exp_idx = 0
    for n_conv in CONV_LAYERS_GRID:
        for filters in FILTERS_GRID:
            for lr in LEARNING_RATES_GRID:
                for opt in OPTIMIZERS_GRID:
                    # 3 runy dla każdej kombinacji (zmniejszone z 5)
                    for run in range(3):
                        exp_idx += 1

                        # Stwórz model
                        model = KerasCNNModel(
                            input_shape=(28, 28, 1),
                            n_conv_layers=n_conv,
                            n_filters=filters[:n_conv],  # Weź tylko odpowiednią liczbę filtrów
                            kernel_size=3,
                            pool_size=2,
                            n_dense_layers=1,
                            n_dense_neurons=128,
                            n_outputs=10,
                            learning_rate=lr,
                            optimizer_name=opt
                        )

                        # Trenuj
                        print(f"\n[{exp_idx}/{total_experiments}] Conv={n_conv}, Filters={filters[:n_conv]}, LR={lr}, Opt={opt}, Run={run+1}")

                        history = model.train(
                            X_train=X_train,
                            y_train=y_train,
                            X_val=X_val,
                            y_val=y_val,
                            epochs=15,  # Zmniejszone z 20
                            batch_size=256,  # Większy batch
                            verbose=0
                        )

                        # Ewaluacja
                        test_loss, test_acc = model.evaluate(X_test, y_test)

                        # Dodatkowe metryki
                        test_probs = model.predict(X_test)
                        from src.manual_mlp.metrics import ModelMetrics
                        metrics_calc = ModelMetrics()

                        test_precision = float(metrics_calc.precision(test_probs, y_test))
                        test_recall = float(metrics_calc.recall(test_probs, y_test))
                        test_f1 = float(metrics_calc.f1_score(test_probs, y_test))

                        # Zapisz wynik
                        result = {
                            "n_conv_layers": n_conv,
                            "filters": str(filters[:n_conv]),
                            "learning_rate": lr,
                            "optimizer": opt,
                            "run": run + 1,
                            "loss": history["loss"][-1],
                            "accuracy": history["accuracy"][-1],
                            "val_loss": history["val_loss"][-1],
                            "val_accuracy": history["val_accuracy"][-1],
                            "test_loss": test_loss,
                            "test_accuracy": test_acc,
                            "test_precision": test_precision,
                            "test_recall": test_recall,
                            "test_f1_score": test_f1,
                        }

                        cnn_results.append(result)

                        print(f"  Train acc: {result['accuracy']:.4f}, Val acc: {result['val_accuracy']:.4f}, Test acc: {test_acc:.4f}")

    # Zapisz wyniki CNN do Excel
    import pandas as pd
    from utils.results_exporter import ResultsExporter

    # Wybierz najlepsze runy
    best_cnn_results = {}
    for result in cnn_results:
        key = (result["n_conv_layers"], result["filters"], result["learning_rate"], result["optimizer"])
        if key not in best_cnn_results or result["val_accuracy"] > best_cnn_results[key]["val_accuracy"]:
            best_cnn_results[key] = result

    best_cnn_list = list(best_cnn_results.values())

    # Eksport
    exporter = ResultsExporter("results/keras_cnn_wyniki_fashion_mnist.xlsx")
    results_dict = {
        key: [r[key] for r in best_cnn_list]
        for key in best_cnn_list[0].keys()
    }

    exporter.export(
        results_dict,
        params_dict={
            "n_conv_layers": [r["n_conv_layers"] for r in best_cnn_list],
            "filters": [r["filters"] for r in best_cnn_list],
            "learning_rate": [r["learning_rate"] for r in best_cnn_list],
            "optimizer": [r["optimizer"] for r in best_cnn_list],
        },
        description="Keras CNN - Fashion MNIST"
    )

    print(f"\n✅ CNN zakończone! Wyniki zapisane do: results/keras_cnn_wyniki_fashion_mnist.xlsx")

    # ========================================================================
    # PODSUMOWANIE
    # ========================================================================
    print("\n" + "="*80)
    print("✅ WSZYSTKIE EKSPERYMENTY FASHION MNIST ZAKOŃCZONE!")
    print("="*80)
    print("\nWygenerowane pliki:")
    print("  1. results/manual_perceptron_wyniki_fashion_mnist.xlsx")
    print("  2. results/keras_mlp_wyniki_fashion_mnist.xlsx")
    print("  3. results/keras_cnn_wyniki_fashion_mnist.xlsx")
    print("\n")
