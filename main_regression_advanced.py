"""
Zaawansowane modele dla regresji (szeregi czasowe).
Uruchamia CNN 1D i LSTM dla Stock Market dataset.
"""

import os
import time
import numpy as np
import pandas as pd

from src.models.keras_cnn_regression import KerasCNN1DRegression
from src.models.keras_lstm_regression import KerasLSTMRegression
from src.manual_mlp.metrics import ModelMetrics
from utils.results_exporter import ResultsExporter


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


if __name__ == "__main__":
    print("\n" + "="*80)
    print("ADVANCED REGRESSION MODELS (CNN 1D & LSTM)")
    print("Dataset: Stock Market (Time Series)")
    print("="*80)

    # Wczytaj dane Stock Market
    data_dir = "data/regression"

    X_train, y_train = load_data(os.path.join(data_dir, "train70.csv"))
    X_val, y_val = load_data(os.path.join(data_dir, "validation15.csv"))
    X_test, y_test = load_data(os.path.join(data_dir, "test15.csv"))

    print(f"\nStock Market data loaded:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")

    n_features = X_train.shape[1]

    # Grid hiperparametrów
    # UWAGA: Stock Market ma tylko 5 cech, więc max 2 warstwy Conv (z pooling)
    CONV_LAYERS_GRID = [1, 2]  # Max 2 warstwy (5 -> 2 -> 1)
    FILTERS_GRID = [32, 64]
    LSTM_LAYERS_GRID = [1, 2]
    LSTM_UNITS_GRID = [32, 64]
    LEARNING_RATES_GRID = [0.001, 0.0001]
    OPTIMIZERS_GRID = ["adam", "rmsprop"]

    metrics_calc = ModelMetrics()

    # ========================================================================
    # 1. CNN 1D FOR REGRESSION
    # ========================================================================
    print("\n" + "-"*80)
    print("EXPERIMENT 1: CNN 1D for Time Series Regression")
    print("-"*80)

    cnn_results = []
    total_cnn = len(CONV_LAYERS_GRID) * len(FILTERS_GRID) * len(LEARNING_RATES_GRID) * len(OPTIMIZERS_GRID)
    total_cnn_exp = total_cnn * 5  # 5 runs

    print(f"\nCNN 1D Grid:")
    print(f"  Conv layers: {CONV_LAYERS_GRID}")
    print(f"  Filters: {FILTERS_GRID}")
    print(f"  Learning rates: {LEARNING_RATES_GRID}")
    print(f"  Optimizers: {OPTIMIZERS_GRID}")
    print(f"  Total: {total_cnn} combinations × 5 runs = {total_cnn_exp} experiments")

    exp_idx = 0
    for n_conv in CONV_LAYERS_GRID:
        for n_filters in FILTERS_GRID:
            for lr in LEARNING_RATES_GRID:
                for opt in OPTIMIZERS_GRID:
                    for run in range(5):
                        exp_idx += 1

                        print(f"\n[{exp_idx}/{total_cnn_exp}] Conv={n_conv}, Filters={n_filters}, LR={lr}, Opt={opt}, Run={run+1}")

                        # Stwórz model
                        model = KerasCNN1DRegression(
                            n_features=n_features,
                            n_conv_layers=n_conv,
                            n_filters=n_filters,
                            kernel_size=3,
                            pool_size=2,
                            n_dense_neurons=64,
                            learning_rate=lr,
                            optimizer_name=opt
                        )

                        # Trenuj
                        history = model.train(
                            X_train=X_train,
                            y_train=y_train,
                            X_val=X_val,
                            y_val=y_val,
                            epochs=50,
                            batch_size=32,
                            verbose=0
                        )

                        # Ewaluacja
                        test_mse, test_mae = model.evaluate(X_test, y_test)

                        # R² score
                        test_preds = model.predict(X_test)
                        test_r2 = float(metrics_calc.r2_score(test_preds, y_test.reshape(-1, 1)))

                        # Zapisz wynik
                        result = {
                            "n_conv_layers": n_conv,
                            "n_filters": n_filters,
                            "learning_rate": lr,
                            "optimizer": opt,
                            "run": run + 1,
                            "mse": history["loss"][-1],
                            "mae": history["mae"][-1],
                            "val_mse": history["val_loss"][-1],
                            "val_mae": history["val_mae"][-1],
                            "test_mse": test_mse,
                            "test_mae": test_mae,
                            "test_r2": test_r2,
                        }

                        cnn_results.append(result)
                        print(f"  Train MSE: {result['mse']:.4f}, Val MSE: {result['val_mse']:.4f}, Test MSE: {test_mse:.4f}")

    # Wybierz najlepsze runy dla CNN
    best_cnn = {}
    for result in cnn_results:
        key = (result["n_conv_layers"], result["n_filters"], result["learning_rate"], result["optimizer"])
        if key not in best_cnn or result["val_mse"] < best_cnn[key]["val_mse"]:
            best_cnn[key] = result

    best_cnn_list = list(best_cnn.values())

    # Eksport CNN
    exporter_cnn = ResultsExporter("results/keras_cnn1d_wyniki_regression.xlsx")
    cnn_dict = {key: [r[key] for r in best_cnn_list] for key in best_cnn_list[0].keys()}

    exporter_cnn.export(
        cnn_dict,
        params_dict={
            "n_conv_layers": [r["n_conv_layers"] for r in best_cnn_list],
            "n_filters": [r["n_filters"] for r in best_cnn_list],
            "learning_rate": [r["learning_rate"] for r in best_cnn_list],
            "optimizer": [r["optimizer"] for r in best_cnn_list],
        },
        description="CNN 1D - Stock Market Regression"
    )

    print(f"\n✅ CNN 1D zakończone! Wyniki: results/keras_cnn1d_wyniki_regression.xlsx")

    # ========================================================================
    # 2. LSTM FOR REGRESSION
    # ========================================================================
    print("\n" + "-"*80)
    print("EXPERIMENT 2: LSTM for Time Series Regression")
    print("-"*80)

    lstm_results = []
    total_lstm = len(LSTM_LAYERS_GRID) * len(LSTM_UNITS_GRID) * len(LEARNING_RATES_GRID) * len(OPTIMIZERS_GRID)
    total_lstm_exp = total_lstm * 5  # 5 runs

    print(f"\nLSTM Grid:")
    print(f"  LSTM layers: {LSTM_LAYERS_GRID}")
    print(f"  LSTM units: {LSTM_UNITS_GRID}")
    print(f"  Learning rates: {LEARNING_RATES_GRID}")
    print(f"  Optimizers: {OPTIMIZERS_GRID}")
    print(f"  Total: {total_lstm} combinations × 5 runs = {total_lstm_exp} experiments")

    exp_idx = 0
    for n_lstm in LSTM_LAYERS_GRID:
        for n_units in LSTM_UNITS_GRID:
            for lr in LEARNING_RATES_GRID:
                for opt in OPTIMIZERS_GRID:
                    for run in range(5):
                        exp_idx += 1

                        print(f"\n[{exp_idx}/{total_lstm_exp}] LSTM={n_lstm}, Units={n_units}, LR={lr}, Opt={opt}, Run={run+1}")

                        # Stwórz model
                        model = KerasLSTMRegression(
                            n_features=n_features,
                            n_lstm_layers=n_lstm,
                            n_lstm_units=n_units,
                            n_dense_neurons=32,
                            learning_rate=lr,
                            optimizer_name=opt,
                            dropout=0.0  # Bez dropout dla uproszczenia
                        )

                        # Trenuj
                        history = model.train(
                            X_train=X_train,
                            y_train=y_train,
                            X_val=X_val,
                            y_val=y_val,
                            epochs=50,
                            batch_size=32,
                            verbose=0
                        )

                        # Ewaluacja
                        test_mse, test_mae = model.evaluate(X_test, y_test)

                        # R² score
                        test_preds = model.predict(X_test)
                        test_r2 = float(metrics_calc.r2_score(test_preds, y_test.reshape(-1, 1)))

                        # Zapisz wynik
                        result = {
                            "n_lstm_layers": n_lstm,
                            "n_lstm_units": n_units,
                            "learning_rate": lr,
                            "optimizer": opt,
                            "run": run + 1,
                            "mse": history["loss"][-1],
                            "mae": history["mae"][-1],
                            "val_mse": history["val_loss"][-1],
                            "val_mae": history["val_mae"][-1],
                            "test_mse": test_mse,
                            "test_mae": test_mae,
                            "test_r2": test_r2,
                        }

                        lstm_results.append(result)
                        print(f"  Train MSE: {result['mse']:.4f}, Val MSE: {result['val_mse']:.4f}, Test MSE: {test_mse:.4f}")

    # Wybierz najlepsze runy dla LSTM
    best_lstm = {}
    for result in lstm_results:
        key = (result["n_lstm_layers"], result["n_lstm_units"], result["learning_rate"], result["optimizer"])
        if key not in best_lstm or result["val_mse"] < best_lstm[key]["val_mse"]:
            best_lstm[key] = result

    best_lstm_list = list(best_lstm.values())

    # Eksport LSTM
    exporter_lstm = ResultsExporter("results/keras_lstm_wyniki_regression.xlsx")
    lstm_dict = {key: [r[key] for r in best_lstm_list] for key in best_lstm_list[0].keys()}

    exporter_lstm.export(
        lstm_dict,
        params_dict={
            "n_lstm_layers": [r["n_lstm_layers"] for r in best_lstm_list],
            "n_lstm_units": [r["n_lstm_units"] for r in best_lstm_list],
            "learning_rate": [r["learning_rate"] for r in best_lstm_list],
            "optimizer": [r["optimizer"] for r in best_lstm_list],
        },
        description="LSTM - Stock Market Regression"
    )

    print(f"\n✅ LSTM zakończone! Wyniki: results/keras_lstm_wyniki_regression.xlsx")

    # ========================================================================
    # PODSUMOWANIE
    # ========================================================================
    print("\n" + "="*80)
    print("✅ WSZYSTKIE ZAAWANSOWANE MODELE REGRESJI ZAKOŃCZONE!")
    print("="*80)
    print("\nWygenerowane pliki:")
    print("  1. results/keras_cnn1d_wyniki_regression.xlsx")
    print("  2. results/keras_lstm_wyniki_regression.xlsx")
    print("\nModele do porównania dla Stock Market:")
    print("  - Manual MLP (już uruchomione)")
    print("  - Keras MLP (już uruchomione)")
    print("  - CNN 1D (nowe)")
    print("  - LSTM (nowe)")
    print("\n")
