"""
Główny skrypt do uruchamiania eksperymentów z modelami Keras/TensorFlow.
Analogiczny do main.py, ale używa KerasMLPModel zamiast ręcznej implementacji.
"""

import os
import time

import numpy as np
import pandas as pd

from utils.keras_experiment_runner import KerasExperimentRunner


def load_data(path: str):
    """
    Wczytuje dane z pliku CSV.

    Args:
        path: Ścieżka do pliku CSV

    Returns:
        (X, y) - dane wejściowe i etykiety
    """
    df = pd.read_csv(path)

    # Jeśli pierwsza kolumna to datetime, ustaw ją jako indeks
    if df.columns[0].lower() == "datetime":
        df.set_index(df.columns[0], inplace=True)

    X = df.iloc[:, :-1].values
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    y = df.iloc[:, -1].values
    return X, y


def run_tabular_keras_experiments():
    # Konfiguracja zbiorów danych
    # (folder, is_regression)
    DATASETS = [
        ("classification", False),
        ("classification_our", False),
        ("regression", True),
        ("regression_our", True),
    ]

    # Konfiguracja podziałów danych
    # (train_file, test_file, val_file)
    # val_file == None -> tylko train/test
    SPLITS = [
        ("train80.csv", "test20.csv", None),
        ("train70.csv", "test15.csv", "validation15.csv"),
    ]

    # Grid hiperparametrów (taki sam jak dla ręcznej implementacji)
    HIDDEN_LAYERS_GRID = [1, 2, 3, 4]
    NEURONS_GRID = [8, 16, 32, 64]
    LEARNING_RATES_GRID = [0.001, 0.005, 0.01, 0.02]

    print("\n" + "=" * 80)
    print("EKSPERYMENTY KERAS/TENSORFLOW")
    print("=" * 80)

    for folder, is_regression in DATASETS:
        data_dir = os.path.join("data", folder)
        if not os.path.isdir(data_dir):
            print(f"\n[INFO] Pomijam {folder} – brak katalogu {data_dir}")
            continue

        files = set(os.listdir(data_dir))

        for train_file, test_file, val_file in SPLITS:
            # Sprawdź, czy dla danego splitu są odpowiednie pliki
            if train_file not in files or test_file not in files:
                continue

            print("\n" + "-" * 80)
            print(
                f"Dataset: {folder} | "
                f"Split: train={train_file}, test={test_file}, val={val_file}"
            )
            print("-" * 80)

            # Wczytaj dane treningowe i testowe
            train_path = os.path.join(data_dir, train_file)
            test_path = os.path.join(data_dir, test_file)

            X_train, y_train = load_data(train_path)
            X_test, y_test = load_data(test_path)

            # Wczytaj dane walidacyjne (jeśli istnieją)
            X_val, y_val = None, None
            if val_file and val_file in files:
                val_path = os.path.join(data_dir, val_file)
                X_val, y_val = load_data(val_path)

            # Przygotuj grid hiperparametrów
            param_grid = {
                "n_hidden_layers": HIDDEN_LAYERS_GRID,
                "n_neurons": NEURONS_GRID,
                "learning_rate": LEARNING_RATES_GRID,
            }

            # Określ task_type i nazwę pliku wyjściowego
            task_type = "regression" if is_regression else "classification"

            if val_file and val_file in files:
                description = (
                    f"keras_{folder}: podział train/val/test "
                    f"({train_file}, {val_file}, {test_file})"
                )
                output_file = os.path.join(
                    "results", f"keras_wyniki_{folder}_train_val_test.xlsx"
                )
            else:
                description = (
                    f"keras_{folder}: podział train/test "
                    f"({train_file}, {test_file})"
                )
                output_file = os.path.join(
                    "results", f"keras_wyniki_{folder}_train_test.xlsx"
                )

            # Utwórz runner eksperymentów
            runner = KerasExperimentRunner(
                X_train=X_train,
                y_train=y_train,
                param_grid=param_grid,
                n_runs=5,  # Min. 5 powtórzeń zgodnie z wymaganiami projektu
                batch_size=32,
                epochs=50,
                description=description,
                task_type=task_type,
                output_file=output_file,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
            )

            # Uruchom eksperymenty
            print(f"\n🚀 Rozpoczynam eksperymenty dla: {folder}")
            start_time = time.time()

            try:
                runner.run_all()
                elapsed = time.time() - start_time
                print(f"\n✅ Zakończono w {elapsed:.2f} sekund")
            except Exception as e:
                print(f"\n❌ Błąd podczas eksperymentu: {e}")
                import traceback

                traceback.print_exc()

    print("\n" + "=" * 80)
    print("✅ WSZYSTKIE EKSPERYMENTY KERAS ZAKOŃCZONE")
    print("=" * 80 + "\n")
