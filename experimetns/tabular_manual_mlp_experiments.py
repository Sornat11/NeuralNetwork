import os
import time

import numpy as np
import pandas as pd

from src.manual_mlp.model import Model
from utils.experiment_runner import ExperimentRunner

def load_data(path: str):
    df = pd.read_csv(path)
    if df.columns[0].lower() == "datetime":
        df.set_index(df.columns[0], inplace=True)
    X = df.iloc[:, :-1].values
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = df.iloc[:, -1].values
    return X, y


def run_manual_mlp_experiments():
    DATASETS = [
        ("classification", False),
        ("classification_our", False),
        ("regression", True),
        ("regression_our", True),
    ]
    SPLITS = [
        ("train80.csv", "test20.csv", None),
        ("train70.csv", "test15.csv", "validation15.csv"),
    ]
    for folder, is_regression in DATASETS:
        data_dir = os.path.join("data", folder)
        if not os.path.isdir(data_dir):
            print(f"\n[INFO] Pomijam {folder} – brak katalogu {data_dir}")
            continue
        files = set(os.listdir(data_dir))
        for train_file, test_file, val_file in SPLITS:
            if train_file not in files or test_file not in files:
                continue
            print(
                f"\nUżywam własnej sieci (manual perceptron) dla: {folder}, "
                f"split: train={train_file}, test={test_file}, val={val_file}"
            )
            train_path = os.path.join(data_dir, train_file)
            test_path = os.path.join(data_dir, test_file)
            X_train, y_train = load_data(train_path)
            X_test, y_test = load_data(test_path)
            if is_regression:
                n_outputs = 1
            else:
                n_outputs = len(np.unique(y_train))
            hidden_layers_grid = [1, 2, 3, 4]
            neurons_grid = [8, 16, 32, 64]
            learning_rates_grid = [0.001, 0.005, 0.01, 0.02]
            param_grid = {
                "n_hidden_layers": hidden_layers_grid,
                "n_neurons": neurons_grid,
                "learning_rate": learning_rates_grid,
                "n_inputs": [X_train.shape[1]],
                "n_outputs": [n_outputs],
            }
            if val_file and val_file in files:
                val_path = os.path.join(data_dir, val_file)
                X_val, y_val = load_data(val_path)
                description = (
                    f"manual_perceptron_{folder}: podział train/val/test "
                    f"({train_file}, {val_file}, {test_file})"
                )
                output_file = f"manual_perceptron_wyniki_{folder}_train_val_test.xlsx"
                runner = ExperimentRunner(
                    model_class=Model,
                    X=X_train,
                    y=y_train,
                    param_grid=param_grid,
                    n_runs=5,
                    batch_size=32,
                    epochs=50,
                    description=description,
                    regression=is_regression,
                    output_file=output_file,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    y_test=y_test,
                )
            else:
                description = (
                    f"manual_perceptron_{folder}: podział train/test "
                    f"({train_file}, {test_file})"
                )
                output_file = f"manual_perceptron_wyniki_{folder}_train_test.xlsx"
                runner = ExperimentRunner(
                    model_class=Model,
                    X=X_train,
                    y=y_train,
                    param_grid=param_grid,
                    n_runs=5,
                    batch_size=32,
                    epochs=50,
                    description=description,
                    regression=is_regression,
                    output_file=output_file,
                    X_test=X_test,
                    y_test=y_test,
                )
            start_time = time.time()
            runner.run_all()
            elapsed = time.time() - start_time
            print(f"Czas trwania eksperymentu: {elapsed:.2f} sekundy")
