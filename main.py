import time

import pandas as pd
from sklearn.model_selection import train_test_split

from data.sample_data_generator import create_data, create_regression_data
from src.manual_mlp.model import Model
from utils.experiment_runner import ExperimentRunner


# Funkcja do wczytywania danych z CSV
def load_data(path):
    df = pd.read_csv(path)
    # Jeśli pierwsza kolumna to datetime, ustaw ją jako indeks
    if df.columns[0].lower() == "datetime":
        df.set_index(df.columns[0], inplace=True)
        X = df.iloc[:, :-1].values
    else:
        X = df.iloc[:, :-1].values
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = df.iloc[:, -1].values
    return X, y


if __name__ == "__main__":
    # --- REGRESJA klasyczna ---
    # Wczytaj dane regresyjne (train/test 80/20) z regression
    X_train_r_std, y_train_r_std = load_data("data/regression/train80.csv")
    X_test_r_std, y_test_r_std = load_data("data/regression/test20.csv")

    param_grid_reg_std = {
        "n_hidden_layers": [1, 2],
        "n_neurons": [8, 16],
        "learning_rate": [0.01, 0.001],
        "n_inputs": [X_train_r_std.shape[1]],
        "n_outputs": [1],
    }

    print("--- Regresja klasyczna: podział 80/20 (train/test) ---")
    start_time_r_std = time.time()
    runner_reg_std = ExperimentRunner(
        model_class=Model,
        X=X_train_r_std,
        y=y_train_r_std,
        param_grid=param_grid_reg_std,
        n_runs=3,
        batch_size=32,
        epochs=50,
        description="manual_perceptron_regresja: podział 80/20 (train/test)",
        regression=True,
        output_file="manual_perceptron_wyniki_regresja_80_20.xlsx",
        X_test=X_test_r_std,
        y_test=y_test_r_std,
    )
    runner_reg_std.run_all()
    elapsed_r_std = time.time() - start_time_r_std
    print(
        f"Czas trwania eksperymentu regresji klasycznej (train/test): {elapsed_r_std:.2f} sekundy"
    )

    # Wczytaj dane regresyjne (train/val/test 70/15/15) z regression
    X_train_r_std2, y_train_r_std2 = load_data("data/regression/train70.csv")
    X_val_r_std, y_val_r_std = load_data("data/regression/validation15.csv")
    X_test_r_std2, y_test_r_std2 = load_data("data/regression/test15.csv")

    print("--- Regresja klasyczna: podział 70/15/15 (train/val/test) ---")
    start_time_r_std2 = time.time()
    runner_reg_std2 = ExperimentRunner(
        model_class=Model,
        X=X_train_r_std2,
        y=y_train_r_std2,
        param_grid=param_grid_reg_std,
        n_runs=3,
        batch_size=32,
        epochs=50,
        description="manual_perceptron_regresja: podział 70/15/15 (train/val/test)",
        regression=True,
        output_file="manual_perceptron_wyniki_regresja_70_15_15.xlsx",
        X_val=X_val_r_std,
        y_val=y_val_r_std,
        X_test=X_test_r_std2,
        y_test=y_test_r_std2,
    )
    runner_reg_std2.run_all()
    elapsed_r_std2 = time.time() - start_time_r_std2
    print(
        f"Czas trwania eksperymentu regresji klasycznej (train/val/test): {elapsed_r_std2:.2f} sekundy"
    )
    # --- REGRESJA ---
    # Wczytaj dane regresyjne (train/test 80/20) z regression_our
    X_train_r, y_train_r = load_data("data/regression_our/train80.csv")
    X_test_r, y_test_r = load_data("data/regression_our/test20.csv")

    param_grid_reg = {
        "n_hidden_layers": [1, 2],
        "n_neurons": [8, 16],
        "learning_rate": [0.01, 0.001],  # tylko bezpieczne wartości
        "n_inputs": [X_train_r.shape[1]],
        "n_outputs": [1],
    }

    print("--- Regresja: podział 80/20 (train/test) ---")
    start_time_r = time.time()
    runner_reg = ExperimentRunner(
        model_class=Model,
        X=X_train_r,
        y=y_train_r,
        param_grid=param_grid_reg,
        n_runs=3,
        batch_size=32,
        epochs=50,
        description="manual_perceptron_regresja (our): podział 80/20 (train/test)",
        regression=True,
        output_file="manual_perceptron_wyniki_regresja_our_80_20.xlsx",
        X_test=X_test_r,
        y_test=y_test_r,
    )
    runner_reg.run_all()
    elapsed_r = time.time() - start_time_r
    print(f"Czas trwania eksperymentu regresji (train/test): {elapsed_r:.2f} sekundy")

    # Wczytaj dane regresyjne (train/val/test 70/15/15) z regression_our
    X_train_r2, y_train_r2 = load_data("data/regression_our/train70.csv")
    X_val_r, y_val_r = load_data("data/regression_our/validation15.csv")
    X_test_r2, y_test_r2 = load_data("data/regression_our/test15.csv")

    print("--- Regresja: podział 70/15/15 (train/val/test) ---")
    start_time_r2 = time.time()
    runner_reg2 = ExperimentRunner(
        model_class=Model,
        X=X_train_r2,
        y=y_train_r2,
        param_grid=param_grid_reg,
        n_runs=3,
        batch_size=32,
        epochs=50,
        description="manual_perceptron_regresja (our): podział 70/15/15 (train/val/test)",
        regression=True,
        output_file="manual_perceptron_wyniki_regresja_our_70_15_15.xlsx",
        X_val=X_val_r,
        y_val=y_val_r,
        X_test=X_test_r2,
        y_test=y_test_r2,
    )
    runner_reg2.run_all()
    elapsed_r2 = time.time() - start_time_r2
    print(
        f"Czas trwania eksperymentu regresji (train/val/test): {elapsed_r2:.2f} sekundy"
    )

    def load_data(path):
        df = pd.read_csv(path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        return X, y

    # Klasyfikacja – podział 80/20 (train/test)
    X_train, y_train = load_data("data/classification/train80.csv")
    X_test, y_test = load_data("data/classification/test20.csv")

    param_grid = {
        "n_hidden_layers": [1, 2],
        "n_neurons": [8, 16],
        "learning_rate": [0.01, 0.05],
        "n_inputs": [X_train.shape[1]],
        "n_outputs": [2],  # liczba klas (0/1) – jeśli softmax, ustaw 2
    }

    print("--- Klasyfikacja: podział 80/20 (train/test) ---")
    start_time = time.time()
    runner = ExperimentRunner(
        model_class=Model,
        X=X_train,
        y=y_train,
        param_grid=param_grid,
        n_runs=3,
        batch_size=32,
        epochs=50,
        description="manual_perceptron_klasyfikacja: podział 80/20 (train/test)",
        regression=False,
        output_file="manual_perceptron_wyniki_klasyfikacja_80_20.xlsx",
        X_test=X_test,
        y_test=y_test,
    )
    runner.run_all()
    elapsed = time.time() - start_time
    print(f"Czas trwania eksperymentu (train/test): {elapsed:.2f} sekundy")

    # Klasyfikacja – podział 70/15/15 (train/val/test)
    X_train2, y_train2 = load_data("data/classification/train70.csv")
    X_val, y_val = load_data("data/classification/validation15.csv")
    X_test2, y_test2 = load_data("data/classification/test15.csv")

    print("--- Klasyfikacja: podział 70/15/15 (train/val/test) ---")
    start_time2 = time.time()
    runner2 = ExperimentRunner(
        model_class=Model,
        X=X_train2,
        y=y_train2,
        param_grid=param_grid,
        n_runs=3,
        batch_size=32,
        epochs=50,
        description="manual_perceptron_klasyfikacja: podział 70/15/15 (train/val/test)",
        regression=False,
        output_file="manual_perceptron_wyniki_klasyfikacja_70_15_15.xlsx",
        X_val=X_val,
        y_val=y_val,
        X_test=X_test2,
        y_test=y_test2,
    )
    runner2.run_all()
    elapsed2 = time.time() - start_time2
    print(f"Czas trwania eksperymentu (train/val/test): {elapsed2:.2f} sekundy")
