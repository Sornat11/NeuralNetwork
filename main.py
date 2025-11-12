import numpy as np

from data.sample_data_generator import create_data, create_regression_data
from src.manual_mlp.model import Model
from utils.experiment_runner import ExperimentRunner

if __name__ == "__main__":
    # Przykładowa siatka hiperparametrów dla klasyfikacji
    param_grid_class = {
        "n_hidden_layers": [1, 2],
        "n_neurons": [8, 16],
        "learning_rate": [0.01, 0.05],
        "points": [100],
        "n_inputs": [2],
        "n_outputs": [3],
    }
    X_class, y_class = create_data(points=100, classes=3)

    runner_class = ExperimentRunner(
        Model,
        X_class,
        y_class,
        param_grid_class,
        n_runs=3,
        epochs=100,
        description="Klasyfikacja: wpływ hiperparametrów na MLP",
        regression=False,
        output_file="wyniki_klasyfikacja.xlsx",
    )
    runner_class.run_all()

    # Przykładowa siatka hiperparametrów dla regresji
    param_grid_reg = {
        "n_hidden_layers": [1, 2],
        "n_neurons": [8, 16],
        "learning_rate": [0.01, 0.05],
        "points": [100],
        "n_inputs": [2],
        "n_outputs": [1],
    }
    X_reg, y_reg = create_regression_data(points=100, features=2, noise=0.1)
    runner_reg = ExperimentRunner(
        Model,
        X_reg,
        y_reg,
        param_grid_reg,
        n_runs=3,
        epochs=100,
        description="Regresja: wpływ hiperparametrów na MLP",
        regression=True,
        output_file="wyniki_regresja.xlsx",
    )
    runner_reg.run_all()
