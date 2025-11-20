"""
ROZSZERZONY skrypt do generowania wizualizacji.
Obsługuje wszystkie modele: MLP, CNN, LSTM dla wszystkich zbiorów danych.
"""

import os
import numpy as np
import pandas as pd

from visualizations.visualization import ResultsVisualizer
from src.models.keras_mlp import KerasMLPModel
from src.models.keras_cnn import KerasCNNModel
from src.models.keras_cnn_regression import KerasCNN1DRegression
from src.models.keras_lstm_regression import KerasLSTMRegression


def load_data_csv(path: str):
    """Wczytuje dane z CSV."""
    df = pd.read_csv(path)
    if df.columns[0].lower() == "datetime":
        df.set_index(df.columns[0], inplace=True)
    X = df.iloc[:, :-1].values
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = df.iloc[:, -1].values
    return X, y


def load_fashion_mnist():
    """Wczytuje Fashion MNIST z .npy."""
    data_dir = "data/fashion_mist"
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    return X_train, y_train, X_val, y_val, X_test, y_test


def generate_fashion_mnist_visualizations(visualizer: ResultsVisualizer):
    """Generuje wizualizacje dla Fashion MNIST (MLP + CNN)."""
    print("\n" + "="*80)
    print("FASHION MNIST VISUALIZATIONS")
    print("="*80)

    # Wczytaj dane
    X_train, y_train, X_val, y_val, X_test, y_test = load_fashion_mnist()

    # === 1. Keras MLP ===
    results_mlp = "results/keras_mlp_wyniki_fashion_mnist.xlsx"
    if os.path.exists(results_mlp):
        print("\n📈 Keras MLP - Fashion MNIST")

        df = pd.read_excel(results_mlp, sheet_name="Results")
        best_idx = df["test_accuracy"].idxmax()
        best = df.loc[best_idx]

        # Trenuj najlepszy model
        model = KerasMLPModel(
            n_inputs=784,
            n_hidden_layers=int(best["n_hidden_layers"]),
            n_neurons=int(best["n_neurons"]),
            n_outputs=10,
            learning_rate=float(best["learning_rate"]),
            task_type="classification",
            optimizer_name=str(best["optimizer_name"])
        )

        history = model.train(X_train, y_train, X_val, y_val, epochs=20, batch_size=128, verbose=0)

        # Learning curves
        visualizer.plot_learning_curves(
            history,
            "Keras MLP - Fashion MNIST",
            "results/visualizations/Fashion_MNIST_Keras_MLP_learning_curves.png"
        )

        # Confusion matrix
        test_probs = model.predict(X_test)
        visualizer.plot_confusion_matrix(
            y_test,
            test_probs,
            title="Keras MLP - Fashion MNIST (Test Set)",
            save_path="results/visualizations/Fashion_MNIST_Keras_MLP_confusion_matrix.png"
        )

    # === 2. Keras CNN ===
    results_cnn = "results/keras_cnn_wyniki_fashion_mnist.xlsx"
    if os.path.exists(results_cnn):
        print("\n📈 Keras CNN - Fashion MNIST")

        df = pd.read_excel(results_cnn, sheet_name="Results")
        best_idx = df["test_accuracy"].idxmax()
        best = df.loc[best_idx]

        # Parse filters string
        import ast
        filters_str = best["filters"]
        filters = ast.literal_eval(filters_str) if isinstance(filters_str, str) else filters_str

        # Trenuj najlepszy model
        model = KerasCNNModel(
            input_shape=(28, 28, 1),
            n_conv_layers=int(best["n_conv_layers"]),
            n_filters=filters,
            kernel_size=3,
            pool_size=2,
            n_dense_layers=1,
            n_dense_neurons=128,
            n_outputs=10,
            learning_rate=float(best["learning_rate"]),
            optimizer_name=str(best["optimizer"])
        )

        history = model.train(X_train, y_train, X_val, y_val, epochs=20, batch_size=128, verbose=0)

        # Learning curves
        visualizer.plot_learning_curves(
            history,
            "Keras CNN - Fashion MNIST",
            "results/visualizations/Fashion_MNIST_Keras_CNN_learning_curves.png"
        )

        # Confusion matrix
        test_probs = model.predict(X_test)
        visualizer.plot_confusion_matrix(
            y_test,
            test_probs,
            title="Keras CNN - Fashion MNIST (Test Set)",
            save_path="results/visualizations/Fashion_MNIST_Keras_CNN_confusion_matrix.png"
        )


def generate_advanced_regression_visualizations(visualizer: ResultsVisualizer):
    """Generuje wizualizacje dla zaawansowanych modeli regresji (CNN 1D + LSTM)."""
    print("\n" + "="*80)
    print("ADVANCED REGRESSION VISUALIZATIONS (CNN 1D & LSTM)")
    print("="*80)

    # Wczytaj dane Stock Market
    data_dir = "data/regression"
    X_train, y_train = load_data_csv(os.path.join(data_dir, "train70.csv"))
    X_val, y_val = load_data_csv(os.path.join(data_dir, "validation15.csv"))
    X_test, y_test = load_data_csv(os.path.join(data_dir, "test15.csv"))

    # === 1. CNN 1D ===
    results_cnn = "results/keras_cnn1d_wyniki_regression.xlsx"
    if os.path.exists(results_cnn):
        print("\n📈 CNN 1D - Stock Market")

        df = pd.read_excel(results_cnn, sheet_name="Results")
        best_idx = df["test_mse"].idxmin()
        best = df.loc[best_idx]

        # Trenuj najlepszy model
        model = KerasCNN1DRegression(
            n_features=X_train.shape[1],
            n_conv_layers=int(best["n_conv_layers"]),
            n_filters=int(best["n_filters"]),
            kernel_size=3,
            pool_size=2,
            n_dense_neurons=64,
            learning_rate=float(best["learning_rate"]),
            optimizer_name=str(best["optimizer"])
        )

        history = model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32, verbose=0)

        # Learning curves
        visualizer.plot_learning_curves(
            history,
            "CNN 1D - Stock Market",
            "results/visualizations/Stock_Market_CNN1D_learning_curves.png"
        )

        # Scatter plot
        test_preds = model.predict(X_test)
        visualizer.plot_regression_scatter(
            y_test,
            test_preds,
            title="CNN 1D - Stock Market (Test Set)",
            save_path="results/visualizations/Stock_Market_CNN1D_scatter.png"
        )

    # === 2. LSTM ===
    results_lstm = "results/keras_lstm_wyniki_regression.xlsx"
    if os.path.exists(results_lstm):
        print("\n📈 LSTM - Stock Market")

        df = pd.read_excel(results_lstm, sheet_name="Results")
        best_idx = df["test_mse"].idxmin()
        best = df.loc[best_idx]

        # Trenuj najlepszy model
        model = KerasLSTMRegression(
            n_features=X_train.shape[1],
            n_lstm_layers=int(best["n_lstm_layers"]),
            n_lstm_units=int(best["n_lstm_units"]),
            n_dense_neurons=32,
            learning_rate=float(best["learning_rate"]),
            optimizer_name=str(best["optimizer"]),
            dropout=0.0
        )

        history = model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32, verbose=0)

        # Learning curves
        visualizer.plot_learning_curves(
            history,
            "LSTM - Stock Market",
            "results/visualizations/Stock_Market_LSTM_learning_curves.png"
        )

        # Scatter plot
        test_preds = model.predict(X_test)
        visualizer.plot_regression_scatter(
            y_test,
            test_preds,
            title="LSTM - Stock Market (Test Set)",
            save_path="results/visualizations/Stock_Market_LSTM_scatter.png"
        )


if __name__ == "__main__":
    print("\n" + "="*80)
    print("GENERATING EXTENDED VISUALIZATIONS")
    print("="*80)

    visualizer = ResultsVisualizer()

    # 1. Fashion MNIST (MLP + CNN)
    try:
        generate_fashion_mnist_visualizations(visualizer)
    except Exception as e:
        print(f"❌ Błąd Fashion MNIST: {e}")
        import traceback
        traceback.print_exc()

    # 2. Advanced Regression (CNN 1D + LSTM)
    try:
        generate_advanced_regression_visualizations(visualizer)
    except Exception as e:
        print(f"❌ Błąd Advanced Regression: {e}")
        import traceback
        traceback.print_exc()

    # 3. Porównania (wszystkie modele)
    print("\n" + "="*80)
    print("GENERATING COMPARISONS")
    print("="*80)

    # Możesz dodać wykresy porównawcze między wszystkimi modelami tutaj
    # Np. bar chart: Manual MLP vs Keras MLP vs CNN vs LSTM dla Stock Market

    print("\n" + "="*80)
    print("✅ WSZYSTKIE WIZUALIZACJE WYGENEROWANE!")
    print(f"📂 Zapisane w: {visualizer.output_dir}")
    print("="*80 + "\n")

    print("Wygenerowane wizualizacje:")
    print("  Fashion MNIST:")
    print("    - Keras MLP: learning curves + confusion matrix")
    print("    - Keras CNN: learning curves + confusion matrix")
    print("  Stock Market (Advanced):")
    print("    - CNN 1D: learning curves + scatter plot")
    print("    - LSTM: learning curves + scatter plot")
    print("\nKolejny krok: Uruchom generate_visualizations.py dla reszty modeli (Manual MLP, Keras MLP dla innych zbiorów)")
