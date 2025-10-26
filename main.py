
"""
Main entry point for the neural network project.
Uses console_ui for all user interaction and model selection.
"""

from console_ui import show_menu, show_welcome, show_result
from src.perceptron_manual.mlp import MLP
from src.keras_rnn import KerasRNN
from src.keras_cnn import KerasCNN
from data.xor import generate_xor
from src.formatting import format_results
import numpy as np

def run_mlp():
    X, y = generate_xor(reps=100, noise=0.1)
    model = MLP(sizes=[2, 8, 1], hidden_act="relu")
    model.train(X, y, epochs=5000, batch_size=32, lr=0.01)
    y_pred = model.predict(X)
    format_results(X, y, y_pred, num_samples=10)

def run_rnn():
    # Example: batch=20, timesteps=10, features=3
    X = np.random.rand(20, 10, 3)
    y = np.random.randint(0, 2, size=(20, 1))
    model = KerasRNN(input_shape=(10, 3), hidden_dim=8, output_dim=1, cell_type='LSTM', seed=123)
    model.train(X, y, epochs=2, batch_size=4)
    y_pred = model.predict(X)
    show_result(f"Keras RNN predictions: {y_pred.flatten()}")

def run_cnn():
    # Example: batch=20, 8x8 grayscale images
    X = np.random.rand(20, 8, 8, 1)
    y = np.random.randint(0, 2, size=(20, 1))
    model = KerasCNN(input_shape=(8, 8, 1), num_filters=4, filter_size=3, pool_size=2, output_dim=1, seed=123)
    model.train(X, y, epochs=2, batch_size=4)
    preds = model.predict(X)
    show_result(f"Keras CNN predictions: {preds.flatten()}")

if __name__ == "__main__":
    show_welcome()
    choice = show_menu()
    if choice == "1":
        run_mlp()
    elif choice == "2":
        run_rnn()
    elif choice == "3":
        run_cnn()