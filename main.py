
from nn.mlp import MLP
from data.xor import generate_xor
from nn.formatting import format_results

if __name__ == "__main__":
    X, y = generate_xor(reps=100, noise=0.1)

    model = MLP(sizes=[2, 8, 1], hidden_act="relu")

    # Train the network
    model.train(X, y, epochs=5000, batch_size=32, lr=0.01)

    # Predict results for the same data
    y_pred = model.predict(X)

    # Format and print results
    format_results(X, y, y_pred, num_samples=10)