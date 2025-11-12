import numpy as np

from .activations import ActivationReLU
from .layers import LayerDense
from .metrics import ModelMetrics


class Model:
    def __init__(
        self, n_inputs, n_hidden_layers, n_neurons, n_outputs, learning_rate=0.01
    ):
        # Tworzymy listę warstw ukrytych
        self.hidden_layers = []
        self.activations = []
        input_size = n_inputs
        for _ in range(n_hidden_layers):
            self.hidden_layers.append(
                LayerDense(input_size, n_neurons, learning_rate=learning_rate)
            )
            self.activations.append(ActivationReLU())
            input_size = n_neurons
        # Warstwa wyjściowa
        self.output_layer = LayerDense(
            input_size, n_outputs, learning_rate=learning_rate
        )
        self.metrics = ModelMetrics()
        self.learning_rate = learning_rate

    def forward(self, X):
        out = X
        for layer, activation in zip(self.hidden_layers, self.activations):
            out = layer.forward(out)
            out = activation.forward(out)
        out = self.output_layer.forward(out)
        return out

    def train(self, X, y, epochs=10):
        for epoch in range(epochs):
            # Forward pass przez wszystkie warstwy
            out = X
            loss = self.metrics.crossentropy_loss(out, y)
            # ...existing code...
            hidden_outputs = []
            for layer, activation in zip(self.hidden_layers, self.activations):
                out = layer.forward(out)
                hidden_outputs.append(out)
                out = activation.forward(out)
            out2 = self.output_layer.forward(out)

            # Calculate loss using metrics
            loss = self.metrics.crossentropy_loss(out2, y)
            acc = self.metrics.accuracy(out2, y)
            print(f"Epoch {epoch+1}, loss: {loss:.4f}, accuracy: {acc:.4f}")
            # Print predictions vs true labels (debug)
            preds = np.argmax(out2, axis=1)
            print(f"Predictions: {preds[:10]}")
            print(f"True labels: {y[:10]}")

            # Backward pass
            samples = len(out2)
            dvalues = out2.copy()
            dvalues[range(samples), y] -= 1
            dvalues = dvalues / samples

            # Warstwa wyjściowa
            d_out = self.output_layer.backward(dvalues, out)

            # Warstwy ukryte (w odwrotnej kolejności)
            for i in reversed(range(len(self.hidden_layers))):
                d_act = self.activations[i].backward(d_out)
                prev_input = X if i == 0 else self.activations[i - 1].output
                d_out = self.hidden_layers[i].backward(d_act, prev_input)
