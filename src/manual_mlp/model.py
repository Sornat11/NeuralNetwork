import numpy as np

from .activations import ActivationReLU, ActivationSoftmax
from .layers import LayerDense
from .metrics import ModelMetrics


class Model:
    def __init__(
        self,
        n_inputs: int,
        n_hidden_layers: int,
        n_neurons: int,
        n_outputs: int,
        learning_rate: float = 0.01,
    ):
        # Warstwy ukryte + ReLU
        self.hidden_layers: list[LayerDense] = []
        self.activations: list[ActivationReLU] = []
        input_size = n_inputs

        for _ in range(n_hidden_layers):
            self.hidden_layers.append(
                LayerDense(input_size, n_neurons, learning_rate=learning_rate)
            )
            self.activations.append(ActivationReLU())
            input_size = n_neurons

        # Warstwa wyjściowa (logity)
        self.output_layer = LayerDense(
            input_size, n_outputs, learning_rate=learning_rate
        )

        # Softmax na wyjściu – tylko dla klasyfikacji
        self.softmax = ActivationSoftmax()

        self.metrics = ModelMetrics()
        self.learning_rate = learning_rate

    def forward(self, X: np.ndarray, classification: bool = False) -> np.ndarray:
        """
        Prosty forward:
        - dla regresji: zwraca logity z warstwy wyjściowej
        - dla klasyfikacji: jeśli classification=True, zwraca prawdopodobieństwa (softmax)
        """
        out = X
        for layer, activation in zip(self.hidden_layers, self.activations):
            out = layer.forward(out)
            out = activation.forward(out)

        logits = self.output_layer.forward(out)

        if classification:
            return self.softmax.forward(logits)  # prawdopodobieństwa klas
        return logits  # regresja / surowe logity

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 32):
        """
        Trening dla KLASYFIKACJI – 1:1 z pętlą w ExperimentRunner:
        - forward: hidden layers + ReLU + output_layer + softmax
        - loss: crossentropy
        - metriki: accuracy (jak w ModelMetrics)
        - backward: gradient softmax+CE (dvalues = probs; dvalues[range,sample,y]-=1; /N)
        """

        metrics = self.metrics
        n_samples = len(X)

        for epoch in range(epochs):
            epoch_losses = []
            epoch_accuracies = []

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X[start:end]
                y_batch = y[start:end]

                # Jeżeli target jest one-hot, konwertujemy na etykiety
                if y_batch.ndim == 2 and y_batch.shape[1] > 1:
                    y_batch_labels = np.argmax(y_batch, axis=1)
                else:
                    y_batch_labels = y_batch

                # ------- FORWARD -------
                out = X_batch
                for layer, activation in zip(self.hidden_layers, self.activations):
                    out = layer.forward(out)
                    out = activation.forward(out)

                # Logity z warstwy wyjściowej
                logits = self.output_layer.forward(out)

                # Softmax -> prawdopodobieństwa
                probs = self.softmax.forward(logits)

                # ------- METRYKI -------
                loss = metrics.crossentropy_loss(probs, y_batch_labels)
                acc = metrics.accuracy(probs, y_batch_labels)

                epoch_losses.append(loss)
                epoch_accuracies.append(acc)

                # ------- BACKWARD (softmax + CE) -------
                samples = len(probs)
                dvalues = probs.copy()
                dvalues[np.arange(samples), y_batch_labels] -= 1
                dvalues = dvalues / samples

                # Warstwa wyjściowa
                prev_hidden_out = out if len(self.hidden_layers) > 0 else X_batch
                d_out = self.output_layer.backward(dvalues, prev_hidden_out)

                # Warstwy ukryte w odwrotnej kolejności
                for i in reversed(range(len(self.hidden_layers))):
                    d_act = self.activations[i].backward(d_out)
                    prev_input = X_batch if i == 0 else self.activations[i - 1].output
                    d_out = self.hidden_layers[i].backward(d_act, prev_input)

            # Średni loss/accuracy po epoce (jak w ExperimentRunner)
            mean_loss = float(np.mean(epoch_losses))
            mean_acc = float(np.mean(epoch_accuracies))

            print(f"Epoch {epoch+1}/{epochs} - loss: {mean_loss:.4f}, acc: {mean_acc:.4f}")
