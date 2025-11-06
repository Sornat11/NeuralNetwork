import numpy as np

from data.sample_data_generator import create_data
from src.manual_mlp.metrics import ModelMetrics
from src.manual_mlp.model import Model
from utils.results_exporter import ResultsExporter

if __name__ == "__main__":
    metrics = ModelMetrics()
    import time

    start = time.time()
    # Parametry danych i modelu
    points = 100
    classes = 3
    n_inputs = 2
    n_hidden_layers = 10
    n_neurons = 20
    n_outputs = classes
    learning_rate = 0.1
    epochs = 50

    # Generowanie przykładowych danych
    X, y = create_data(points, classes)

    # Tworzenie i trenowanie modelu
    model = Model(n_inputs, n_hidden_layers, n_neurons, n_outputs, learning_rate)
    losses = []
    accuracies = []
    for epoch in range(epochs):
        # Forward i backward w jednej epoce
        out = X
        hidden_outputs = []
        for layer, activation in zip(model.hidden_layers, model.activations):
            out = layer.forward(out)
            hidden_outputs.append(out)
            out = activation.forward(out)
        out2 = model.output_layer.forward(out)
        loss = metrics.crossentropy_loss(out2, y)
        acc = metrics.accuracy(out2, y)
        losses.append(loss)
        accuracies.append(acc)
        print(f"Epoch {epoch+1}, loss: {loss:.4f}, accuracy: {acc:.4f}")
        # Backward
        samples = len(out2)
        dvalues = out2.copy()
        dvalues[range(samples), y] -= 1
        dvalues = dvalues / samples
        d_out = model.output_layer.backward(dvalues, out)
        for i in reversed(range(len(model.hidden_layers))):
            d_act = model.activations[i].backward(d_out)
            prev_input = X if i == 0 else model.activations[i - 1].output
            d_out = model.hidden_layers[i].backward(d_act, prev_input)

    # Pomiar czasu treningu
    training_time = time.time() - start

    # Eksport wyników, parametrów i opisu eksperymentu do Excela (po treningu)
    exporter = ResultsExporter("wyniki.xlsx")
    params = {
        "points": points,
        "classes": classes,
        "n_inputs": n_inputs,
        "n_hidden_layers": n_hidden_layers,
        "n_neurons": n_neurons,
        "n_outputs": n_outputs,
        "learning_rate": learning_rate,
        "epochs": epochs,
    }
    description = "Test manualnej sieci na danych spiralnych."
    exporter.export(
        {
            "epoch": list(range(1, len(losses) + 1)),
            "loss": losses,
            "accuracy": accuracies,
        },
        params_dict=params,
        description=description,
        training_time=training_time,
    )
