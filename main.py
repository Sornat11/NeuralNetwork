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
    batch_size = 32
    losses = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    for epoch in range(epochs):
        epoch_losses = []
        epoch_accuracies = []
        epoch_precisions = []
        epoch_recalls = []
        epoch_f1s = []
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            X_batch = X[start:end]
            y_batch = y[start:end]
            # Forward
            out = X_batch
            hidden_outputs = []
            for layer, activation in zip(model.hidden_layers, model.activations):
                out = layer.forward(out)
                hidden_outputs.append(out)
                out = activation.forward(out)
            out2 = model.output_layer.forward(out)
            # Metrics for batch
            batch_loss = metrics.crossentropy_loss(out2, y_batch)
            batch_acc = metrics.accuracy(out2, y_batch)
            batch_prec = metrics.precision(out2, y_batch)
            batch_rec = metrics.recall(out2, y_batch)
            batch_f1 = metrics.f1_score(out2, y_batch)
            epoch_losses.append(batch_loss)
            epoch_accuracies.append(batch_acc)
            epoch_precisions.append(batch_prec)
            epoch_recalls.append(batch_rec)
            epoch_f1s.append(batch_f1)
            # Backward
            samples = len(out2)
            dvalues = out2.copy()
            dvalues[range(samples), y_batch] -= 1
            dvalues = dvalues / samples
            d_out = model.output_layer.backward(dvalues, out)
            for i in reversed(range(len(model.hidden_layers))):
                d_act = model.activations[i].backward(d_out)
                prev_input = X_batch if i == 0 else model.activations[i - 1].output
                d_out = model.hidden_layers[i].backward(d_act, prev_input)
        # Średnie metryki z batchy w epoce
        loss = float(np.mean(epoch_losses))
        acc = float(np.mean(epoch_accuracies))
        prec = float(np.mean(epoch_precisions))
        rec = float(np.mean(epoch_recalls))
        f1 = float(np.mean(epoch_f1s))
        losses.append(loss)
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
        print(
            f"Epoch {epoch+1}, loss: {loss:.4f}, acc: {acc:.4f}, prec: {prec:.4f}, rec: {rec:.4f}, f1: {f1:.4f}"
        )

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
            "precision": precisions,
            "recall": recalls,
            "f1_score": f1_scores,
        },
        params_dict=params,
        description=description,
        training_time=training_time,
    )
