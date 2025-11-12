import itertools

import numpy as np
from tqdm import tqdm, trange

from src.manual_mlp.metrics import ModelMetrics
from utils.results_exporter import ResultsExporter


class ExperimentRunner:
    def __init__(
        self,
        model_class,
        X,
        y,
        param_grid,
        n_runs=5,
        k_folds=1,
        batch_size=32,
        epochs=50,
        description=None,
        regression=False,
        output_file="experiment_results.xlsx",
    ):
        """
        model_class: klasa modelu (np. Model)
        X, y: gotowe dane wejściowe i etykiety
        param_grid: dict z listami wartości hiperparametrów
        n_runs: ile razy powtarzać każdy eksperyment
        k_folds: liczba foldów (crossval, domyślnie 1 = bez crossval)
        batch_size, epochs: parametry treningu
        description: opis eksperymentu
        regression: True dla regresji, False dla klasyfikacji
        """
        self.model_class = model_class
        self.X = X
        self.y = y
        self.param_grid = param_grid
        self.n_runs = n_runs
        self.k_folds = k_folds
        self.batch_size = batch_size
        self.epochs = epochs
        self.description = description
        self.regression = regression
        self.output_file = output_file

    def run_all(self):
        param_names = list(self.param_grid.keys())
        param_combinations = list(itertools.product(*self.param_grid.values()))
        all_results = []

        total_experiments = len(param_combinations) * self.n_runs

        for exp_idx in trange(total_experiments, desc="Postęp eksperymentów"):
            combo_idx = exp_idx // self.n_runs
            run = exp_idx % self.n_runs
            combo = param_combinations[combo_idx]
            param_dict = dict(zip(param_names, combo))

            # Dane są przekazane w konstruktorze
            X, y = self.X, self.y

            # Inicjalizacja modelu (z param_dict z sensownymi domyślnymi wartościami)
            model = self.model_class(
                param_dict.get("n_inputs", 2),
                param_dict.get("n_hidden_layers", 2),
                param_dict.get("n_neurons", 8),
                param_dict.get("n_outputs", param_dict.get("classes", 3)),
                param_dict.get("learning_rate", 0.01),
            )

            metrics = ModelMetrics()

            if self.regression:
                # ------- REGRESJA -------
                losses, maes, r2s = [], [], []

                for epoch in tqdm(
                    range(self.epochs),
                    desc=f"Epoki (run {run+1}/{self.n_runs}, combo {combo_idx+1}/{len(param_combinations)})",
                    leave=False,
                ):
                    epoch_losses, epoch_maes, epoch_r2s = [], [], []

                    for start in range(0, len(X), self.batch_size):
                        end = start + self.batch_size
                        X_batch = X[start:end]
                        y_batch = y[start:end]

                        # KLUCZOWE: target na kształt (B, 1) żeby uniknąć broadcastingu
                        if y_batch.ndim == 1:
                            y_batch = y_batch.reshape(-1, 1)

                        # Forward
                        out = X_batch
                        for layer, activation in zip(
                            model.hidden_layers, model.activations
                        ):
                            out = layer.forward(out)
                            out = activation.forward(out)
                        out2 = model.output_layer.forward(out)  # (B, 1)

                        # Metryki
                        batch_loss = metrics.mse(out2, y_batch)
                        batch_mae = metrics.mae(out2, y_batch)
                        batch_r2 = metrics.r2_score(out2, y_batch)
                        epoch_losses.append(batch_loss)
                        epoch_maes.append(batch_mae)
                        epoch_r2s.append(batch_r2)

                        # Backward
                        dvalues = out2 - y_batch  # (B, 1)
                        # Spójnie przekazujemy wejście do warstwy wyjściowej
                        prev_hidden_out = (
                            out if len(model.hidden_layers) > 0 else X_batch
                        )
                        d_out = model.output_layer.backward(dvalues, prev_hidden_out)

                        for i in reversed(range(len(model.hidden_layers))):
                            d_act = model.activations[i].backward(d_out)
                            prev_input = (
                                X_batch if i == 0 else model.activations[i - 1].output
                            )
                            d_out = model.hidden_layers[i].backward(d_act, prev_input)

                    # Średnie metryki z batchy w epoce
                    losses.append(float(np.mean(epoch_losses)))
                    maes.append(float(np.mean(epoch_maes)))
                    r2s.append(float(np.mean(epoch_r2s)))

                # Zapisz wyniki dla tej kombinacji i runa
                result = {
                    **param_dict,
                    "run": run + 1,
                    "mse": losses[-1],
                    "mae": maes[-1],
                    "r2": r2s[-1],
                }
                all_results.append(result)

            else:
                # ------- KLASYFIKACJA -------
                losses, accuracies, precisions, recalls, f1_scores = [], [], [], [], []

                for epoch in tqdm(
                    range(self.epochs),
                    desc=f"Epoki (run {run+1}/{self.n_runs}, combo {combo_idx+1}/{len(param_combinations)})",
                    leave=False,
                ):
                    epoch_losses, epoch_accuracies = [], []
                    epoch_precisions, epoch_recalls, epoch_f1s = [], [], []

                    for start in range(0, len(X), self.batch_size):
                        end = start + self.batch_size
                        X_batch = X[start:end]
                        y_batch = y[start:end]

                        # Jeżeli target jest one-hot, konwertujemy na etykiety
                        if y_batch.ndim == 2 and y_batch.shape[1] > 1:
                            y_batch = np.argmax(y_batch, axis=1)

                        # Forward
                        out = X_batch
                        for layer, activation in zip(
                            model.hidden_layers, model.activations
                        ):
                            out = layer.forward(out)
                            out = activation.forward(out)
                        out2 = model.output_layer.forward(
                            out
                        )  # (B, C) – logits lub softmax pre-activation

                        # Metryki (zakładamy, że metrics.* potrafią policzyć na podstawie prawdopodobieństw/logitów i etykiet)
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

                        # Backward dla softmax+CE (lub CE na logitach) – wariant dvalues jak w Twoim kodzie
                        samples = len(out2)
                        dvalues = out2.copy()
                        dvalues[np.arange(samples), y_batch] -= 1
                        dvalues = dvalues / samples

                        prev_hidden_out = (
                            out if len(model.hidden_layers) > 0 else X_batch
                        )
                        d_out = model.output_layer.backward(dvalues, prev_hidden_out)

                        for i in reversed(range(len(model.hidden_layers))):
                            d_act = model.activations[i].backward(d_out)
                            prev_input = (
                                X_batch if i == 0 else model.activations[i - 1].output
                            )
                            d_out = model.hidden_layers[i].backward(d_act, prev_input)

                    # Średnie metryki z batchy w epoce
                    losses.append(float(np.mean(epoch_losses)))
                    accuracies.append(float(np.mean(epoch_accuracies)))
                    precisions.append(float(np.mean(epoch_precisions)))
                    recalls.append(float(np.mean(epoch_recalls)))
                    f1_scores.append(float(np.mean(epoch_f1s)))

                # Zapisz wyniki dla tej kombinacji i runa
                result = {
                    **param_dict,
                    "run": run + 1,
                    "loss": losses[-1],
                    "accuracy": accuracies[-1],
                    "precision": precisions[-1],
                    "recall": recalls[-1],
                    "f1_score": f1_scores[-1],
                }
                all_results.append(result)

        # Wybierz najlepszy run dla każdej kombinacji hiperparametrów
        best_results = {}
        # Wybierz odpowiednie pole do selekcji najlepszego runa
        best_metric = "loss" if not self.regression else "mse"
        for result in all_results:
            combo_key = tuple((k, result[k]) for k in self.param_grid.keys())
            if (
                combo_key not in best_results
                or result[best_metric] < best_results[combo_key][best_metric]
            ):
                best_results[combo_key] = result

        best_results_list = list(best_results.values())

        # Eksport do Excela
        exporter = ResultsExporter(self.output_file)
        exporter.export(
            {
                key: [r[key] for r in best_results_list]
                for key in best_results_list[0].keys()
            },
            params_dict={
                k: [r[k] for r in best_results_list] for k in self.param_grid.keys()
            },
            description=self.description,
        )
        print(
            "Eksperymenty zakończone i wyeksportowane (tylko najlepsze runy dla każdej kombinacji)."
        )
