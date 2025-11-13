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
        X_val=None,
        y_val=None,
        X_test=None,
        y_test=None,
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
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

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
            if self.regression:
                n_outputs = 1
            else:
                n_outputs = param_dict.get("n_outputs", param_dict.get("classes", 3))

            model = self.model_class(
                param_dict.get("n_inputs", 2),
                param_dict.get("n_hidden_layers", 2),
                param_dict.get("n_neurons", 8),
                n_outputs,
                param_dict.get("learning_rate", 0.01),
            )

            if self.regression:
                # ------- REGRESJA -------
                metrics = ModelMetrics()
                losses, maes, r2s = [], [], []
                val_losses, val_maes, val_r2s = [], [], []
                test_losses, test_maes, test_r2s = [], [], []

                for epoch in tqdm(
                    range(self.epochs),
                    desc=(
                        f"Epoki (run {run+1}/{self.n_runs}, "
                        f"combo {combo_idx+1}/{len(param_combinations)})"
                    ),
                    leave=False,
                ):
                    epoch_losses, epoch_maes, epoch_r2s = [], [], []

                    for start in range(0, len(X), self.batch_size):
                        end = start + self.batch_size
                        X_batch = X[start:end]
                        y_batch = y[start:end]

                        # Upewniamy się, że kształt to (B, 1)
                        if y_batch.ndim == 1:
                            y_batch = y_batch.reshape(-1, 1)

                        # Forward przez model (bez softmaxa)
                        preds = model.forward(X_batch)  # (B, 1)

                        # Metryki batchowe
                        batch_loss = metrics.mse(preds, y_batch)
                        batch_mae = metrics.mae(preds, y_batch)
                        batch_r2 = metrics.r2_score(preds, y_batch)
                        epoch_losses.append(batch_loss)
                        epoch_maes.append(batch_mae)
                        epoch_r2s.append(batch_r2)

                        # Backward: dL/dy = y_pred - y_true
                        dvalues = preds - y_batch

                        # wejście do warstwy wyjściowej (ostatnia ReLU albo X_batch)
                        if len(model.hidden_layers) > 0:
                            prev_hidden_out = model.activations[-1].output
                        else:
                            prev_hidden_out = X_batch

                        d_out = model.output_layer.backward(dvalues, prev_hidden_out)

                        # Warstwy ukryte w odwrotnej kolejności
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

                    # Walidacja po każdej epoce (jeśli podano dane walidacyjne)
                    if self.X_val is not None and self.y_val is not None:
                        val_preds = model.forward(self.X_val)  # (N_val, 1)

                        y_val_true = self.y_val
                        if y_val_true.ndim == 1:
                            y_val_true = y_val_true.reshape(-1, 1)

                        val_loss = metrics.mse(val_preds, y_val_true)
                        val_mae = metrics.mae(val_preds, y_val_true)
                        val_r2 = metrics.r2_score(val_preds, y_val_true)
                        val_losses.append(val_loss)
                        val_maes.append(val_mae)
                        val_r2s.append(val_r2)

                # Test po treningu (jeśli podano dane testowe)
                if self.X_test is not None and self.y_test is not None:
                    test_preds = model.forward(self.X_test)

                    y_test_true = self.y_test
                    if y_test_true.ndim == 1:
                        y_test_true = y_test_true.reshape(-1, 1)

                    test_loss = metrics.mse(test_preds, y_test_true)
                    test_mae = metrics.mae(test_preds, y_test_true)
                    test_r2 = metrics.r2_score(test_preds, y_test_true)
                    test_losses.append(test_loss)
                    test_maes.append(test_mae)
                    test_r2s.append(test_r2)

                # Zapisz wyniki dla tej kombinacji i runa
                result = {
                    **param_dict,
                    "run": run + 1,
                    "mse": losses[-1],
                    "mae": maes[-1],
                    "r2": r2s[-1],
                }
                # Dodaj metryki walidacyjne jeśli są dostępne
                if self.X_val is not None and self.y_val is not None:
                    result["val_mse"] = val_losses[-1]
                    result["val_mae"] = val_maes[-1]
                    result["val_r2"] = val_r2s[-1]
                # Dodaj metryki testowe jeśli są dostępne
                if self.X_test is not None and self.y_test is not None:
                    result["test_mse"] = test_losses[-1]
                    result["test_mae"] = test_maes[-1]
                    result["test_r2"] = test_r2s[-1]

                all_results.append(result)

            else:
                # ------- KLASYFIKACJA -------
                metrics = ModelMetrics()
                losses, accuracies, precisions, recalls, f1_scores = [], [], [], [], []

                for epoch in tqdm(
                    range(self.epochs),
                    desc=(
                        f"Epoki (run {run+1}/{self.n_runs}, "
                        f"combo {combo_idx+1}/{len(param_combinations)})"
                    ),
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
                            y_labels = np.argmax(y_batch, axis=1)
                        else:
                            y_labels = y_batch

                        # Forward przez model z softmaxem
                        probs = model.forward(X_batch, classification=True)  # (B, C)

                        # Metryki
                        batch_loss = metrics.crossentropy_loss(probs, y_labels)
                        batch_acc = metrics.accuracy(probs, y_labels)
                        batch_prec = metrics.precision(probs, y_labels)
                        batch_rec = metrics.recall(probs, y_labels)
                        batch_f1 = metrics.f1_score(probs, y_labels)
                        epoch_losses.append(batch_loss)
                        epoch_accuracies.append(batch_acc)
                        epoch_precisions.append(batch_prec)
                        epoch_recalls.append(batch_rec)
                        epoch_f1s.append(batch_f1)

                        # Backward dla softmax+CE z y jako etykietami
                        samples = len(probs)
                        dvalues = probs.copy()
                        dvalues[np.arange(samples), y_labels] -= 1
                        dvalues = dvalues / samples

                        # wejście do warstwy wyjściowej
                        if len(model.hidden_layers) > 0:
                            prev_hidden_out = model.activations[-1].output
                        else:
                            prev_hidden_out = X_batch

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

                # Metryki walidacyjne (jeśli podano dane walidacyjne)
                val_loss = val_acc = val_prec = val_rec = val_f1 = None
                if self.X_val is not None and self.y_val is not None:
                    val_probs = model.forward(self.X_val, classification=True)
                    val_loss = metrics.crossentropy_loss(val_probs, self.y_val)
                    val_acc = metrics.accuracy(val_probs, self.y_val)
                    val_prec = metrics.precision(val_probs, self.y_val)
                    val_rec = metrics.recall(val_probs, self.y_val)
                    val_f1 = metrics.f1_score(val_probs, self.y_val)

                # Metryki testowe (jeśli podano dane testowe)
                test_loss = test_acc = test_prec = test_rec = test_f1 = None
                if self.X_test is not None and self.y_test is not None:
                    test_probs = model.forward(self.X_test, classification=True)
                    test_loss = metrics.crossentropy_loss(test_probs, self.y_test)
                    test_acc = metrics.accuracy(test_probs, self.y_test)
                    test_prec = metrics.precision(test_probs, self.y_test)
                    test_rec = metrics.recall(test_probs, self.y_test)
                    test_f1 = metrics.f1_score(test_probs, self.y_test)

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
                if val_loss is not None:
                    result["val_loss"] = val_loss
                    result["val_accuracy"] = val_acc
                    result["val_precision"] = val_prec
                    result["val_recall"] = val_rec
                    result["val_f1_score"] = val_f1
                if test_loss is not None:
                    result["test_loss"] = test_loss
                    result["test_accuracy"] = test_acc
                    result["test_precision"] = test_prec
                    result["test_recall"] = test_rec
                    result["test_f1_score"] = test_f1

                all_results.append(result)

        # Wybierz najlepszy run dla każdej kombinacji hiperparametrów
        best_results = {}
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
            "Eksperymenty zakończone i wyeksportowane "
            "(tylko najlepsze runy dla każdej kombinacji)."
        )
