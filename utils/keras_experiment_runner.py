"""
Experiment Runner dla modeli Keras.
Analogiczny do experiment_runner.py, ale dostosowany do API Keras.
"""

import itertools
import numpy as np
from tqdm import tqdm, trange
from typing import Dict, List, Any

from src.models.keras_mlp import KerasMLPModel
from utils.results_exporter import ResultsExporter


class KerasExperimentRunner:
    """
    Runner dla eksperymentów z modelami Keras.
    Uruchamia grid search po hiperparametrach, zapisuje wyniki do Excel.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: Dict[str, List[Any]],
        n_runs: int = 3,
        batch_size: int = 32,
        epochs: int = 50,
        description: str = None,
        task_type: str = "classification",  # "classification" lub "regression"
        output_file: str = "keras_experiment_results.xlsx",
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        X_test: np.ndarray = None,
        y_test: np.ndarray = None,
    ):
        """
        Args:
            X_train: Dane treningowe
            y_train: Etykiety treningowe
            param_grid: Słownik z listami hiperparametrów do przeszukania
                        np. {"n_hidden_layers": [1, 2], "n_neurons": [16, 32], ...}
            n_runs: Liczba powtórzeń dla każdej kombinacji hiperparametrów
            batch_size: Rozmiar batcha
            epochs: Liczba epok treningu
            description: Opis eksperymentu (dla Excel)
            task_type: "classification" lub "regression"
            output_file: Nazwa pliku Excel z wynikami
            X_val, y_val: Dane walidacyjne (opcjonalne)
            X_test, y_test: Dane testowe (opcjonalne)
        """
        self.X_train = X_train
        self.y_train = y_train
        self.param_grid = param_grid
        self.n_runs = n_runs
        self.batch_size = batch_size
        self.epochs = epochs
        self.description = description
        self.task_type = task_type
        self.output_file = output_file
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

        # Automatycznie określamy n_inputs i n_outputs z danych
        self.n_inputs = X_train.shape[1]
        if task_type == "classification":
            # Liczba klas = max(y) + 1 (zakładamy etykiety 0, 1, 2, ...)
            self.n_outputs = int(np.max(y_train)) + 1
        else:
            # Regresja - zawsze 1 wyjście
            self.n_outputs = 1

    def run_all(self) -> List[Dict]:
        """
        Uruchamia wszystkie eksperymenty (grid search).

        Returns:
            Lista wyników (najlepsze runy dla każdej kombinacji hiperparametrów)
        """
        param_names = list(self.param_grid.keys())
        param_combinations = list(itertools.product(*self.param_grid.values()))
        all_results = []

        total_experiments = len(param_combinations) * self.n_runs

        print(f"\n{'='*60}")
        print(f"Rozpoczynam eksperymenty Keras - {self.task_type}")
        print(f"Kombinacji hiperparametrów: {len(param_combinations)}")
        print(f"Runów na kombinację: {self.n_runs}")
        print(f"Razem eksperymentów: {total_experiments}")
        print(f"{'='*60}\n")

        for exp_idx in trange(total_experiments, desc="Postęp eksperymentów"):
            combo_idx = exp_idx // self.n_runs
            run_idx = exp_idx % self.n_runs
            combo = param_combinations[combo_idx]
            param_dict = dict(zip(param_names, combo))

            # Trenujemy model i zbieramy wyniki
            result = self._run_single_experiment(
                param_dict=param_dict,
                run_number=run_idx + 1,
                combo_number=combo_idx + 1,
                total_combos=len(param_combinations)
            )

            all_results.append(result)

        # Wybieramy najlepszy run dla każdej kombinacji
        best_results = self._select_best_runs(all_results)

        # Eksportujemy do Excel
        self._export_results(best_results)

        print(f"\n{'='*60}")
        print(f"✅ Eksperymenty zakończone!")
        print(f"📊 Wyniki zapisane do: {self.output_file}")
        print(f"{'='*60}\n")

        return best_results

    def _run_single_experiment(
        self,
        param_dict: Dict[str, Any],
        run_number: int,
        combo_number: int,
        total_combos: int
    ) -> Dict:
        """
        Uruchamia pojedynczy eksperyment (jedna kombinacja + jeden run).

        Returns:
            Słownik z wynikami eksperymentu
        """
        # Tworzymy model Keras
        model = KerasMLPModel(
            n_inputs=self.n_inputs,
            n_hidden_layers=param_dict.get("n_hidden_layers", 2),
            n_neurons=param_dict.get("n_neurons", 16),
            n_outputs=self.n_outputs,
            learning_rate=param_dict.get("learning_rate", 0.01),
            task_type=self.task_type
        )

        # Trenujemy model (verbose=0 żeby nie spamowało)
        history = model.train(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0  # Cicho, bo mamy tqdm
        )

        # Zbieramy metryki z ostatniej epoki
        result = {
            **param_dict,
            "run": run_number,
        }

        if self.task_type == "classification":
            # KLASYFIKACJA
            # Metryki treningowe (ostatnia epoka)
            result["loss"] = float(history["loss"][-1])
            result["accuracy"] = float(history["accuracy"][-1])

            # Metryki walidacyjne (jeśli dostępne)
            if "val_loss" in history:
                result["val_loss"] = float(history["val_loss"][-1])
                result["val_accuracy"] = float(history["val_accuracy"][-1])

            # Metryki testowe
            if self.X_test is not None and self.y_test is not None:
                test_loss, test_acc = model.evaluate(self.X_test, self.y_test)
                result["test_loss"] = float(test_loss)
                result["test_accuracy"] = float(test_acc)

                # Dodatkowe metryki (precision, recall, f1) - obliczamy ręcznie
                test_probs = model.predict(self.X_test)
                test_preds = np.argmax(test_probs, axis=1)

                # Używamy funkcji z manual_mlp/metrics.py
                from src.manual_mlp.metrics import ModelMetrics
                metrics_calc = ModelMetrics()

                result["test_precision"] = float(metrics_calc.precision(test_probs, self.y_test))
                result["test_recall"] = float(metrics_calc.recall(test_probs, self.y_test))
                result["test_f1_score"] = float(metrics_calc.f1_score(test_probs, self.y_test))

        else:
            # REGRESJA
            # Metryki treningowe (ostatnia epoka)
            result["mse"] = float(history["loss"][-1])  # loss = MSE dla regresji
            result["mae"] = float(history["mae"][-1])

            # R² score na train
            train_preds = model.predict(self.X_train)
            from src.manual_mlp.metrics import ModelMetrics
            metrics_calc = ModelMetrics()
            result["r2"] = float(metrics_calc.r2_score(train_preds, self.y_train.reshape(-1, 1)))

            # Metryki walidacyjne (jeśli dostępne)
            if "val_loss" in history:
                result["val_mse"] = float(history["val_loss"][-1])
                result["val_mae"] = float(history["val_mae"][-1])

                # R² score na val
                val_preds = model.predict(self.X_val)
                result["val_r2"] = float(metrics_calc.r2_score(val_preds, self.y_val.reshape(-1, 1)))

            # Metryki testowe
            if self.X_test is not None and self.y_test is not None:
                test_loss, test_mae = model.evaluate(self.X_test, self.y_test)
                result["test_mse"] = float(test_loss)
                result["test_mae"] = float(test_mae)

                # R² score na test
                test_preds = model.predict(self.X_test)
                result["test_r2"] = float(metrics_calc.r2_score(test_preds, self.y_test.reshape(-1, 1)))

        return result

    def _select_best_runs(self, all_results: List[Dict]) -> List[Dict]:
        """
        Wybiera najlepszy run dla każdej kombinacji hiperparametrów.

        Args:
            all_results: Wszystkie wyniki eksperymentów

        Returns:
            Lista najlepszych wyników (po jednym na kombinację)
        """
        best_results = {}

        # Wybieramy metrykę do porównania
        if self.task_type == "classification":
            best_metric = "loss"
        else:
            best_metric = "mse"

        for result in all_results:
            # Klucz kombinacji (bez "run")
            combo_key = tuple((k, result[k]) for k in self.param_grid.keys())

            # Wybieramy run z najniższym loss/mse
            if (
                combo_key not in best_results
                or result[best_metric] < best_results[combo_key][best_metric]
            ):
                best_results[combo_key] = result

        return list(best_results.values())

    def _export_results(self, results: List[Dict]) -> None:
        """
        Eksportuje wyniki do pliku Excel.

        Args:
            results: Lista wyników do eksportu
        """
        exporter = ResultsExporter(self.output_file)

        # Przygotowujemy dane dla exportera
        results_dict = {
            key: [r[key] for r in results]
            for key in results[0].keys()
        }

        params_dict = {
            k: [r[k] for r in results]
            for k in self.param_grid.keys()
        }

        exporter.export(
            results_dict,
            params_dict=params_dict,
            description=self.description
        )
