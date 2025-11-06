"""
Framework do uruchamiania eksperymentów z różnymi parametrami.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional, Tuple
import numpy as np
import pandas as pd
from itertools import product

# Import własnych modułów (będą importowane dynamicznie)


class ExperimentRunner:
    """
    Klasa do uruchamiania i zarządzania eksperymentami.
    """

    def __init__(self, results_dir: str = "results"):
        """
        Args:
            results_dir: Katalog do zapisywania wyników
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict] = []

    def run_experiment(
        self,
        experiment_name: str,
        model_fn: Callable,
        train_fn: Callable,
        eval_fn: Callable,
        data: Dict[str, np.ndarray],
        param_grid: Dict[str, List[Any]],
        n_repeats: int = 5,
        save_results: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Uruchamia eksperymenty z różnymi parametrami i wielokrotnym powtórzeniem.

        Args:
            experiment_name: Nazwa eksperymentu
            model_fn: Funkcja tworząca model (przyjmuje **params)
            train_fn: Funkcja trenująca model (model, data, **params)
            eval_fn: Funkcja ewaluująca model (model, data) -> metrics dict
            data: Dict z danymi (X_train, y_train, X_val, y_val, X_test, y_test)
            param_grid: Dict z parametrami do przetestowania
            n_repeats: Liczba powtórzeń każdego zestawu parametrów
            save_results: Czy zapisać wyniki
            verbose: Czy wyświetlać postęp

        Returns:
            DataFrame z wynikami
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Eksperyment: {experiment_name}")
            print(f"{'='*60}\n")

        # Generuj wszystkie kombinacje parametrów
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))

        total_runs = len(param_combinations) * n_repeats

        if verbose:
            print(f"Liczba kombinacji parametrów: {len(param_combinations)}")
            print(f"Liczba powtórzeń: {n_repeats}")
            print(f"Całkowita liczba uruchomień: {total_runs}\n")

        results = []
        run_counter = 0

        # Iteruj po wszystkich kombinacjach parametrów
        for param_combo in param_combinations:
            params = dict(zip(param_names, param_combo))

            if verbose:
                print(f"\nTestowanie parametrów: {params}")

            # Powtórz n_repeats razy
            repeat_results = {
                "train": [],
                "val": [],
                "test": [],
            }

            for repeat in range(n_repeats):
                run_counter += 1
                start_time = time.time()

                if verbose:
                    print(f"  Powtórzenie {repeat + 1}/{n_repeats}...", end=" ")

                try:
                    # Stwórz model
                    model = model_fn(**params)

                    # Trenuj model
                    history = train_fn(model, data, **params)

                    # Ewaluuj na wszystkich zbiorach
                    train_metrics = eval_fn(model, {
                        "X": data["X_train"],
                        "y": data["y_train"]
                    })
                    test_metrics = eval_fn(model, {
                        "X": data["X_test"],
                        "y": data["y_test"]
                    })

                    # Walidacja (jeśli istnieje)
                    if data.get("X_val") is not None:
                        val_metrics = eval_fn(model, {
                            "X": data["X_val"],
                            "y": data["y_val"]
                        })
                    else:
                        val_metrics = {}

                    # Zapisz wyniki
                    repeat_results["train"].append(train_metrics)
                    repeat_results["test"].append(test_metrics)
                    if val_metrics:
                        repeat_results["val"].append(val_metrics)

                    elapsed_time = time.time() - start_time

                    if verbose:
                        # Wyświetl kluczową metrykę
                        key_metric = list(train_metrics.keys())[0]
                        print(
                            f"Done! Test {key_metric}: {test_metrics[key_metric]:.4f} "
                            f"({elapsed_time:.2f}s)"
                        )

                except Exception as e:
                    if verbose:
                        print(f"BŁĄD: {e}")
                    continue

            # Oblicz statystyki (średnie, std, min, max)
            result_entry = {
                "experiment": experiment_name,
                "timestamp": datetime.now().isoformat(),
                **params,
            }

            # Dodaj statystyki dla każdego zbioru danych
            for dataset_name in ["train", "val", "test"]:
                if repeat_results[dataset_name]:
                    metrics_df = pd.DataFrame(repeat_results[dataset_name])

                    for metric_name in metrics_df.columns:
                        values = metrics_df[metric_name].values
                        result_entry[f"{dataset_name}_{metric_name}_mean"] = np.mean(values)
                        result_entry[f"{dataset_name}_{metric_name}_std"] = np.std(values)
                        result_entry[f"{dataset_name}_{metric_name}_min"] = np.min(values)
                        result_entry[f"{dataset_name}_{metric_name}_max"] = np.max(values)
                        result_entry[f"{dataset_name}_{metric_name}_best"] = np.max(values) if "accuracy" in metric_name or "r2" in metric_name else np.min(values)

            results.append(result_entry)

        # Konwertuj do DataFrame
        results_df = pd.DataFrame(results)

        # Zapisz wyniki
        if save_results:
            self._save_results(experiment_name, results_df)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Eksperyment zakończony!")
            print(f"{'='*60}\n")

        return results_df

    def _save_results(self, experiment_name: str, results_df: pd.DataFrame) -> None:
        """
        Zapisuje wyniki eksperymentu.

        Args:
            experiment_name: Nazwa eksperymentu
            results_df: DataFrame z wynikami
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = self.results_dir / f"{experiment_name}_{timestamp}"

        # Zapisz CSV
        csv_path = base_path.with_suffix(".csv")
        results_df.to_csv(csv_path, index=False)
        print(f"Wyniki zapisane do: {csv_path}")

        # Zapisz JSON (bardziej szczegółowy)
        json_path = base_path.with_suffix(".json")
        results_df.to_json(json_path, orient="records", indent=2)
        print(f"Wyniki zapisane do: {json_path}")

    def compare_experiments(self, experiment_names: List[str]) -> pd.DataFrame:
        """
        Porównuje wyniki z różnych eksperymentów.

        Args:
            experiment_names: Lista nazw eksperymentów do porównania

        Returns:
            DataFrame z porównaniem
        """
        # TODO: Implementacja porównywania eksperymentów
        pass

    def get_best_params(
        self, results_df: pd.DataFrame, metric: str = "test_accuracy_mean", maximize: bool = True
    ) -> Dict:
        """
        Zwraca najlepsze parametry na podstawie metryki.

        Args:
            results_df: DataFrame z wynikami
            metric: Nazwa metryki do optymalizacji
            maximize: Czy maksymalizować metrykę (True) czy minimalizować (False)

        Returns:
            Dict z najlepszymi parametrami
        """
        if metric not in results_df.columns:
            raise ValueError(f"Metryka {metric} nie istnieje w wynikach")

        # Znajdź najlepszy wiersz
        if maximize:
            best_idx = results_df[metric].idxmax()
        else:
            best_idx = results_df[metric].idxmin()

        best_row = results_df.loc[best_idx]

        # Wyodrębnij tylko parametry (nie metryki)
        param_cols = [col for col in results_df.columns if not any(
            x in col for x in ["_mean", "_std", "_min", "_max", "_best", "experiment", "timestamp"]
        )]

        best_params = {col: best_row[col] for col in param_cols}

        return best_params


def create_param_grid(base_params: Dict, variations: Dict) -> Dict[str, List]:
    """
    Tworzy param_grid na podstawie bazowych parametrów i wariacji.

    Args:
        base_params: Bazowe parametry (nie będą wariantowane)
        variations: Parametry do testowania z listą wartości

    Returns:
        Param grid gotowy dla ExperimentRunner

    Example:
        >>> base = {"activation": "relu"}
        >>> variations = {"n_layers": [1, 2, 3], "learning_rate": [0.001, 0.01]}
        >>> param_grid = create_param_grid(base, variations)
    """
    param_grid = {**base_params}

    for param_name, param_values in variations.items():
        if not isinstance(param_values, list):
            param_values = [param_values]
        param_grid[param_name] = param_values

    # Upewnij się, że bazowe parametry są listami
    for param_name, param_value in base_params.items():
        if not isinstance(param_value, list):
            param_grid[param_name] = [param_value]

    return param_grid
