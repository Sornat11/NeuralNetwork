"""
Wizualizacje wyników eksperymentów.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


def plot_learning_curves(
    history: Dict,
    metrics: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Learning Curves",
    figsize: Tuple[int, int] = (12, 4),
) -> None:
    """
    Wizualizuje learning curves (loss i metryki w czasie).

    Args:
        history: Dict z historią treningu (z model.fit())
        metrics: Lista metryk do wizualizacji (None = wszystkie)
        save_path: Ścieżka do zapisania wykresu
        title: Tytuł wykresu
        figsize: Rozmiar figury
    """
    if metrics is None:
        # Automatycznie wykryj metryki (bez val_)
        metrics = [k for k in history.keys() if not k.startswith("val_")]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Wartości treningowe
        epochs = range(1, len(history[metric]) + 1)
        ax.plot(epochs, history[metric], "b-", label=f"Train {metric}")

        # Wartości walidacyjne (jeśli istnieją)
        val_metric = f"val_{metric}"
        if val_metric in history:
            ax.plot(epochs, history[val_metric], "r-", label=f"Val {metric}")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"{metric.capitalize()} over Epochs")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Learning curves zapisane do: {save_path}")

    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
    normalize: bool = False,
) -> None:
    """
    Wizualizuje macierz pomyłek (confusion matrix).

    Args:
        cm: Macierz pomyłek
        class_names: Nazwy klas
        save_path: Ścieżka do zapisania wykresu
        title: Tytuł wykresu
        figsize: Rozmiar figury
        normalize: Czy normalizować wartości (do [0,1])
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Frequency" if not normalize else "Proportion"},
    )

    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix zapisana do: {save_path}")

    plt.show()


def plot_parameter_comparison(
    results_df: pd.DataFrame,
    param_name: str,
    metrics: List[str] = ["test_accuracy_mean"],
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """
    Porównuje wyniki dla różnych wartości parametru.

    Args:
        results_df: DataFrame z wynikami eksperymentów
        param_name: Nazwa parametru do porównania
        metrics: Lista metryk do wizualizacji
        save_path: Ścieżka do zapisania wykresu
        title: Tytuł wykresu
        figsize: Rozmiar figury
    """
    if param_name not in results_df.columns:
        raise ValueError(f"Parametr {param_name} nie istnieje w wynikach")

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        ax = axes[i]

        if metric not in results_df.columns:
            print(f"Metryka {metric} nie istnieje, pomijam...")
            continue

        # Grupuj po parametrze
        grouped = results_df.groupby(param_name)[metric].agg(["mean", "std"])

        # Wykres słupkowy z error bars
        x = range(len(grouped))
        ax.bar(x, grouped["mean"], yerr=grouped["std"], capsize=5, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(grouped.index, rotation=45)
        ax.set_xlabel(param_name)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs {param_name}")
        ax.grid(True, alpha=0.3, axis="y")

    if title:
        plt.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Wykres porównania zapisany do: {save_path}")

    plt.show()


def plot_heatmap_comparison(
    results_df: pd.DataFrame,
    param1: str,
    param2: str,
    metric: str = "test_accuracy_mean",
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Tworzy heatmap porównującą dwa parametry.

    Args:
        results_df: DataFrame z wynikami
        param1: Nazwa pierwszego parametru (oś Y)
        param2: Nazwa drugiego parametru (oś X)
        metric: Metryka do wizualizacji
        save_path: Ścieżka do zapisania
        title: Tytuł wykresu
        figsize: Rozmiar figury
    """
    # Pivot table
    pivot = results_df.pivot_table(
        values=metric, index=param1, columns=param2, aggfunc="mean"
    )

    plt.figure(figsize=figsize)
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis", cbar_kws={"label": metric})

    if title is None:
        title = f"{metric} - {param1} vs {param2}"

    plt.title(title)
    plt.xlabel(param2)
    plt.ylabel(param1)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Heatmap zapisana do: {save_path}")

    plt.show()


def plot_model_comparison(
    results_dfs: Dict[str, pd.DataFrame],
    metric: str = "test_accuracy_mean",
    save_path: Optional[str] = None,
    title: str = "Model Comparison",
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """
    Porównuje wyniki różnych modeli.

    Args:
        results_dfs: Dict {model_name: results_df}
        metric: Metryka do porównania
        save_path: Ścieżka do zapisania
        title: Tytuł wykresu
        figsize: Rozmiar figury
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    model_names = list(results_dfs.keys())
    means = []
    stds = []
    bests = []

    for model_name, df in results_dfs.items():
        if metric in df.columns:
            means.append(df[metric].mean())
            stds.append(df[metric].std())
            # Najlepszy wynik
            maximize = "accuracy" in metric or "r2" in metric
            bests.append(df[metric].max() if maximize else df[metric].min())
        else:
            means.append(0)
            stds.append(0)
            bests.append(0)

    # Wykres 1: Średnie wyniki z error bars
    x = range(len(model_names))
    ax1.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color="skyblue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha="right")
    ax1.set_ylabel(metric)
    ax1.set_title(f"Średnie {metric}")
    ax1.grid(True, alpha=0.3, axis="y")

    # Wykres 2: Najlepsze wyniki
    ax2.bar(x, bests, alpha=0.7, color="coral")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha="right")
    ax2.set_ylabel(metric)
    ax2.set_title(f"Najlepsze {metric}")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Wykres porównania modeli zapisany do: {save_path}")

    plt.show()


def plot_training_time_comparison(
    results_df: pd.DataFrame,
    param_name: str,
    time_col: str = "training_time",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Porównuje czas treningu dla różnych parametrów.

    Args:
        results_df: DataFrame z wynikami
        param_name: Nazwa parametru do porównania
        time_col: Nazwa kolumny z czasem treningu
        save_path: Ścieżka do zapisania
        figsize: Rozmiar figury
    """
    if time_col not in results_df.columns:
        print(f"Kolumna {time_col} nie istnieje w wynikach")
        return

    plt.figure(figsize=figsize)

    grouped = results_df.groupby(param_name)[time_col].agg(["mean", "std"])

    x = range(len(grouped))
    plt.bar(x, grouped["mean"], yerr=grouped["std"], capsize=5, alpha=0.7)
    plt.xticks(x, grouped.index, rotation=45)
    plt.xlabel(param_name)
    plt.ylabel("Training Time (seconds)")
    plt.title(f"Training Time vs {param_name}")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Wykres czasu treningu zapisany do: {save_path}")

    plt.show()


def create_results_summary_table(
    results_df: pd.DataFrame,
    metrics: List[str],
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Tworzy tabelę podsumowującą wyniki.

    Args:
        results_df: DataFrame z wynikami
        metrics: Lista metryk do uwzględnienia
        save_path: Ścieżka do zapisania (CSV)

    Returns:
        DataFrame z podsumowaniem
    """
    summary_data = []

    for metric in metrics:
        if metric in results_df.columns:
            summary_data.append({
                "Metric": metric,
                "Mean": results_df[metric].mean(),
                "Std": results_df[metric].std(),
                "Min": results_df[metric].min(),
                "Max": results_df[metric].max(),
                "Median": results_df[metric].median(),
            })

    summary_df = pd.DataFrame(summary_data)

    if save_path:
        summary_df.to_csv(save_path, index=False)
        print(f"Tabela podsumowania zapisana do: {save_path}")

    return summary_df
