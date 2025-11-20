"""
Narzędzia do wizualizacji wyników eksperymentów.
Generuje wykresy: learning curves, confusion matrices, porównania manual vs Keras.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Tuple, Optional

# Ustawienia dla lepszej jakości wykresów
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['figure.figsize'] = (10, 6)


class ResultsVisualizer:
    """Klasa do wizualizacji wyników eksperymentów."""

    def __init__(self, output_dir: str = "results/visualizations"):
        """
        Args:
            output_dir: Folder do zapisywania wykresów
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_learning_curves(
        self,
        history: Dict[str, List[float]],
        title: str,
        save_path: Optional[str] = None
    ) -> None:
        """
        Rysuje learning curves (loss i accuracy/mae przez epoki).

        Args:
            history: Słownik z historią treningu z Keras
                     (keys: 'loss', 'accuracy'/'mae', 'val_loss', 'val_accuracy'/'val_mae')
            title: Tytuł wykresu
            save_path: Ścieżka do zapisania (jeśli None, używa self.output_dir)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        axes[0].plot(history['loss'], label='Train', linewidth=2)
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Validation', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{title} - Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy (klasyfikacja) lub MAE (regresja)
        if 'accuracy' in history:
            metric_name = 'Accuracy'
            train_key = 'accuracy'
            val_key = 'val_accuracy'
        else:
            metric_name = 'MAE'
            train_key = 'mae'
            val_key = 'val_mae'

        axes[1].plot(history[train_key], label='Train', linewidth=2)
        if val_key in history:
            axes[1].plot(history[val_key], label='Validation', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(metric_name)
        axes[1].set_title(f'{title} - {metric_name}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            # Zamień znaki niedozwolone w nazwach plików
            safe_title = title.replace(' ', '_').replace('/', '_').replace('\\', '_')
            save_path = os.path.join(self.output_dir, f"{safe_title}_learning_curves.png")

        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✅ Zapisano: {save_path}")

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None
    ) -> None:
        """
        Rysuje macierz pomyłek (confusion matrix).

        Args:
            y_true: Prawdziwe etykiety
            y_pred: Predykcje (prawdopodobieństwa lub klasy)
            class_names: Nazwy klas (opcjonalne)
            title: Tytuł wykresu
            save_path: Ścieżka do zapisania
        """
        # Jeśli y_pred to prawdopodobieństwa, konwertuj na klasy
        if y_pred.ndim == 2:
            y_pred = np.argmax(y_pred, axis=1)

        # Oblicz confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)

        # Normalizuj (wartości w %)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # Rysuj
        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax, label='Percentage (%)')

        # Etykiety osi
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(cm))]

        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel='True label',
            xlabel='Predicted label',
            title=title
        )

        # Rotacja etykiet
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Dodaj wartości do komórek
        thresh = cm_normalized.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.1f}%)',
                        ha="center", va="center",
                        color="white" if cm_normalized[i, j] > thresh else "black",
                        fontsize=9)

        plt.tight_layout()

        if save_path is None:
            safe_title = title.replace(' ', '_').replace('/', '_').replace('\\', '_')
            save_path = os.path.join(self.output_dir, f"{safe_title}_confusion_matrix.png")

        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✅ Zapisano: {save_path}")

    def plot_regression_scatter(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Regression: Predicted vs Actual",
        save_path: Optional[str] = None
    ) -> None:
        """
        Rysuje scatter plot: predykcje vs rzeczywiste wartości (dla regresji).

        Args:
            y_true: Prawdziwe wartości
            y_pred: Predykcje
            title: Tytuł wykresu
            save_path: Ścieżka do zapisania
        """
        # Flatten jeśli wielowymiarowe
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        # Oblicz R²
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 8))

        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)

        # Idealna linia (y = x)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')

        ax.set_xlabel('True values')
        ax.set_ylabel('Predicted values')
        ax.set_title(f'{title}\nR² = {r2:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            safe_title = title.replace(' ', '_').replace('/', '_').replace('\\', '_')
            save_path = os.path.join(self.output_dir, f"{safe_title}_scatter.png")

        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✅ Zapisano: {save_path}")

    def plot_comparison_bar(
        self,
        manual_results: pd.DataFrame,
        keras_results: pd.DataFrame,
        metric: str,
        dataset_name: str,
        save_path: Optional[str] = None
    ) -> None:
        """
        Rysuje wykres słupkowy porównujący wyniki manual vs Keras.

        Args:
            manual_results: DataFrame z wynikami ręcznej implementacji
            keras_results: DataFrame z wynikami Keras
            metric: Nazwa metryki do porównania (np. 'test_accuracy', 'test_mse')
            dataset_name: Nazwa zbioru danych
            save_path: Ścieżka do zapisania
        """
        # Znajdź najlepszy wynik dla manual i keras
        if 'loss' in metric or 'mse' in metric:
            # Im mniejsze, tym lepsze
            manual_best = manual_results[metric].min()
            keras_best = keras_results[metric].min()
        else:
            # Im większe, tym lepsze (accuracy, f1, r2, etc.)
            manual_best = manual_results[metric].max()
            keras_best = keras_results[metric].max()

        fig, ax = plt.subplots(figsize=(8, 6))

        implementations = ['Manual MLP', 'Keras MLP']
        values = [manual_best, keras_best]

        bars = ax.bar(implementations, values, color=['#3498db', '#e74c3c'], alpha=0.8, edgecolor='black')

        # Dodaj wartości na słupkach
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{dataset_name}: Comparison of {metric.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(
                self.output_dir,
                f"{dataset_name}_{metric}_comparison.png"
            )

        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✅ Zapisano: {save_path}")

    def create_all_comparisons(
        self,
        results_dir: str = "results"
    ) -> None:
        """
        Tworzy wszystkie wykresy porównawcze manual vs Keras.

        Args:
            results_dir: Folder z wynikami Excel
        """
        # Konfiguracja zbiorów i metryk
        datasets = [
            ("classification", "test_accuracy", False),
            ("classification_our", "test_accuracy", False),
            ("regression", "test_mse", True),
            ("regression_our", "test_mse", True),
        ]

        splits = [
            "train_test",
            "train_val_test"
        ]

        print(f"\n{'='*60}")
        print("Tworzenie wykresów porównawczych Manual vs Keras")
        print(f"{'='*60}\n")

        for dataset_name, metric, is_regression in datasets:
            for split in splits:
                manual_file = os.path.join(
                    results_dir,
                    f"manual_perceptron_wyniki_{dataset_name}_{split}.xlsx"
                )
                keras_file = os.path.join(
                    results_dir,
                    f"keras_wyniki_{dataset_name}_{split}.xlsx"
                )

                # Sprawdź czy oba pliki istnieją
                if not os.path.exists(manual_file) or not os.path.exists(keras_file):
                    print(f"⚠️  Pomijam {dataset_name}_{split} - brak plików")
                    continue

                try:
                    # Wczytaj wyniki
                    manual_df = pd.read_excel(manual_file, sheet_name="Results")
                    keras_df = pd.read_excel(keras_file, sheet_name="Results")

                    # Utwórz wykres porównawczy
                    title = f"{dataset_name.replace('_', ' ').title()} ({split.replace('_', '/')})"
                    self.plot_comparison_bar(
                        manual_df,
                        keras_df,
                        metric,
                        title
                    )

                    # Dodatkowe metryki jeśli dostępne
                    if not is_regression:
                        # Klasyfikacja: dodaj f1_score
                        if "test_f1_score" in manual_df.columns and "test_f1_score" in keras_df.columns:
                            self.plot_comparison_bar(
                                manual_df,
                                keras_df,
                                "test_f1_score",
                                title
                            )
                    else:
                        # Regresja: dodaj R²
                        if "test_r2" in manual_df.columns and "test_r2" in keras_df.columns:
                            self.plot_comparison_bar(
                                manual_df,
                                keras_df,
                                "test_r2",
                                title
                            )

                except Exception as e:
                    print(f"❌ Błąd dla {dataset_name}_{split}: {e}")

        print(f"\n{'='*60}")
        print("✅ Zakończono tworzenie wykresów porównawczych")
        print(f"{'='*60}\n")