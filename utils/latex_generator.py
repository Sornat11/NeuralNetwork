"""
Generator fragmentów LaTeX-a z wyników eksperymentów.
Automatycznie generuje tabele i wstawia referencje do wykresów.
"""

import os
import pandas as pd
from typing import List, Tuple


class LaTeXGenerator:
    """Klasa do generowania fragmentów LaTeX-a z wyników."""

    def __init__(self, results_dir: str = "results", viz_dir: str = "results/visualizations"):
        self.results_dir = results_dir
        self.viz_dir = viz_dir

    def generate_best_config_table(
        self,
        dataset_name: str,
        split_name: str,
        implementation: str = "manual"
    ) -> str:
        """
        Generuje tabelę LaTeX z najlepszą konfiguracją hiperparametrów.

        Args:
            dataset_name: Nazwa zbioru danych
            split_name: train_test lub train_val_test
            implementation: "manual" lub "keras"

        Returns:
            Kod LaTeX tabeli
        """
        # Wczytaj wyniki
        if implementation == "manual":
            filename = f"manual_perceptron_wyniki_{dataset_name}_{split_name}.xlsx"
        else:
            filename = f"keras_wyniki_{dataset_name}_{split_name}.xlsx"

        filepath = os.path.join(self.results_dir, filename)

        if not os.path.exists(filepath):
            return f"% Brak pliku: {filename}\n"

        df = pd.read_excel(filepath, sheet_name="Results")

        # Określ metrykę do optymalizacji
        is_regression = "regression" in dataset_name
        if is_regression:
            metric = "test_mse"
            best_idx = df[metric].idxmin()
        else:
            metric = "test_accuracy"
            best_idx = df[metric].idxmax()

        best_row = df.loc[best_idx]

        # Generuj kod LaTeX
        latex = "\\begin{table}[H]\n"
        latex += "\\centering\n"
        latex += "\\begin{tabular}{ll}\n"
        latex += "\\toprule\n"
        latex += "\\textbf{Hiperparametr} & \\textbf{Wartość} \\\\\n"
        latex += "\\midrule\n"
        latex += f"Liczba warstw ukrytych & {int(best_row['n_hidden_layers'])} \\\\\n"
        latex += f"Liczba neuronów & {int(best_row['n_neurons'])} \\\\\n"
        latex += f"Learning rate & {best_row['learning_rate']:.4f} \\\\\n"
        latex += "\\midrule\n"

        # Dodaj metryki
        if is_regression:
            latex += f"Test MSE & {best_row['test_mse']:.4f} \\\\\n"
            latex += f"Test MAE & {best_row['test_mae']:.4f} \\\\\n"
            latex += f"Test R² & {best_row['test_r2']:.4f} \\\\\n"
        else:
            latex += f"Test Accuracy & {best_row['test_accuracy']:.4f} \\\\\n"
            latex += f"Test Precision & {best_row['test_precision']:.4f} \\\\\n"
            latex += f"Test Recall & {best_row['test_recall']:.4f} \\\\\n"
            latex += f"Test F1-score & {best_row['test_f1_score']:.4f} \\\\\n"

        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"

        impl_name = "Manual MLP" if implementation == "manual" else "Keras MLP"
        dataset_title = dataset_name.replace('_', ' ').title()
        split_title = split_name.replace('_', '/')

        latex += f"\\caption{{Najlepsza konfiguracja - {impl_name}, {dataset_title} ({split_title})}}\n"
        latex += "\\end{table}\n\n"

        return latex

    def generate_comparison_table(
        self,
        dataset_name: str,
        split_name: str
    ) -> str:
        """
        Generuje tabelę porównawczą manual vs Keras.

        Args:
            dataset_name: Nazwa zbioru danych
            split_name: train_test lub train_val_test

        Returns:
            Kod LaTeX tabeli
        """
        manual_file = os.path.join(
            self.results_dir,
            f"manual_perceptron_wyniki_{dataset_name}_{split_name}.xlsx"
        )
        keras_file = os.path.join(
            self.results_dir,
            f"keras_wyniki_{dataset_name}_{split_name}.xlsx"
        )

        if not os.path.exists(manual_file) or not os.path.exists(keras_file):
            return f"% Brak plików porównawczych dla {dataset_name}_{split_name}\n"

        manual_df = pd.read_excel(manual_file, sheet_name="Results")
        keras_df = pd.read_excel(keras_file, sheet_name="Results")

        is_regression = "regression" in dataset_name

        # Znajdź najlepsze wyniki
        if is_regression:
            manual_best = manual_df.loc[manual_df["test_mse"].idxmin()]
            keras_best = keras_df.loc[keras_df["test_mse"].idxmin()]

            latex = "\\begin{table}[H]\n"
            latex += "\\centering\n"
            latex += "\\begin{tabular}{lcc}\n"
            latex += "\\toprule\n"
            latex += "\\textbf{Metryka} & \\textbf{Manual MLP} & \\textbf{Keras MLP} \\\\\n"
            latex += "\\midrule\n"
            latex += f"Test MSE & {manual_best['test_mse']:.4f} & {keras_best['test_mse']:.4f} \\\\\n"
            latex += f"Test MAE & {manual_best['test_mae']:.4f} & {keras_best['test_mae']:.4f} \\\\\n"
            latex += f"Test R² & {manual_best['test_r2']:.4f} & {keras_best['test_r2']:.4f} \\\\\n"
            latex += "\\bottomrule\n"
            latex += "\\end{tabular}\n"
        else:
            manual_best = manual_df.loc[manual_df["test_accuracy"].idxmax()]
            keras_best = keras_df.loc[keras_df["test_accuracy"].idxmax()]

            latex = "\\begin{table}[H]\n"
            latex += "\\centering\n"
            latex += "\\begin{tabular}{lcc}\n"
            latex += "\\toprule\n"
            latex += "\\textbf{Metryka} & \\textbf{Manual MLP} & \\textbf{Keras MLP} \\\\\n"
            latex += "\\midrule\n"
            latex += f"Test Accuracy & {manual_best['test_accuracy']:.4f} & {keras_best['test_accuracy']:.4f} \\\\\n"
            latex += f"Test Precision & {manual_best['test_precision']:.4f} & {keras_best['test_precision']:.4f} \\\\\n"
            latex += f"Test Recall & {manual_best['test_recall']:.4f} & {keras_best['test_recall']:.4f} \\\\\n"
            latex += f"Test F1-score & {manual_best['test_f1_score']:.4f} & {keras_best['test_f1_score']:.4f} \\\\\n"
            latex += "\\bottomrule\n"
            latex += "\\end{tabular}\n"

        dataset_title = dataset_name.replace('_', ' ').title()
        split_title = split_name.replace('_', '/')

        latex += f"\\caption{{Porównanie Manual vs Keras - {dataset_title} ({split_title})}}\n"
        latex += "\\end{table}\n\n"

        return latex

    def generate_figure_reference(
        self,
        figure_path: str,
        caption: str,
        label: str = None,
        width: float = 0.9
    ) -> str:
        """
        Generuje kod LaTeX dla wstawienia figury.

        Args:
            figure_path: Ścieżka do pliku (relatywna do report/)
            caption: Podpis figury
            label: Etykieta do referencji (opcjonalna)
            width: Szerokość figury (0-1, jako fraction of textwidth)

        Returns:
            Kod LaTeX figury
        """
        latex = "\\begin{figure}[H]\n"
        latex += "\\centering\n"
        latex += f"\\includegraphics[width={width}\\textwidth]{{{figure_path}}}\n"
        latex += f"\\caption{{{caption}}}\n"

        if label:
            latex += f"\\label{{fig:{label}}}\n"

        latex += "\\end{figure}\n\n"

        return latex

    def generate_all_tables_for_dataset(
        self,
        dataset_name: str,
        split_name: str = "train_val_test"
    ) -> str:
        """
        Generuje wszystkie tabele dla danego zbioru danych.

        Returns:
            Kod LaTeX ze wszystkimi tabelami
        """
        latex = f"% Tabele dla: {dataset_name} - {split_name}\n\n"

        # Tabele konfiguracji
        latex += "\\subsubsection{Manual MLP}\n\n"
        latex += self.generate_best_config_table(dataset_name, split_name, "manual")

        latex += "\\subsubsection{Keras MLP}\n\n"
        latex += self.generate_best_config_table(dataset_name, split_name, "keras")

        # Tabela porównawcza
        latex += "\\subsubsection{Porównanie}\n\n"
        latex += self.generate_comparison_table(dataset_name, split_name)

        return latex

    def generate_results_section(self) -> str:
        """
        Generuje całą sekcję wyników (z tabelami i figurami).

        Returns:
            Kod LaTeX sekcji wyników
        """
        datasets = [
            ("classification", "Adult Income"),
            ("classification_our", "Loan Approval"),
            ("regression", "Stock Market"),
            ("regression_our", "Student Performance")
        ]

        latex = ""

        for dataset_name, dataset_title in datasets:
            is_regression = "regression" in dataset_name

            latex += f"\\subsection{{{dataset_title}}}\n\n"

            # Najlepsze konfiguracje
            latex += "\\subsubsection{Najlepsze konfiguracje}\n\n"
            latex += self.generate_all_tables_for_dataset(dataset_name, "train_val_test")

            # Learning curves
            latex += "\\subsubsection{Learning Curves}\n\n"

            # Manual
            manual_lc_path = f"../results/visualizations/Manual_MLP_-_{dataset_title.replace(' ', '_')}_train_val_test_learning_curves.png"
            latex += self.generate_figure_reference(
                manual_lc_path,
                f"Learning curves - Manual MLP, {dataset_title}",
                f"manual_lc_{dataset_name}"
            )

            # Keras
            keras_lc_path = f"../results/visualizations/Keras_MLP_-_{dataset_title.replace(' ', '_')}_train_val_test_learning_curves.png"
            latex += self.generate_figure_reference(
                keras_lc_path,
                f"Learning curves - Keras MLP, {dataset_title}",
                f"keras_lc_{dataset_name}"
            )

            # Confusion matrix (klasyfikacja) lub scatter (regresja)
            if not is_regression:
                latex += "\\subsubsection{Confusion Matrix}\n\n"

                manual_cm_path = f"../results/visualizations/Manual_MLP_-_{dataset_title.replace(' ', '_')}_Test_Set_confusion_matrix.png"
                latex += self.generate_figure_reference(
                    manual_cm_path,
                    f"Confusion Matrix - Manual MLP, {dataset_title}",
                    f"manual_cm_{dataset_name}"
                )

                keras_cm_path = f"../results/visualizations/Keras_MLP_-_{dataset_title.replace(' ', '_')}_Test_Set_confusion_matrix.png"
                latex += self.generate_figure_reference(
                    keras_cm_path,
                    f"Confusion Matrix - Keras MLP, {dataset_title}",
                    f"keras_cm_{dataset_name}"
                )
            else:
                latex += "\\subsubsection{Predykcja vs Rzeczywiste}\n\n"

                manual_scatter_path = f"../results/visualizations/Manual_MLP_-_{dataset_title.replace(' ', '_')}_Test_Set_scatter.png"
                latex += self.generate_figure_reference(
                    manual_scatter_path,
                    f"Scatter plot - Manual MLP, {dataset_title}",
                    f"manual_scatter_{dataset_name}"
                )

                keras_scatter_path = f"../results/visualizations/Keras_MLP_-_{dataset_title.replace(' ', '_')}_Test_Set_scatter.png"
                latex += self.generate_figure_reference(
                    keras_scatter_path,
                    f"Scatter plot - Keras MLP, {dataset_title}",
                    f"keras_scatter_{dataset_name}"
                )

            latex += "\n"

        return latex


if __name__ == "__main__":
    """Przykład użycia."""
    generator = LaTeXGenerator()

    print("="*80)
    print("Generowanie fragmentów LaTeX-a dla raportu")
    print("="*80)
    print()

    # Wygeneruj sekcję wyników
    results_section = generator.generate_results_section()

    # Zapisz do pliku
    output_file = "report/wyniki_generated.tex"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(results_section)

    print(f"✅ Wygenerowano sekcję wyników do: {output_file}")
    print()
    print("Możesz teraz wkleić zawartość tego pliku do sekcji 'Wyniki i analiza' w raport.tex")
    print()