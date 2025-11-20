import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Ścieżka zapisu
output_dir = "report/parameter_visualizations"
os.makedirs(output_dir, exist_ok=True)

# Wczytanie plików klasyfikacji (wspólny zbiór)
mp_80_20      = pd.read_excel('results/manual_perceptron_wyniki_classification_train_test.xlsx')
mp_70_15_15   = pd.read_excel('results/manual_perceptron_wyniki_classification_train_val_test.xlsx')
keras_80_20   = pd.read_excel('results/keras_wyniki_classification_train_test.xlsx')
keras_70_15_15= pd.read_excel('results/keras_wyniki_classification_train_val_test.xlsx')

# Znalezienie wspólnych kolumn
common_cols = list(set(mp_80_20.columns) & set(keras_80_20.columns))

mp_80_20      = mp_80_20[common_cols]
mp_70_15_15   = mp_70_15_15[common_cols]
keras_80_20   = keras_80_20[common_cols]
keras_70_15_15= keras_70_15_15[common_cols]

# Połączenie w jeden dataframe
aggregated = pd.concat(
    [mp_80_20, mp_70_15_15, keras_80_20, keras_70_15_15],
    ignore_index=True
)

# Parametry i metryki
metrics = ["test_accuracy", "test_precision", "test_recall"]
params  = ["learning_rate", "n_hidden_layers", "n_neurons"]

for param in params:
    grouped = aggregated.groupby(param)[metrics].mean().reset_index()

    x = np.arange(len(grouped[param]))  # pozycje na osi X
    width = 0.25                        # szerokość jednego słupka

    # Rozmiar pod Worda: dwa wykresy obok siebie
    plt.figure(figsize=(7, 6))

    for i, metric in enumerate(metrics):
        plt.bar(
            x + i * width,
            grouped[metric],
            width=width,
            label=metric
        )

    plt.xticks(x + width, grouped[param])
    plt.xlabel(param)
    plt.ylabel("value")
    plt.title(f"Classification metrics vs {param} - common dataset")
    plt.ylim(0.8, 0.83)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

plt.show()