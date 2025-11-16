import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Our classification

our_80_20 = pd.read_excel('results/manual_perceptron_wyniki_classification_our_train_test.xlsx')
our_70_15_15 = pd.read_excel('results/manual_perceptron_wyniki_classification_our_train_val_test.xlsx')
keras_80_20 = pd.read_excel('results/keras_wyniki_classification_our_train_test.xlsx')
keras_70_15_15 = pd.read_excel('results/keras_wyniki_classification_our_train_val_test.xlsx')

common = list(set(our_80_20.columns) & set(keras_80_20.columns))

our_80_20 = our_80_20[common]
our_70_15_15 = our_70_15_15[common]

aggregated = pd.concat([our_80_20, our_70_15_15,keras_80_20, keras_70_15_15], ignore_index=True)

metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1_score"]
params  = ["learning_rate", "n_hidden_layers", "n_neurons"]

for param in params:
    # grupowanie – średnia jeśli są powtórzenia dla danej wartości parametru
    grouped = aggregated.groupby(param)[metrics].mean().reset_index()

    x = np.arange(len(grouped[param]))  # pozycje na osi X
    width = 0.2  # szerokość słupka

    plt.figure(figsize=(8,5))

    for i, metric in enumerate(metrics):
        plt.bar(
            x + i*width,
            grouped[metric],
            width=width,
            label=metric
        )

    plt.xticks(x + width, grouped[param])
    plt.xlabel(param)
    plt.ylabel("value")
    plt.title(f"Metrics vs {param}")
    plt.ylim(0.82, 0.88) 
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

plt.show()
