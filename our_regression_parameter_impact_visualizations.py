import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Regression – our

our_80_20 = pd.read_excel('results/manual_perceptron_wyniki_regression_our_train_test.xlsx')
our_70_15_15 = pd.read_excel('results/manual_perceptron_wyniki_regression_our_train_val_test.xlsx')
keras_80_20 = pd.read_excel('results/keras_wyniki_regression_our_train_test.xlsx')
keras_70_15_15 = pd.read_excel('results/keras_wyniki_regression_our_train_val_test.xlsx')

common = list(set(our_80_20.columns) & set(keras_80_20.columns))

our_80_20 = our_80_20[common]
our_70_15_15 = our_70_15_15[common]
keras_80_20 = keras_80_20[common]
keras_70_15_15 = keras_70_15_15[common]

aggregated = pd.concat(
    [our_80_20, our_70_15_15, keras_80_20, keras_70_15_15],
    ignore_index=True
)

metrics = ["test_mse", "test_mae"]
params  = ["learning_rate", "n_hidden_layers", "n_neurons"]

out_dir = "report/parameter_visualizations"
os.makedirs(out_dir, exist_ok=True)

for param in params:
    # średnie wartości metryk dla każdej wartości parametru
    grouped = aggregated.groupby(param)[metrics].max().reset_index()

    x = np.arange(len(grouped[param]))
    width = 0.4

    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)  

    for i, metric in enumerate(metrics):
        ax.bar(
            x + i * width,
            grouped[metric],
            width=width,
            label=metric
        )

    ax.set_xticks(x + width)
    ax.set_xticklabels(grouped[param])
    ax.set_xlabel(param)
    ax.set_ylabel("value")
    ax.set_title(f"Regression metrics vs {param} - our dataset")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    fig.tight_layout()
plt.show()