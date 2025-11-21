

# Neural Networks

**A scientific project developed as part of a course/thesis.**

This project is a modular, educational framework for experimenting with neural networks in Python. It supports both classic NumPy-based models and modern Keras models for classification and regression tasks, offering a clear CLI interface and an extensive set of tests.

The project meets academic guidelines (see: `wytyczne_do_projektu.pdf`) and is ready for extension with new architectures, datasets, and analysis methods.

---

## Table of Contents
1. [Project Goal](#project-goal)
2. [Features](#features)
3. [Directory Structure](#directory-structure)
4. [Getting Started](#getting-started)
5. [Running Experiments](#running-experiments)
6. [Visualizations](#visualizations)
7. [What to Add in the Report](#what-to-add-in-the-report)
8. [Estimated Work Time](#estimated-work-time)
9. [Troubleshooting](#troubleshooting)
10. [Unit Tests (Optional)](#unit-tests-optional)
11. [Authors](#authors)

---

## Project Goal

This project was created as a scientific work aiming to:
- compare a custom neural network implementation with Keras solutions,
- analyze the impact of hyperparameters on results,
- automate experiments and reporting,
- draw conclusions useful for further research in machine learning.

---

## Features
- Manual MLP implementation (NumPy)
- Keras models: MLP, 2D CNN, 1D CNN, LSTM
- Automatic grid search over hyperparameters
- Experiments on 5 datasets (tabular and image)
- Export results to Excel
- Generate plots (learning curves, confusion matrix, scatter)
- Detailed run instructions
- Unit tests (optional)

---

## Directory Structure
```
NeuralNetwork/
├── main.py
├── README.md
├── requirements.txt
├── data/
│   ├── sample_data_generator.py
│   ├── classification/
│   │   ├── adult_preprocessed.csv
│   │   ├── classification.csv
│   │   ├── correlation.py
│   │   ├── preprocessing.py
│   │   ├── train70.csv
│   │   ├── train80.csv
│   │   ├── test15.csv
│   │   ├── test20.csv
│   │   ├── validation15.csv
│   ├── classification_our/
│   │   ├── correlation.py
│   │   ├── loan_approval.csv
│   │   ├── preprocessing.py
│   │   ├── train70.csv
│   │   ├── train80.csv
│   │   ├── test15.csv
│   │   ├── test20.csv
│   │   ├── validation15.csv
│   ├── fashion_mist/
│   │   ├── preprocessing.py
│   │   ├── t10k-images-idx3-ubyte
│   │   ├── t10k-labels-idx1-ubyte
│   │   ├── train-images-idx3-ubyte
│   │   ├── train-labels-idx1-ubyte
│   ├── image_analysis/
│   ├── regression/
│   │   ├── regression_preprocessing.py
│   │   ├── regression.csv
│   │   ├── train70.csv
│   │   ├── train80.csv
│   │   ├── test15.csv
│   │   ├── test20.csv
│   │   ├── validation15.csv
│   ├── regression_our/
│   │   ├── correlation.py
│   │   ├── student_preformance.csv
│   │   ├── preprocessing.py
│   │   ├── train70.csv
│   │   ├── train80.csv
│   │   ├── test15.csv
│   │   ├── test20.csv
│   │   ├── validation15.csv
├── experimetns/
│   ├── advanced_regression_experiments.py
│   ├── fashion_mnist_experiment.py
│   ├── run_all_experiments.py
│   ├── run_tabular_keras_experiments.py
│   ├── tabular_manual_mlp_experiments.py
├── report/
├── results/
├── src/
│   ├── manual_mlp/
│   │   ├── activations.py
│   │   ├── layers.py
│   │   ├── metrics.py
│   │   ├── model.py
│   ├── models/
│   │   ├── convolutional_nn.py
│   │   ├── keras_cnn_regression.py
│   │   ├── keras_cnn.py
│   │   ├── keras_lstm_regression.py
│   │   ├── keras_mlp.py
│   │   ├── multilayer_perceptron.py
│   │   ├── recurrent_nn.py
├── tests/
├── utils/
    ├── experiment_runner.py
    ├── keras_experiment_runner.py
    ├── results_exporter.py
    ├── seed.py
    ├── visualizations/
        ├── classification_parameter_impact_visualization.py
        ├── generate_visualizations_extended.py
        ├── generate_visualizations.py
        ├── our_classification_parameter_impact_visualizations.py
        ├── our_regression_parameter_impact_visualizations.py
        ├── visualization.py
```

---

## Getting Started

### Virtual Environment (Recommended)

A virtual environment allows you to isolate the project's dependencies from the rest of your system and avoid conflicts between different Python projects. This ensures that all packages are compatible with the project's requirements.

**Creating and activating the environment:**

On Windows:
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```
On Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

**Deactivating the environment:**
```bash
deactivate
```

---

1. Create and activate a virtual environment (see above).
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run all experiments (6–10h):
    ```bash
    python run_all_experiments.py
    ```

---

## Running Experiments

1. **Manual MLP (if not run before):**
    ```bash
    python experimetns/run_tabular_manual_mlp_experiments.py
    ```
2. **Keras MLP:**
    ```bash
    python experimetns/run_tabular_keras_experiments.py
    ```
3. **Fashion MNIST (MLP + CNN):**
    ```bash
    python experimetns/run_fashion_mnist_experiment.py
    ```
4. **Advanced regressions (CNN-1D + LSTM):**
    ```bash
    python experimetns/runn_advanced_regression_experiments.py
    ```

Result: ~21 Excel files in `results/`.

---

## Visualizations

- **Basic (Manual + Keras MLP):**
    ```bash
    python generate_visualizations.py
    ```
- **Extended (CNN, LSTM, Fashion MNIST):**
    ```bash
    python generate_visualizations_extended.py
    ```

Result: ~40–46 PNG files in `results/visualizations/`.

---

## Troubleshooting

- **Missing packages:**
    ```bash
    pip install -r requirements.txt
    ```
- **Experiments too slow:**
    Reduce grids:
    ```python
    HIDDEN_LAYERS_GRID = [2, 3]
    NEURONS_GRID = [16, 32]
    ```
- **Out of RAM:**
    Reduce batch_size in the relevant files.
- **TensorFlow warnings:**
    You can ignore binary optimization warnings.
- **Visualizations not displaying:**
    Check the paths to PNG files.

---

## Unit Tests (Optional)
```bash
pip install pytest
pytest tests/
```

---

## Authors

- Jakub Sornat
- Maciej Tajs
- Bartłomiej Sadza

---
