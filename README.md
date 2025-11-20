

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
├── main.py, main_keras.py, main_fashion_mnist.py, main_regression_advanced.py
├── run_all_experiments.py
├── requirements.txt
├── src/
│   ├── manual_mlp/         # Manual MLP implementation
│   └── models/             # Keras models: MLP, CNN, LSTM
├── utils/
│   ├── experiment_runner.py
│   ├── keras_experiment_runner.py
│   └── visualization.py
├── data/                   # Datasets and preprocessing
├── results/                # Results (Excel, plots)
│   └── visualizations/
├── report/
│   ├── raport.tex, literatura_template.md
│   └── Makefile
├── tests/                  # Unit tests (optional)
├── README.md, JAK_URUCHOMIC.md, PODSUMOWANIE_PRACY.md, INSTRUKCJA_UZUPELNIENIA.md
├── wytyczne_do_projektu.pdf
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
    python main.py
    ```
2. **Keras MLP:**
    ```bash
    python main_keras.py
    ```
3. **Fashion MNIST (MLP + CNN):**
    ```bash
    python main_fashion_mnist.py
    ```
4. **Advanced regressions (CNN-1D + LSTM):**
    ```bash
    python main_regression_advanced.py
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
