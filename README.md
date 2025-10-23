

# Neural Network Playground: Classification & Regression

## Overview
This project is a modular, educational framework for experimenting with neural networks in Python. It supports both classic (NumPy-based) and modern (Keras-based) models for classification and regression tasks, with a clean CLI and comprehensive test suite.

---

## Features

- **Modular architecture**: Easily extendable for new models and tasks
- **MLP (NumPy)**: Custom Multilayer Perceptron implementation
- **RNN (Keras)**: LSTM/GRU/vanilla RNN via TensorFlow Keras
- **CNN (Keras)**: Convolutional Neural Network via TensorFlow Keras
- **Data preprocessing**: Utilities for handling missing values, encoding, normalization (see `data/preprocessing.py`)
- **Rich CLI**: User-friendly console interface using the `rich` library
- **Comprehensive unit tests**: High coverage for all modules
- **Reproducibility**: Utilities for random seeds and logging
- **Example datasets**: XOR, classification, and regression CSVs

---

## Getting Started


### Requirements


#### Create a Virtual Environment (Recommended)

On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```
On Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```


### Usage Example

Run the main script and select a model from the menu:
```bash
python main.py
```

You will see a menu:

```
╭────────────────────────────────────────────────────────────╮
│ Neural Network Model Selection                            │
├──────────┬───────────────────────────────────────────────┤
│ 1        │ MLP (Multilayer Perceptron, numpy)            │
│ 2        │ RNN (Keras LSTM)                              │
│ 3        │ CNN (Keras Conv2D)                            │
╰──────────┴───────────────────────────────────────────────╯
```
Choose a model to see a demo run with example data. Results are shown in a formatted table or panel.

---


## Running Tests

To run all unit tests (recommended):

```powershell
pip install pytest
$env:PYTHONPATH = (Get-Location)
pytest
```

Test coverage includes all modules: MLP, Keras models, layers, activations, losses, optimizers, utils, and data.

---

## Project Structure

```
NeuralNetwork/
│   main.py                # Entry point, CLI, model selection
│   requirements.txt       # Dependencies
│   README.md
│
├── nn/                    # Neural network modules
│   ├── mlp.py             # NumPy MLP implementation
│   ├── keras_cnn.py       # Keras CNN implementation
│   ├── keras_rnn.py       # Keras RNN implementation
│   ├── activationFunctions.py
│   ├── layers.py
│   ├── losses.py
│   ├── optim.py
│   ├── utils.py
│   └── formatting.py
│
├── data/                  # Data and preprocessing
│   ├── xor.py             # XOR data generator
│   ├── classification.csv # Example classification data
│   ├── regression.csv     # Example regression data
│   └── preprocessing.py   # Data cleaning, encoding, normalization
│
├── tests/                 # Unit and integration tests
│   ├── test_mlp.py
│   ├── test_keras_keras.py
│   ├── test_layers.py
│   ├── test_activationFunctions.py
│   ├── test_losses.py
│   ├── test_optim.py
│   ├── test_utils.py
│   └── test_mlp_module.py
│
└── console_ui.py          # Rich-based CLI interface
```

---


## Authors

- **Jakub Sornat**
- **Maciej Tajs**
- **Bartłomiej Sadza**

---

## License
This project is for educational purposes only.