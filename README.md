

# Neural Network Playground: Classification & Regression

## Overview
This project is a modular, educational framework for experimenting with neural networks in Python. It supports both classic (NumPy-based) and modern (Keras-based) models for classification and regression tasks, with a clean CLI and comprehensive test suite.

---

## Features


---

## Getting Started


### Requirements


#### Create a Virtual Environment (Recommended)

On Windows:
```bash
python -m venv venv
 .\.venv\Scripts\Activate.ps1
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

---


## Authors

- **Jakub Sornat**
- **Maciej Tajs**
- **Bartłomiej Sadza**

---

## License
This project is for educational purposes only.