
# Neural Network for Classification & Regression

## Overview
This project is a simple, educational implementation of a multi-layer neural network (MLP) in Python using only NumPy. It is designed for learning and experimenting with neural networks for both classification and regression tasks.

---

## Features
- Pure Python & NumPy implementation
- Multi-layer perceptron (MLP) with customizable architecture
- Supports classification (e.g. XOR) and can be easily extended for regression
- Modular code: layers, losses, optimizers, utils
- Simple training loop and prediction interface
- Example: solving the XOR problem

---

## Getting Started

### Requirements
- Python 3.8+
- NumPy

Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage Example
Run the main script to train and test the network on XOR data:
```bash
python main.py
```

You will see training progress and sample predictions formatted in a readable table.

---

## Project Structure

```
NeuralNetwork/
├── main.py              # Main script: training & evaluation
├── requirements.txt     # Dependencies
├── README.md            # Project info
├── data/
│   └── xor.py           # XOR data generator
├── nn/
│   ├── __init__.py      # Package init
│   ├── layers.py        # Layer and activation functions
│   ├── losses.py        # Loss functions
│   ├── mlp.py           # MLP class
│   ├── optim.py         # Optimizers (e.g. Adam)
│   └── utils.py         # Utility functions
└── tests/
	└── test_mlp.py      # Example tests
```

---

## Authors

- **Jakub Sornat**
- **Maciej Tajs**
- **Bartłomiej Sadza**

---

## Planned Features
- Add support for regression tasks (linear output, MSE loss)
- More activation functions (tanh, softmax)
- More loss functions (cross-entropy, MSE)
- Mini-batch training and data shuffling
- Model saving/loading
- More example datasets
- Unit tests for all modules

---

## License
This project is for educational purposes.