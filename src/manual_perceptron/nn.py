import numpy as np

from data.sample_data_generator import create_data
from utils.seed import set_seed

set_seed(0)


X, y = create_data(100, 3)


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

layer1.forward(X)


activation1.forward(layer1.output)

print(X, y)
print(activation1.output)
