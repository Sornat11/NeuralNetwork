"""
Keras-based Recurrent Neural Network (RNN) module
"""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

class KerasRNN:
    def __init__(self, input_shape=(10, 1), hidden_dim=16, output_dim=1, cell_type='SimpleRNN', seed=42):
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        keras.utils.set_random_seed(seed)
        if cell_type == 'LSTM':
            rnn_layer = layers.LSTM(hidden_dim)
        elif cell_type == 'GRU':
            rnn_layer = layers.GRU(hidden_dim)
        else:
            rnn_layer = layers.SimpleRNN(hidden_dim)
        self.model = keras.Sequential([
            layers.Input(shape=input_shape),
            rnn_layer,
            layers.Dense(output_dim, activation='sigmoid' if output_dim == 1 else 'softmax')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy' if output_dim == 1 else 'categorical_crossentropy', metrics=['accuracy'])

    def train(self, X, y, epochs=10, batch_size=8):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self, X):
        preds = self.model.predict(X, verbose=0)
        if self.output_dim == 1:
            return (preds > 0.5).astype(int)
        return np.argmax(preds, axis=1)

    def summary(self):
        self.model.summary()
