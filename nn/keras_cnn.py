"""
Keras-based Convolutional Neural Network (CNN) module
"""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

class KerasCNN:
    def __init__(self, input_shape=(6, 6, 1), num_filters=8, filter_size=3, pool_size=2, output_dim=1, seed=42):
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.output_dim = output_dim
        keras.utils.set_random_seed(seed)
        self.model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(num_filters, filter_size, activation='relu'),
            layers.MaxPooling2D(pool_size=pool_size),
            layers.Flatten(),
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
