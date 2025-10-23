"""
Test KerasCNN and KerasRNN integration with random data.
"""
import numpy as np
from nn.keras_cnn import KerasCNN
from nn.keras_rnn import KerasRNN

def test_keras_cnn_train_predict():
    # Example: grayscale images 8x8, binary classification
    X = np.random.rand(20, 8, 8, 1)
    y = np.random.randint(0, 2, size=(20, 1))
    model = KerasCNN(input_shape=(8, 8, 1), num_filters=4, filter_size=3, pool_size=2, output_dim=1, seed=123)
    model.train(X, y, epochs=2, batch_size=4)
    preds = model.predict(X)
    assert preds.shape == (20, 1) or preds.shape == (20,)

def test_keras_rnn_train_predict():
    # Example: sequence data (batch, timesteps, features), binary classification
    X = np.random.rand(20, 10, 3)
    y = np.random.randint(0, 2, size=(20, 1))
    model = KerasRNN(input_shape=(10, 3), hidden_dim=8, output_dim=1, cell_type='LSTM', seed=123)
    model.train(X, y, epochs=2, batch_size=4)
    preds = model.predict(X)
    assert preds.shape == (20, 1) or preds.shape == (20,)
