 # Loss functions
import numpy as np

def bce_loss(y_hat, y_true, eps=1e-7):
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat))
