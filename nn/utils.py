
"""
Utility functions for reproducibility, logging, and metrics.
"""

import random
import numpy as np
import time

def set_seed(seed: int):
	"""
	Set random seed for reproducibility (Python, NumPy).
	Args:
		seed (int): The seed value to use.
	"""
	random.seed(seed)
	np.random.seed(seed)


def log(message: str):
	"""
	Simple logger that prints a message with a timestamp.
	Args:
		message (str): The message to log.
	"""
	print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def accuracy(y_true, y_pred):
	"""
	Compute accuracy metric for classification tasks.
	Args:
		y_true (array-like): True labels.
		y_pred (array-like): Predicted labels.
	Returns:
		float: Accuracy score (0.0 - 1.0)
	"""
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	return np.mean(y_true == y_pred)
