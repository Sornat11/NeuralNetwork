import random

import numpy as np


def set_seed(seed: int = 0):
    """Set the random seed for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
