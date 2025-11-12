def create_regression_data(points, features=1, noise=0.1):
    """
    Generates synthetic regression data: X (features), y (target)
    y = 3 * x + 2 + noise
    """
    X = np.random.rand(points, features)
    coef = np.arange(1, features + 1) * 3  # [3, 6, 9, ...] for multiple features
    y = X @ coef + 2
    y += np.random.randn(points) * noise
    return X, y


import numpy as np

from utils.seed import set_seed

set_seed(0)


def create_data(points, classes):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype="uint8")

    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points)
        t = (
            np.linspace(class_number * 4, (class_number + 1) * 4, points)
            + np.random.randn(points) * 0.2
        )
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number

    return X, y
