
import numpy as np

class Model:
    def __init__(self):
        self.layers: list = []

    def add(self, layer) -> None:
        self.layers.append(layer)

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out