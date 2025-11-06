import numpy as np
from typing import List


class OptimizerSGD:
    """
    Stochastic Gradient Descent optimizer z opcjonalnym momentum.

    Args:
        learning_rate: Współczynnik uczenia
        decay: Decay learning rate (zmniejszanie w czasie)
        momentum: Współczynnik momentum (0 = brak momentum)
    """

    def __init__(self, learning_rate: float = 1.0, decay: float = 0.0, momentum: float = 0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        """Aktualizacja learning rate z decay"""
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    def update_params(self, layer):
        """
        Aktualizacja wag i biasów warstwy.

        Args:
            layer: Warstwa z atrybutami weights, biases, dweights, dbiases
        """
        if self.momentum:
            # Inicjalizacja momentum jeśli nie istnieje
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Aktualizacja momentum
            weight_updates = (
                self.momentum * layer.weight_momentums
                - self.current_learning_rate * layer.dweights
            )
            layer.weight_momentums = weight_updates

            bias_updates = (
                self.momentum * layer.bias_momentums
                - self.current_learning_rate * layer.dbiases
            )
            layer.bias_momentums = bias_updates
        else:
            # Zwykłe SGD bez momentum
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        # Aktualizacja wag i biasów
        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        """Inkrementacja licznika iteracji"""
        self.iterations += 1


class OptimizerAdam:
    """
    Adam (Adaptive Moment Estimation) optimizer.

    Args:
        learning_rate: Współczynnik uczenia
        decay: Decay learning rate
        epsilon: Wartość zapobiegająca dzieleniu przez zero
        beta_1: Exponential decay rate dla pierwszego momentu
        beta_2: Exponential decay rate dla drugiego momentu
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        decay: float = 0.0,
        epsilon: float = 1e-7,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
    ):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        """Aktualizacja learning rate z decay"""
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    def update_params(self, layer):
        """
        Aktualizacja wag i biasów warstwy używając Adam.

        Args:
            layer: Warstwa z atrybutami weights, biases, dweights, dbiases
        """
        # Inicjalizacja cache jeśli nie istnieje
        if not hasattr(layer, "weight_cache"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Aktualizacja momentum (pierwszego momentu)
        layer.weight_momentums = (
            self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        )
        layer.bias_momentums = (
            self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        )

        # Korekcja bias dla momentum
        weight_momentums_corrected = layer.weight_momentums / (
            1 - self.beta_1 ** (self.iterations + 1)
        )
        bias_momentums_corrected = layer.bias_momentums / (
            1 - self.beta_1 ** (self.iterations + 1)
        )

        # Aktualizacja cache (drugiego momentu)
        layer.weight_cache = (
            self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        )
        layer.bias_cache = (
            self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        )

        # Korekcja bias dla cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Aktualizacja wag i biasów
        layer.weights += (
            -self.current_learning_rate
            * weight_momentums_corrected
            / (np.sqrt(weight_cache_corrected) + self.epsilon)
        )
        layer.biases += (
            -self.current_learning_rate
            * bias_momentums_corrected
            / (np.sqrt(bias_cache_corrected) + self.epsilon)
        )

    def post_update_params(self):
        """Inkrementacja licznika iteracji"""
        self.iterations += 1


class OptimizerRMSprop:
    """
    RMSprop (Root Mean Square Propagation) optimizer.

    Args:
        learning_rate: Współczynnik uczenia
        decay: Decay learning rate
        epsilon: Wartość zapobiegająca dzieleniu przez zero
        rho: Decay rate dla moving average kwadratów gradientów
    """

    def __init__(
        self, learning_rate: float = 0.001, decay: float = 0.0, epsilon: float = 1e-7, rho: float = 0.9
    ):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):
        """Aktualizacja learning rate z decay"""
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    def update_params(self, layer):
        """
        Aktualizacja wag i biasów warstwy używając RMSprop.

        Args:
            layer: Warstwa z atrybutami weights, biases, dweights, dbiases
        """
        # Inicjalizacja cache jeśli nie istnieje
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Aktualizacja cache (moving average kwadratów gradientów)
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2

        # Aktualizacja wag i biasów
        layer.weights += (
            -self.current_learning_rate
            * layer.dweights
            / (np.sqrt(layer.weight_cache) + self.epsilon)
        )
        layer.biases += (
            -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
        )

    def post_update_params(self):
        """Inkrementacja licznika iteracji"""
        self.iterations += 1
