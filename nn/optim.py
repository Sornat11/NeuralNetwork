"""
Optimizers for neural network training.
Includes: SGD, Adam.
All functions use NumPy arrays as input.
"""

import numpy as np

def sgd_step(params, grads, lr=1e-3, weight_decay=0.0):
    """
    Stochastic Gradient Descent (SGD) optimizer step.
    Args:
        params (list): List of parameter dicts (with 'W', 'b').
        grads (list): List of gradient dicts (with 'dW', 'db').
        lr (float): Learning rate.
        weight_decay (float): L2 regularization coefficient.
    Returns:
        list: Updated parameters.
    """
    for p, g in zip(params, grads):
        p['W'] -= lr * (g['dW'] + weight_decay * p['W'])
        p['b'] -= lr * g['db']
    return params

def init_adam_state(params):
    """
    Initialize Adam optimizer state (momentum and RMSProp terms).
    Args:
        params (list): List of parameter dicts (with 'W', 'b').
    Returns:
        list: State dicts for Adam optimizer.
    """
    state = []
    for p in params:
        mW = np.zeros_like(p["W"])
        vW = np.zeros_like(p["W"])
        mb = np.zeros_like(p["b"])
        vb = np.zeros_like(p["b"])
        state.append({"mW": mW, "vW": vW, "mb": mb, "vb": vb})
    return state

def adam_step(params, grads, state, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
    """
    Adam optimizer step.
    Args:
        params (list): List of parameter dicts (with 'W', 'b').
        grads (list): List of gradient dicts (with 'dW', 'db').
        state (list): Adam state dicts (with 'mW', 'vW', 'mb', 'vb').
        t (int): Current timestep (epoch or batch count).
        lr (float): Learning rate.
        beta1 (float): Exponential decay rate for first moment estimates.
        beta2 (float): Exponential decay rate for second moment estimates.
        eps (float): Small value to avoid division by zero.
        weight_decay (float): L2 regularization coefficient.
    Returns:
        tuple: (updated params, updated state)
    """
    for p, g, s in zip(params, grads, state):
        s["mW"] = beta1 * s["mW"] + (1 - beta1) * g["dW"]
        s["vW"] = beta2 * s["vW"] + (1 - beta2) * (g["dW"] ** 2)
        mW_hat = s["mW"] / (1 - beta1 ** t)
        vW_hat = s["vW"] / (1 - beta2 ** t)
        p["W"] -= lr * (mW_hat / (np.sqrt(vW_hat) + eps) + weight_decay * p["W"])
        s["mb"] = beta1 * s["mb"] + (1 - beta1) * g["db"]
        s["vb"] = beta2 * s["vb"] + (1 - beta2) * (g["db"] ** 2)
        mb_hat = s["mb"] / (1 - beta1 ** t)
        vb_hat = s["vb"] / (1 - beta2 ** t)
        p["b"] -= lr * (mb_hat / (np.sqrt(vb_hat) + eps))
    return params, state
