 # Optimizers
import numpy as np

def init_adam_state(params):
    state = []
    for p in params:
        mW = np.zeros_like(p["W"]); vW = np.zeros_like(p["W"])
        mb = np.zeros_like(p["b"]); vb = np.zeros_like(p["b"])
        state.append({"mW": mW, "vW": vW, "mb": mb, "vb": vb})
    return state

def adam_step(params, grads, state, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
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
