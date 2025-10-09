import numpy as np

# --- aktywacje ---
def relu(x): return np.maximum(0.0, x)
def drelu(y):  # y = relu(x)
    g = np.zeros_like(y); g[y > 0] = 1.0; return g

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

# --- inicjalizacja wag (Xavier/He zależnie od aktywacji) ---
def init_params(sizes, hidden_act="relu", seed=42):
    rng = np.random.default_rng(seed)
    params = []
    for i in range(len(sizes)-1):
        fan_in, fan_out = sizes[i], sizes[i+1]
        if hidden_act.lower() == "relu" and i < len(sizes)-2:
            scale = np.sqrt(2.0 / fan_in)  # He
        else:
            scale = np.sqrt(1.0 / fan_in)  # Xavier
        W = rng.normal(0.0, scale, size=(fan_out, fan_in))
        b = np.zeros((fan_out, 1))
        params.append({"W": W, "b": b})
    return params

# --- forward (z buforami do backprop) ---
def forward(params, X, hidden_act="relu"):
    """
    X: (features, batch)
    Zs, As: listy po warstwach (As[0] = X)
    """
    A = X
    Zs, As = [], [X]
    L = len(params)
    for l, p in enumerate(params):
        Z = p["W"] @ A + p["b"]
        if l < L-1:
            A = relu(Z) if hidden_act == "relu" else np.tanh(Z)
        else:
            A = sigmoid(Z)  # ostatnia: sigmoid do BCE
        Zs.append(Z); As.append(A)
    return Zs, As

# --- strata BCE + stabilizacja ---
def bce_loss(y_hat, y_true, eps=1e-7):
    """
    y_hat, y_true: (1, batch)
    """
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat))

# --- backprop ---
def backward(params, Zs, As, y_true, hidden_act="relu"):
    """
    Zs, As z forward; y_true: (1, batch)
    Zwraca grads: lista słowników dW, db o tych samych kształtach co params.
    """
    L = len(params)
    m = y_true.shape[1]
    grads = [None] * L

    # dL/dA_L dla BCE + sigmoid: (y_hat - y)
    dA = (As[-1] - y_true)  # (1, m)

    for l in reversed(range(L)):
        A_prev = As[l]       # (n_{l-1}, m)
        Z = Zs[l]
        W = params[l]["W"]

        if l < L-1:
            if hidden_act == "relu":
                dZ = dA * drelu(As[l+1])
            else:
                # dla tanh: derivative = 1 - tanh^2
                dZ = dA * (1.0 - np.square(As[l+1]))
        else:
            # wyjściowa: już uwzględniliśmy pochodną sigmoid w (y_hat - y)
            dZ = dA

        dW = (dZ @ A_prev.T) / m
        db = np.mean(dZ, axis=1, keepdims=True)
        dA = (W.T @ dZ)
        grads[l] = {"dW": dW, "db": db}

    return grads

# --- Adam (z korekcją biasu i opcjonalnym weight decay w stylu AdamW) ---
def init_adam_state(params):
    state = []
    for p in params:
        mW = np.zeros_like(p["W"]); vW = np.zeros_like(p["W"])
        mb = np.zeros_like(p["b"]); vb = np.zeros_like(p["b"])
        state.append({"mW": mW, "vW": vW, "mb": mb, "vb": vb})
    return state

def adam_step(params, grads, state, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
    """
    AdamW: weight_decay działa jak L2 na parametrach (decoupled).
    """
    for p, g, s in zip(params, grads, state):
        # W
        s["mW"] = beta1 * s["mW"] + (1 - beta1) * g["dW"]
        s["vW"] = beta2 * s["vW"] + (1 - beta2) * (g["dW"] ** 2)
        mW_hat = s["mW"] / (1 - beta1 ** t)
        vW_hat = s["vW"] / (1 - beta2 ** t)
        # decoupled weight decay
        p["W"] -= lr * (mW_hat / (np.sqrt(vW_hat) + eps) + weight_decay * p["W"])

        # b
        s["mb"] = beta1 * s["mb"] + (1 - beta1) * g["db"]
        s["vb"] = beta2 * s["vb"] + (1 - beta2) * (g["db"] ** 2)
        mb_hat = s["mb"] / (1 - beta1 ** t)
        vb_hat = s["vb"] / (1 - beta2 ** t)
        p["b"] -= lr * (mb_hat / (np.sqrt(vb_hat) + eps))
    return params, state

# --- batching ---
def iterate_minibatches(X, y, batch_size, rng):
    m = X.shape[1]
    perm = rng.permutation(m)
    for i in range(0, m, batch_size):
        idx = perm[i:i+batch_size]
        yield X[:, idx], y[:, idx]

# --- trening ---
def train_mlp(
    X, y,
    sizes=(2, 16, 1),
    hidden_act="relu",
    epochs=2000,
    batch_size=32,
    lr=1e-3,
    beta1=0.9, beta2=0.999, eps=1e-8,
    weight_decay=0.0,
    seed=123
):
    rng = np.random.default_rng(seed)
    params = init_params(sizes, hidden_act, seed)
    state = init_adam_state(params)
    t = 0

    for epoch in range(1, epochs+1):
        epoch_loss = 0.0
        batches = 0
        for Xb, yb in iterate_minibatches(X, y, batch_size, rng):
            Zs, As = forward(params, Xb, hidden_act)
            loss = bce_loss(As[-1], yb)
            grads = backward(params, Zs, As, yb, hidden_act)
            t += 1
            params, state = adam_step(params, grads, state, t, lr, beta1, beta2, eps, weight_decay)
            epoch_loss += loss; batches += 1

        if epoch % max(1, epochs // 10) == 0:
            print(f"epoch {epoch:4d} | loss {epoch_loss / max(1,batches):.4f}")
    return params

def predict(params, X, hidden_act="relu", threshold=0.5):
    _, As = forward(params, X, hidden_act)
    p = As[-1]
    return (p >= threshold).astype(np.float32), p

# --- DEMO: XOR ---
if __name__ == "__main__":
    # Dane XOR (powielone, żeby mini-batch miał sens)
    X_base = np.array([[0,0,1,1],
                       [0,1,0,1]], dtype=np.float32)  # (2,4)
    y_base = np.array([[0,1,1,0]], dtype=np.float32)  # (1,4)

    # zróbmy 400 próbek przez losowe powtórzenia i drobny szum (opcjonalny)
    rng = np.random.default_rng(7)
    reps = 100
    X = np.repeat(X_base, reps, axis=1)
    y = np.repeat(y_base, reps, axis=1)
    X = X + rng.normal(0, 0.02, size=X.shape).astype(np.float32)

    # Trening
    params = train_mlp(
        X, y,
        sizes=(2, 16, 1),
        hidden_act="relu",
        epochs=2000,
        batch_size=32,
        lr=5e-3,          # Adam zwykle lubi 1e-3…5e-3
        weight_decay=0.0  # możesz dać np. 1e-4
    )

    # Predykcja na czystym XOR
    X_test = X_base
    y_hat_cls, y_hat_prob = predict(params, X_test, hidden_act="relu")
    for i in range(X_test.shape[1]):
        print(f"{X_test[0,i]:.0f} xor {X_test[1,i]:.0f} -> p={y_hat_prob[0,i]:.3f}  cls={int(y_hat_cls[0,i])}")