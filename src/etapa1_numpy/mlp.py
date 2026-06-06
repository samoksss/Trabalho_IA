"""
Etapa 1 — Rede neural (MLP) do zero, usando APENAS NumPy.

Cobre todas as exigências da Seção 4.1 do enunciado:
- Arquitetura flexível (nº de neurônios configurável por camada).
- Ativações com derivada implementada à mão (ReLU / Tanh).
- Softmax na saída + Entropia Cruzada numericamente estável.
- SGD com Momentum do zero (acúmulo de velocidade).
- Gradient checking: [f(W+eps) - f(W-eps)] / 2*eps.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Ativações e suas derivadas
# ---------------------------------------------------------------------------
def relu(z):
    return np.maximum(0.0, z)


def relu_grad(z):
    return (z > 0.0).astype(z.dtype)


def tanh(z):
    return np.tanh(z)


def tanh_grad(z):
    return 1.0 - np.tanh(z) ** 2


ACTS = {"relu": (relu, relu_grad), "tanh": (tanh, tanh_grad)}


# ---------------------------------------------------------------------------
# Softmax + Entropia Cruzada (estável: desloca pelo máximo antes do exp)
# ---------------------------------------------------------------------------
def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)        # estabilidade
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def softmax_cross_entropy(logits, y):
    """Retorna (loss escalar, probs). y são índices inteiros (N,)."""
    probs = softmax(logits)
    n = logits.shape[0]
    # -log da prob da classe correta, com clip para evitar log(0)
    correct = probs[np.arange(n), y]
    loss = -np.mean(np.log(np.clip(correct, 1e-12, 1.0)))
    return loss, probs


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------
class MLP:
    """MLP totalmente conectada. layer_sizes ex.: [2352, 256, 128, 9]."""

    def __init__(self, layer_sizes, activation="relu", seed=42):
        self.sizes = layer_sizes
        self.act, self.act_grad = ACTS[activation]
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for fin, fout in zip(layer_sizes[:-1], layer_sizes[1:]):
            # Inicialização He (boa para ReLU)
            self.W.append(rng.standard_normal((fin, fout)) * np.sqrt(2.0 / fin))
            self.b.append(np.zeros(fout))

    def forward(self, X):
        """Propaga e guarda cache para o backward. Saída = logits (sem softmax)."""
        a = X
        cache = {"a": [X], "z": []}
        L = len(self.W)
        for i in range(L):
            z = a @ self.W[i] + self.b[i]
            cache["z"].append(z)
            # ativação em todas as ocultas; saída fica como logits crus
            a = self.act(z) if i < L - 1 else z
            cache["a"].append(a)
        return a, cache

    def backward(self, cache, y):
        """Backpropagation manual (regra da cadeia). Retorna grads dW, db."""
        n = y.shape[0]
        L = len(self.W)
        dW = [None] * L
        db = [None] * L

        # Gradiente da Softmax+CE em relação aos logits: (probs - onehot) / N
        _, probs = softmax_cross_entropy(cache["a"][-1], y)
        delta = probs.copy()
        delta[np.arange(n), y] -= 1.0
        delta /= n

        for i in reversed(range(L)):
            a_prev = cache["a"][i]
            dW[i] = a_prev.T @ delta
            db[i] = delta.sum(axis=0)
            if i > 0:                       # propaga o erro para a camada anterior
                da = delta @ self.W[i].T
                delta = da * self.act_grad(cache["z"][i - 1])
        return dW, db

    def loss_and_grads(self, X, y):
        logits, cache = self.forward(X)
        loss, _ = softmax_cross_entropy(logits, y)
        dW, db = self.backward(cache, y)
        return loss, dW, db

    def predict(self, X):
        logits, _ = self.forward(X)
        return logits.argmax(axis=1)


# ---------------------------------------------------------------------------
# Otimizador: SGD com Momentum (do zero)
#   v_t = beta * v_{t-1} + lr * gradW
#   W   = W - v_t
# ---------------------------------------------------------------------------
class SGDMomentum:
    def __init__(self, model, lr=0.1, beta=0.9):
        self.model, self.lr, self.beta = model, lr, beta
        self.vW = [np.zeros_like(w) for w in model.W]   # velocidade acumulada
        self.vb = [np.zeros_like(b) for b in model.b]

    def step(self, dW, db):
        for i in range(len(self.model.W)):
            self.vW[i] = self.beta * self.vW[i] + self.lr * dW[i]
            self.vb[i] = self.beta * self.vb[i] + self.lr * db[i]
            self.model.W[i] -= self.vW[i]
            self.model.b[i] -= self.vb[i]


# ---------------------------------------------------------------------------
# Gradient checking (validação numérica obrigatória — Seção 4.1)
# ---------------------------------------------------------------------------
def gradient_check(model, X, y, eps=1e-5, n_samples=40, seed=0):
    """Compara o gradiente analítico de W[0] com a aproximação numérica.

    Diferença relativa esperada << 1e-4 (idealmente ~1e-7).
    """
    _, dW, _ = model.loss_and_grads(X, y)
    W = model.W[0]
    g_analytic = dW[0]

    rng = np.random.default_rng(seed)
    idx = [
        (rng.integers(W.shape[0]), rng.integers(W.shape[1]))
        for _ in range(n_samples)
    ]
    num, ana = [], []
    for (r, c) in idx:
        orig = W[r, c]
        W[r, c] = orig + eps
        loss_plus, _ = softmax_cross_entropy(model.forward(X)[0], y)
        W[r, c] = orig - eps
        loss_minus, _ = softmax_cross_entropy(model.forward(X)[0], y)
        W[r, c] = orig                       # restaura
        num.append((loss_plus - loss_minus) / (2 * eps))
        ana.append(g_analytic[r, c])

    num, ana = np.array(num), np.array(ana)
    rel = np.linalg.norm(num - ana) / (np.linalg.norm(num) + np.linalg.norm(ana) + 1e-12)
    return rel


# ---------------------------------------------------------------------------
# Autoteste (dados sintéticos — não precisa baixar o PathMNIST)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, d, k = 300, 20, 4
    X = rng.standard_normal((n, d))
    true_W = rng.standard_normal((d, k))
    y = (X @ true_W).argmax(axis=1)          # rótulos linearmente separáveis

    # 1) Gradient check em rede pequena (tanh -> verificação bem limpa)
    net = MLP([d, 16, k], activation="tanh", seed=1)
    rel = gradient_check(net, X[:50], y[:50])
    print(f"[grad-check] diferenca relativa = {rel:.2e}  (esperado << 1e-4)")
    assert rel < 1e-6, "Gradient check falhou!"

    # 2) Treino curto deve reduzir a perda e subir a acurácia
    net = MLP([d, 64, 32, k], activation="relu", seed=1)
    opt = SGDMomentum(net, lr=0.05, beta=0.9)
    for epoch in range(200):
        loss, dW, db = net.loss_and_grads(X, y)
        opt.step(dW, db)
    acc = (net.predict(X) == y).mean()
    print(f"[treino] loss final = {loss:.4f} | acuracia = {acc:.3f}")
