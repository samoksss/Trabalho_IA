"""
Etapa 1 — Treino da MLP em NumPy no PathMNIST (28x28 achatado).

Roda na máquina de vocês (precisa de internet na 1a vez p/ baixar o dataset):
    python -m src.etapa1_numpy.train

Gera:
- results/logs/etapa1_numpy.csv   (loss de treino e acc de validação por época)
- results/figures/etapa1_loss.png (curva de perda)
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.etapa1_numpy.mlp import MLP, SGDMomentum, gradient_check, softmax_cross_entropy
from src.utils.seeds import set_seed

LOG_DIR = "results/logs"
FIG_DIR = "results/figures"


def iterate_minibatches(X, y, batch_size, rng):
    idx = rng.permutation(len(X))
    for start in range(0, len(X), batch_size):
        b = idx[start:start + batch_size]
        yield X[b], y[b]


def train(X_tr, y_tr, X_val, y_val, layer_sizes,
          epochs=20, batch_size=128, lr=0.1, beta=0.9, seed=42, verbose=True):
    """Treina a MLP e retorna (modelo, histórico)."""
    set_seed(seed)
    model = MLP(layer_sizes, activation="relu", seed=seed)
    opt = SGDMomentum(model, lr=lr, beta=beta)
    rng = np.random.default_rng(seed)

    # Gradient check no modelo real (exigência da Seção 4.1) — em um mini-batch
    rel = gradient_check(model, X_tr[:64], y_tr[:64])
    if verbose:
        print(f"[grad-check] diferenca relativa = {rel:.2e}  (esperado << 1e-4)")

    history = []
    for epoch in range(1, epochs + 1):
        losses = []
        for xb, yb in iterate_minibatches(X_tr, y_tr, batch_size, rng):
            loss, dW, db = model.loss_and_grads(xb, yb)
            opt.step(dW, db)
            losses.append(loss)
        train_loss = float(np.mean(losses))
        val_acc = float((model.predict(X_val) == y_val).mean())
        history.append((epoch, train_loss, val_acc))
        if verbose:
            print(f"epoca {epoch:2d} | loss {train_loss:.4f} | val_acc {val_acc:.4f}")
    return model, history


def save_outputs(history):
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "etapa1_numpy.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoca", "train_loss", "val_acc"])
        w.writerows(history)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ep = [h[0] for h in history]
        plt.figure(figsize=(6, 4))
        plt.plot(ep, [h[1] for h in history], label="train loss")
        plt.plot(ep, [h[2] for h in history], label="val acc")
        plt.xlabel("epoca"); plt.legend(); plt.title("Etapa 1 — MLP NumPy")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "etapa1_loss.png"), dpi=120)
        print(f"[ok] figura salva em {FIG_DIR}/etapa1_loss.png")
    except ImportError:
        print("[aviso] matplotlib nao instalado — pulei a figura.")


def main():
    from src.utils.data import get_numpy_flat, N_CLASSES

    X_tr, y_tr = get_numpy_flat("train")
    X_val, y_val = get_numpy_flat("val")
    print(f"[data] treino: {X_tr.shape} | validacao: {X_val.shape}")

    layer_sizes = [X_tr.shape[1], 256, 128, N_CLASSES]   # 2352 -> 256 -> 128 -> 9
    _, history = train(X_tr, y_tr, X_val, y_val, layer_sizes,
                       epochs=20, batch_size=128, lr=0.1, beta=0.9)
    save_outputs(history)


if __name__ == "__main__":
    main()
