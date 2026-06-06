"""
Etapa 2 — Prova de equivalência NumPy vs PyTorch (Seção 4.2).

Treina a MESMA arquitetura, com os MESMOS dados, hiperparâmetros e seed,
nas duas implementações, e verifica que a diferença de acurácia de
validação é <= 2 pontos percentuais ao final.

Rodar:
    python -m src.etapa2_pytorch.equivalence
"""

import numpy as np

from src.etapa1_numpy.train import train as train_numpy
from src.etapa2_pytorch.mlp_torch import train_torch
from src.utils.seeds import set_seed

GAP_MAXIMO_PP = 2.0


def comparar(X_tr, y_tr, X_val, y_val, layer_sizes,
             epochs=20, batch_size=128, lr=0.1, beta=0.9, seed=42):
    print("=" * 60)
    print("NumPy (Etapa 1)")
    print("=" * 60)
    set_seed(seed)
    _, hist_np = train_numpy(X_tr, y_tr, X_val, y_val, layer_sizes,
                             epochs=epochs, batch_size=batch_size,
                             lr=lr, beta=beta, seed=seed, verbose=False)
    acc_np = hist_np[-1][2]

    print("=" * 60)
    print("PyTorch (Etapa 2)")
    print("=" * 60)
    _, hist_pt, acc_pt = train_torch(X_tr, y_tr, X_val, y_val, layer_sizes,
                                     epochs=epochs, batch_size=batch_size,
                                     lr=lr, beta=beta, seed=seed, verbose=False)

    gap_pp = abs(acc_np - acc_pt) * 100
    print("\n" + "-" * 60)
    print(f"  Acurácia validação NumPy   : {acc_np*100:.2f}%")
    print(f"  Acurácia validação PyTorch : {acc_pt*100:.2f}%")
    print(f"  Diferença (gap)            : {gap_pp:.2f} p.p.")
    print(f"  Critério (<= {GAP_MAXIMO_PP:.0f} p.p.)        : "
          f"{'PASSOU ✓' if gap_pp <= GAP_MAXIMO_PP else 'FALHOU ✗'}")
    print("-" * 60)
    return acc_np, acc_pt, gap_pp


def main():
    from src.utils.data import get_numpy_flat, N_CLASSES

    X_tr, y_tr = get_numpy_flat("train")
    X_val, y_val = get_numpy_flat("val")
    layer_sizes = [X_tr.shape[1], 256, 128, N_CLASSES]
    comparar(X_tr, y_tr, X_val, y_val, layer_sizes)


if __name__ == "__main__":
    main()
