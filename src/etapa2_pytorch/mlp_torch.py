"""
Etapa 2 — A mesma MLP da Etapa 1, agora em PyTorch.

Equivalência da regra de atualização:
  NumPy (nosso):   v = beta*v + lr*grad ;  W -= v
  PyTorch SGD:     buf = beta*buf + grad ;  W -= lr*buf
Definindo U = lr*buf, sai U = beta*U + lr*grad  ->  mesma recorrência.
Logo, com mesmo lr/beta/seed e dados iguais, ambas convergem juntas.

Arquitetura: Linear -> ReLU -> ... -> Linear (logits).
A Softmax NÃO entra no forward: nn.CrossEntropyLoss já aplica
log-softmax + NLL internamente (equivale ao nosso softmax_cross_entropy).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLPTorch(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        layers = []
        for i, (fin, fout) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layers.append(nn.Linear(fin, fout))
            if i < len(layer_sizes) - 2:        # ReLU em todas menos a última
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)       # saída = logits (sem softmax)

    def forward(self, x):
        return self.net(x)


def train_torch(X_tr, y_tr, X_val, y_val, layer_sizes,
                epochs=20, batch_size=128, lr=0.1, beta=0.9, seed=42, verbose=True):
    """Treina a MLP em PyTorch. Retorna (modelo, historico, val_acc_final)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr = torch.tensor(X_tr, dtype=torch.float32)
    ytr = torch.tensor(y_tr, dtype=torch.long)
    Xval = torch.tensor(X_val, dtype=torch.float32).to(device)
    yval = torch.tensor(y_val, dtype=torch.long).to(device)

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(seed))

    model = MLPTorch(layer_sizes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=beta)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_acc = (model(Xval).argmax(1) == yval).float().mean().item()
        history.append((epoch, float(np.mean(losses)), val_acc))
        if verbose:
            print(f"epoca {epoch:2d} | loss {np.mean(losses):.4f} | val_acc {val_acc:.4f}")
    return model, history, history[-1][2]
