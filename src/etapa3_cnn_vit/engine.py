"""
Etapa 3 — Engine genérica de treino/avaliação.

Funciona para QUALQUER modelo da fábrica (CNN autoral, pré-treinadas, ViT).
Registra o que a tabela comparativa do enunciado exige:
- acurácia de validação
- tempo de treino (segundos)
- pico de VRAM (MB), via torch.cuda.max_memory_allocated
"""

import time

import numpy as np
import torch


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate(model, loader, device=None):
    """Acurácia média no loader."""
    device = device or _device()
    model.eval()
    certos, total = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(1)
        certos += (pred == yb).sum().item()
        total += yb.numel()
    return certos / max(total, 1)


def train_model(model, train_loader, val_loader, *, optimizer,
                epochs=10, device=None, ckpt_path=None, use_amp=None, verbose=True):
    """Treina o modelo. Retorna dict com histórico, melhor acc, tempo e VRAM.

    use_amp: mixed precision (float16). Reduz ~metade da VRAM — essencial em
    GPUs de 4 GB. Por padrão, liga sozinho quando há GPU.
    """
    device = device or _device()
    if use_amp is None:
        use_amp = (device == "cuda")
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    history, best_acc = [], 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())

        val_acc = evaluate(model, val_loader, device)
        history.append((epoch, float(np.mean(losses)), val_acc))
        if val_acc > best_acc:
            best_acc = val_acc
            if ckpt_path:
                torch.save(model.state_dict(), ckpt_path)
        if verbose:
            print(f"  epoca {epoch:2d} | loss {np.mean(losses):.4f} | val_acc {val_acc:.4f}")

    tempo_s = time.time() - t0
    vram_mb = (torch.cuda.max_memory_allocated() / 1024**2) if device == "cuda" else 0.0
    return {
        "history": history,
        "best_val_acc": best_acc,
        "tempo_s": round(tempo_s, 1),
        "vram_mb": round(vram_mb, 1),
    }
