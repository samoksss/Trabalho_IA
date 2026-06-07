"""
Etapa 5 — Treino final + avaliação no teste (Seção 4.5).

Combina:
- Otimizador AdamW com weight decay (regularização L2).
- Scheduler Cosine (decai o learning rate suavemente).
- Label smoothing na perda (evita excesso de confiança).
- Early stopping (para quando a validação para de melhorar).
- Gradient clipping (estabilidade).

REGRA METODOLÓGICA: o conjunto de TESTE só é tocado em evaluate_test(), uma
única vez, sobre o melhor modelo final.
"""

import time

import numpy as np
import torch

from src.etapa3_cnn_vit.engine import evaluate


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_final(model, train_loader, val_loader, *, epochs=15, lr=1e-3,
                weight_decay=1e-4, label_smoothing=0.1, patience=4,
                grad_clip=1.0, device=None, ckpt_path=None, use_amp=None, verbose=True):
    device = device or _device()
    if use_amp is None:
        use_amp = (device == "cuda")
    model = model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    history, best_acc, sem_melhora = [], 0.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()

        val_acc = evaluate(model, val_loader, device)
        history.append((epoch, float(np.mean(losses)), val_acc))
        if verbose:
            lr_atual = scheduler.get_last_lr()[0]
            print(f"  epoca {epoch:2d} | loss {np.mean(losses):.4f} | val_acc {val_acc:.4f} | lr {lr_atual:.2e}")

        if val_acc > best_acc:
            best_acc, sem_melhora = val_acc, 0
            if ckpt_path:
                torch.save(model.state_dict(), ckpt_path)
        else:
            sem_melhora += 1
            if sem_melhora >= patience:
                print(f"  [early stopping] sem melhora há {patience} épocas. Parando.")
                break

    return {"history": history, "best_val_acc": best_acc,
            "tempo_s": round(time.time() - t0, 1),
            "vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1) if device == "cuda" else 0.0}


@torch.no_grad()
def evaluate_test(model, test_loader, device=None):
    """USO ÚNICO: avalia no conjunto de teste. Retorna (y_true, y_pred, acuracia)."""
    device = device or _device()
    model = model.to(device).eval()
    trues, preds = [], []
    for xb, yb in test_loader:
        p = model(xb.to(device)).argmax(1).cpu().numpy()
        preds.extend(p.tolist())
        trues.extend(yb.numpy().tolist())
    trues, preds = np.array(trues), np.array(preds)
    return trues, preds, float((trues == preds).mean())


def plot_confusion(y_true, y_pred, path, class_names=None):
    """Matriz de confusão 9x9 (exigida no relatório)."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False, cmap="Blues")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[ok] matriz de confusão -> {path}")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
