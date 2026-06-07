"""
Etapa 4 — Feature Maps e plotagem dos casos Grad-CAM (Seção 4.4).

- Feature Maps: ativações das primeiras camadas convolucionais (o que cada
  filtro "enxerga" na imagem de entrada).
- Painel Grad-CAM: grade de imagens com o mapa de calor sobreposto, anotando
  classe verdadeira, predita e confiança.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.etapa4_xai.gradcam import denormalize, overlay_cam
from src.utils.data import LABELS


def feature_maps(model, image_tensor, layer, max_maps=16):
    """Captura as ativações de `layer` para uma imagem. Retorna (max_maps, h, w)."""
    cap = {}
    h = layer.register_forward_hook(lambda m, i, o: cap.__setitem__("a", o.detach()))
    model.eval()
    with torch.no_grad():
        model(image_tensor.unsqueeze(0).to(next(model.parameters()).device))
    h.remove()
    return cap["a"][0, :max_maps].cpu().numpy()


def plot_feature_maps(fmaps, path, cols=4):
    n = len(fmaps)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for i, ax in enumerate(np.array(axes).flat):
        if i < n:
            ax.imshow(fmaps[i], cmap="viridis")
            ax.set_title(f"filtro {i}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[ok] feature maps -> {path}")


def plot_gradcam_grid(cam_obj, dataset, casos, device, path, titulo=""):
    """Plota uma grade Grad-CAM para uma lista de casos (idx, verd, pred, conf, prob_v)."""
    n = len(casos)
    fig, axes = plt.subplots(1, n, figsize=(n * 2.6, 3.0))
    if n == 1:
        axes = [axes]
    for ax, (idx, verd, pred, conf, _) in zip(axes, casos):
        x, _ = dataset[idx]
        cam, _, _ = cam_obj(x.unsqueeze(0).to(device), class_idx=torch.tensor([pred]).to(device))
        vis = overlay_cam(denormalize(x), cam[0])
        ax.imshow(vis)
        cor = "green" if pred == verd else "red"
        ax.set_title(f"V: {LABELS[str(verd)][:10]}\nP: {LABELS[str(pred)][:10]}\nconf {conf:.2f}",
                     fontsize=7, color=cor)
        ax.axis("off")
    fig.suptitle(titulo, fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[ok] grid Grad-CAM -> {path}")
