"""
Etapa 4 — Grad-CAM implementado do zero (Seção 4.4).

Grad-CAM: pondera os mapas de ativação da última camada convolucional pelos
gradientes da classe alvo, destacando as regiões que mais pesaram na decisão.

Passos:
  A = ativações da camada alvo            (B, C, H, W)   [forward hook]
  G = dScore/dA                           (B, C, H, W)   [backward hook]
  w = média espacial de G                 (B, C)         (importância de cada canal)
  cam = ReLU(sum_c w_c * A_c)             (B, H, W)
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.etapa2_pytorch.data_224 import IMAGENET_MEAN, IMAGENET_STD


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.acts = None
        self.grads = None
        self._fh = target_layer.register_forward_hook(self._save_acts)
        self._bh = target_layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, module, inp, out):
        self.acts = out.detach()

    def _save_grads(self, module, grad_in, grad_out):
        self.grads = grad_out[0].detach()

    def __call__(self, x, class_idx=None):
        """x: (B,3,H,W). Retorna (cam [B,h,w] em [0,1], probs [B,classes], classe usada)."""
        self.model.eval()
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1)
        score = logits.gather(1, class_idx.view(-1, 1)).sum()
        self.model.zero_grad()
        score.backward()

        w = self.grads.mean(dim=(2, 3), keepdim=True)          # (B,C,1,1)
        cam = F.relu((w * self.acts).sum(dim=1))               # (B,h,w)
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        probs = logits.softmax(1).detach()
        return cam.cpu().numpy(), probs.cpu().numpy(), class_idx.detach().cpu().numpy()

    def remove(self):
        self._fh.remove()
        self._bh.remove()


# ---------------------------------------------------------------------------
# Utilidades de visualização
# ---------------------------------------------------------------------------
def denormalize(t):
    """Desfaz a normalização ImageNet -> imagem RGB em [0,1], shape (H,W,3)."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (t.detach().cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def overlay_cam(img_rgb, cam, alpha=0.5):
    """Sobrepõe o mapa de calor (jet) na imagem. Retorna RGB em [0,1]."""
    import matplotlib.cm as cm
    h, w = img_rgb.shape[:2]
    cam_img = Image.fromarray((cam * 255).astype("uint8")).resize((w, h), Image.BILINEAR)
    cam_resized = np.asarray(cam_img) / 255.0
    heat = cm.jet(cam_resized)[..., :3]
    return (1 - alpha) * img_rgb + alpha * heat


# ---------------------------------------------------------------------------
# Seleção dos casos exigidos pelo enunciado
# ---------------------------------------------------------------------------
@torch.no_grad()
def gather_predictions(model, loader, device):
    """Coleta (idx, verdadeiro, predito, confianca, prob_verdadeira) na ordem do dataset."""
    model.eval()
    linhas, idx = [], 0
    for xb, yb in loader:
        probs = model(xb.to(device)).softmax(1).cpu().numpy()
        preds = probs.argmax(1)
        for j in range(len(yb)):
            t = int(yb[j])
            linhas.append((idx, t, int(preds[j]), float(probs[j].max()), float(probs[j, t])))
            idx += 1
    return linhas


def select_cases(linhas, n_acertos=5, n_erros=5):
    """Separa: acertos confiantes, erros grosseiros (confiantes mas errados) e 1 acerto 'por sorte'."""
    acertos = [r for r in linhas if r[2] == r[1]]
    erros = [r for r in linhas if r[2] != r[1]]
    acertos_conf = sorted(acertos, key=lambda r: r[3], reverse=True)[:n_acertos]
    erros_gross = sorted(erros, key=lambda r: r[3], reverse=True)[:n_erros]
    # 'por sorte' = acerto com a MENOR confiança (mal passou da segunda classe)
    sorte = sorted(acertos, key=lambda r: r[3])[:1]
    return {"acertos_confiantes": acertos_conf,
            "erros_grosseiros": erros_gross,
            "acerto_por_sorte": sorte}
