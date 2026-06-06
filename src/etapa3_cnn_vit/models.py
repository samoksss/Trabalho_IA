"""
Etapa 3 — Fábrica de modelos (Seção 4.3).

Modelos disponíveis:
- "custom"        : CNN autoral (treinada do zero)
- "resnet50"      : família ResNet
- "vgg16"         : família VGG
- "efficientnet"  : família EfficientNet (B0)   -> 3 famílias distintas de CNN
- "vit"           : Vision Transformer (vit_b_16)

Modos de transfer learning:
- "feature_extraction": congela o backbone, treina só a cabeça nova.
- "fine_tuning"       : tudo treinável (backbone descongelado).
- (para a "custom", o modo é ignorado: ela treina inteira, do zero.)
"""

import torch.nn as nn
from torchvision import models

from src.etapa3_cnn_vit.cnn_custom import CustomCNN

MODELOS = ["custom", "resnet50", "vgg16", "efficientnet", "vit"]
MODOS = ["feature_extraction", "fine_tuning"]


def _congela(modulo):
    for p in modulo.parameters():
        p.requires_grad = False


def build_model(nome, n_classes=9, modo="fine_tuning", pretrained=True):
    """Constrói o modelo com a cabeça ajustada para n_classes."""
    w = "DEFAULT" if pretrained else None

    if nome == "custom":
        return CustomCNN(n_classes=n_classes)         # sem pré-treino

    if nome == "resnet50":
        m = models.resnet50(weights=w)
        if modo == "feature_extraction":
            _congela(m)
        m.fc = nn.Linear(m.fc.in_features, n_classes)

    elif nome == "vgg16":
        m = models.vgg16(weights=w)
        if modo == "feature_extraction":
            _congela(m)                               # congela tudo; só a cabeça nova treina
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, n_classes)

    elif nome == "efficientnet":
        m = models.efficientnet_b0(weights=w)
        if modo == "feature_extraction":
            _congela(m)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)

    elif nome == "vit":
        m = models.vit_b_16(weights=w)
        if modo == "feature_extraction":
            _congela(m)
        m.heads.head = nn.Linear(m.heads.head.in_features, n_classes)

    else:
        raise ValueError(f"Modelo desconhecido: {nome}. Use um de {MODELOS}.")

    return m


def conta_parametros(model):
    """Retorna (treináveis, total) — útil para a tabela comparativa."""
    treinaveis = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return treinaveis, total
