"""
Etapa 3 — CNN autoral (Seção 4.3).

Requisitos: mínimo 3 blocos convolucionais, com pooling e dropout.
Arquitetura: 3 blocos [Conv-BN-ReLU x2 -> MaxPool] com canais 32->64->128,
depois global average pooling, dropout e camada totalmente conectada.

Entrada: (B, 3, 224, 224). Saída: logits (B, n_classes).
"""

import torch.nn as nn


def _bloco(c_in, c_out):
    """Bloco convolucional: 2x (Conv 3x3 -> BN -> ReLU) seguido de MaxPool."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),                 # reduz resolução pela metade
    )


class CustomCNN(nn.Module):
    def __init__(self, n_classes=9, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            _bloco(3, 32),               # 224 -> 112
            _bloco(32, 64),              # 112 -> 56
            _bloco(64, 128),             # 56 -> 28
        )
        self.pool = nn.AdaptiveAvgPool2d(1)   # (B,128,28,28) -> (B,128,1,1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
