"""
Etapa 2 — Pipeline de dados 224x224 com torchvision.transforms (Seção 4.2).

- Conversão de canais (RGB garantido via as_rgb no medmnist) + ToTensor.
- Normalização com estatísticas da ImageNet (necessária para usar backbones
  pré-treinados nas Etapas 3+ sem distribuir mal as entradas).

Decisões de DataLoader (justificadas):
- batch_size: 64 é um bom padrão para imagens 224x224 em GPUs de ~12-16GB
  (Colab/Kaggle). Reduza para 32 se estourar a VRAM; aumente se sobrar.
- num_workers: 2 no Colab (CPUs limitadas). Localmente, ~nº de núcleos.
- shuffle: True só no treino (quebra correlação entre amostras vizinhas);
  val/test SEM shuffle, para avaliação determinística.
- pin_memory: True quando há GPU (acelera a transferência CPU->GPU).
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.utils.data import get_torch_dataset

# Estatísticas da ImageNet (RGB)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Transform de avaliação (val/test): determinístico, sem aumento de dados
eval_tf = transforms.Compose([
    transforms.ToTensor(),                       # PIL -> tensor [0,1], (C,H,W)
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Transform de treino: aqui mínimo (flip). Aumento agressivo fica na Etapa 5,
# com cuidado — transformações geométricas fortes distorcem histologia.
train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def _flatten_label(t):
    """PathMNIST devolve rótulo shape (1,); achatamos para escalar long."""
    return int(t[0])


def make_loaders(size=224, batch_size=64, num_workers=2, augment=True, root=None,
                 train_transform=None):
    """Cria os DataLoaders de treino, validação e teste.

    train_transform: se fornecido, substitui o transform de treino padrão
    (usado na Etapa 5 para aplicar o augmentation avançado).
    """
    pin = torch.cuda.is_available()
    tr_tf = train_transform if train_transform is not None else (train_tf if augment else eval_tf)

    train_ds = get_torch_dataset("train", size=size, transform=tr_tf, root=root)
    val_ds = get_torch_dataset("val", size=size, transform=eval_tf, root=root)
    test_ds = get_torch_dataset("test", size=size, transform=eval_tf, root=root)
    for ds in (train_ds, val_ds, test_ds):
        ds.target_transform = _flatten_label

    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin)
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)   # USAR 1x só, no fim
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    tl, vl, _ = make_loaders(batch_size=32)
    xb, yb = next(iter(tl))
    print(f"[224] batch X: {tuple(xb.shape)} | y: {tuple(yb.shape)}")
    print(f"[224] X dtype {xb.dtype} | range [{xb.min():.2f}, {xb.max():.2f}]")
