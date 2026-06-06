"""
Carregamento do dataset PathMNIST (coleção MedMNIST v2).

Regra metodológica CRÍTICA do enunciado (Seção 4.5):
- A divisão oficial treino/validação/teste DEVE ser respeitada.
- O conjunto de TESTE só pode ser usado UMA ÚNICA VEZ, no final, sobre
  o melhor modelo. Por isso `get_test_*` fica separado e deve ser chamado
  apenas na avaliação final.

Estratégia de resolução:
- size=28  -> usado na ETAPA 1 (MLP em NumPy). Achatado em vetores de 784*3.
- size=224 -> usado nas ETAPAS 2+ (PyTorch, CNNs, ViTs).

Na primeira execução o medmnist baixa os arquivos para ~/.medmnist (ou para
`root`, se passado). O download exige internet.
"""

from medmnist import INFO, PathMNIST

FLAG = "pathmnist"
INFO_PATH = INFO[FLAG]

N_CLASSES = len(INFO_PATH["label"])          # 9
N_CHANNELS = INFO_PATH["n_channels"]         # 3 (RGB)
LABELS = INFO_PATH["label"]                  # dict id -> nome do tecido


def _load_numpy(split: str, size: int):
    """Retorna (imgs, labels) como arrays NumPy para um split.

    imgs:   uint8, shape (N, size, size, 3)
    labels: int,   shape (N, 1)  -> achatamos para (N,)
    """
    ds = PathMNIST(split=split, size=size, download=True)
    imgs = ds.imgs           # (N, size, size, 3) uint8
    labels = ds.labels.reshape(-1)  # (N,)
    return imgs, labels


# ----------------------------------------------------------------------
# ETAPA 1 — versão achatada 28x28 para a MLP em NumPy
# ----------------------------------------------------------------------
def get_numpy_flat(split: str = "train", normalize: bool = True):
    """Dados achatados para a MLP da Etapa 1.

    Returns:
        X: float32 (N, 28*28*3) = (N, 2352), em [0,1] se normalize=True
        y: int     (N,)
    """
    imgs, y = _load_numpy(split, size=28)
    X = imgs.reshape(imgs.shape[0], -1).astype("float32")
    if normalize:
        X /= 255.0
    return X, y


# ----------------------------------------------------------------------
# ETAPAS 2+ — datasets PyTorch 224x224 (transforms aplicados fora daqui)
# ----------------------------------------------------------------------
def get_torch_dataset(split: str = "train", size: int = 224, transform=None,
                      root=None):
    """Retorna um objeto PathMNIST (torch.utils.data.Dataset).

    root: pasta onde salvar/ler o dataset. No Colab, aponte para o Drive
    (ex.: '/content/drive/MyDrive/pathmnist') para não rebaixar a cada sessão.
    """
    as_rgb = INFO_PATH["n_channels"] == 3
    kwargs = dict(split=split, size=size, transform=transform,
                  download=True, as_rgb=as_rgb)
    if root is not None:
        kwargs["root"] = root
    return PathMNIST(**kwargs)


if __name__ == "__main__":
    # Sanity check rápido (precisa de internet na 1a vez).
    Xtr, ytr = get_numpy_flat("train")
    print(f"[data] Etapa 1 | X_train: {Xtr.shape}  y_train: {ytr.shape}")
    print(f"[data] classes: {N_CLASSES} | canais: {N_CHANNELS}")
    print(f"[data] range X: [{Xtr.min():.3f}, {Xtr.max():.3f}]")
    print(f"[data] rótulos: {LABELS}")
