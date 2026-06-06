"""
Carregamento econômico de RAM do PathMNIST 224x224 (para PCs com 16 GB).

O split de treino do PathMNIST 224 ocupa ~13,5 GB se carregado inteiro na RAM
(89.996 x 224 x 224 x 3 bytes). Em 16 GB isso estoura ou força swap lento.

Solução em 2 partes:
1) convert(): lê o .npz EM BLOCOS (direto de dentro do zip, sem carregar o
   array inteiro) e grava .npy NÃO comprimidos, que podem ser memory-mappeados.
2) MmapPathMNIST: lê cada imagem sob demanda do .npy mmapeado -> só o batch
   fica na RAM.

Custo em disco da conversão: ~16 GB (train 13,5 + val 1,5 + test 1,1).

Uso:
    # 1) Conversão única (precisa do pathmnist_224.npz já baixado):
    python -m src.utils.data_mmap --npz data/pathmnist_224.npz --out data/pathmnist224

    # 2) Loaders econômicos:
    from src.utils.data_mmap import make_loaders_local
    tr, vl, te = make_loaders_local("data/pathmnist224", batch_size=16)
"""

import argparse
import os
import zipfile

import numpy as np
from numpy.lib import format as npformat
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SPLITS = ["train", "val", "test"]


# ---------------------------------------------------------------------------
# Conversão em streaming: .npz -> .npy mmapeáveis
# ---------------------------------------------------------------------------
def _read_header(f):
    version = npformat.read_magic(f)
    if version == (1, 0):
        return npformat.read_array_header_1_0(f)
    if version == (2, 0):
        return npformat.read_array_header_2_0(f)
    raise ValueError(f"Versão .npy não suportada: {version}")


def _stream_member_to_npy(zf, member, out_path, block=2000):
    """Copia um .npy de dentro do zip para um .npy de saída, em blocos."""
    with zf.open(member) as f:
        shape, fortran, dtype = _read_header(f)
        assert not fortran, "array Fortran-order não suportado"
        out = npformat.open_memmap(out_path, mode="w+", dtype=dtype, shape=shape)
        n = shape[0]
        bytes_por_linha = int(np.prod(shape[1:])) * dtype.itemsize
        i = 0
        while i < n:
            k = min(block, n - i)
            buf = f.read(k * bytes_por_linha)
            arr = np.frombuffer(buf, dtype=dtype).reshape((k,) + shape[1:])
            out[i:i + k] = arr
            i += k
        out.flush()
        del out
    return shape


def convert(npz_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(npz_path) as zf:
        nomes = set(zf.namelist())
        for split in SPLITS:
            img_m, lbl_m = f"{split}_images.npy", f"{split}_labels.npy"
            if img_m not in nomes:
                raise KeyError(f"'{img_m}' não está no npz. Conteúdo: {sorted(nomes)}")
            shape = _stream_member_to_npy(zf, img_m, os.path.join(out_dir, img_m))
            with zf.open(lbl_m) as f:                 # labels são pequenos
                lbl = npformat.read_array(f)
            np.save(os.path.join(out_dir, lbl_m), lbl)
            print(f"  {split}: imagens {shape} | labels {lbl.shape}")
    print(f"[ok] conversão concluída em {out_dir}")


# ---------------------------------------------------------------------------
# Dataset que lê sob demanda (mmap)
# ---------------------------------------------------------------------------
class MmapPathMNIST(Dataset):
    def __init__(self, root, split, transform=None):
        self.imgs = np.load(os.path.join(root, f"{split}_images.npy"), mmap_mode="r")
        self.labels = np.load(os.path.join(root, f"{split}_labels.npy")).reshape(-1)
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = Image.fromarray(np.asarray(self.imgs[i]))   # copia só 1 imagem
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[i])


def make_loaders_local(root, batch_size=16, num_workers=4, augment=True):
    """DataLoaders econômicos de RAM. Use após converter o npz."""
    import torch
    from src.etapa2_pytorch.data_224 import train_tf, eval_tf

    pin = torch.cuda.is_available()
    tr_tf = train_tf if augment else eval_tf
    train_ds = MmapPathMNIST(root, "train", transform=tr_tf)
    val_ds = MmapPathMNIST(root, "val", transform=eval_tf)
    test_ds = MmapPathMNIST(root, "test", transform=eval_tf)

    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin)
    return (DataLoader(train_ds, shuffle=True, **common),
            DataLoader(val_ds, shuffle=False, **common),
            DataLoader(test_ds, shuffle=False, **common))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="caminho do pathmnist_224.npz")
    ap.add_argument("--out", required=True, help="pasta de saída dos .npy")
    a = ap.parse_args()
    convert(a.npz, a.out)
