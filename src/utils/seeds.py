"""
Utilitário de reprodutibilidade.

O enunciado exige (Seção 6.1): "Fixar seeds em numpy, torch e random;
documentar a seed usada". Este módulo centraliza isso para que TODAS as
etapas usem exatamente a mesma seed e fiquem reprodutíveis.

Uso:
    from src.utils.seeds import set_seed
    set_seed(42)
"""

import os
import random

import numpy as np

# Seed oficial do projeto. Documente este valor no README.
SEED = 42


def set_seed(seed: int = SEED, deterministic_torch: bool = True) -> None:
    """Fixa a seed em random, numpy e (se instalado) torch.

    Args:
        seed: valor da semente.
        deterministic_torch: se True, força operações determinísticas no
            PyTorch (mais lento, porém reprodutível). Use True para os
            experimentos que entram no relatório.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # torch é opcional: na Etapa 1 (NumPy puro) ele nem é importado.
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    print(f"[seeds] Seed fixada em {seed}.")


if __name__ == "__main__":
    set_seed()
    # Sanity check: dois sorteios após reset devem ser idênticos.
    set_seed(123)
    a = np.random.rand(3)
    set_seed(123)
    b = np.random.rand(3)
    assert np.allclose(a, b), "Seed não está reprodutível!"
    print("[seeds] OK — reprodutibilidade verificada.")
