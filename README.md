# Do NumPy ao Estado da Arte em Visão Computacional
### Classificação de Tecidos Histopatológicos com PathMNIST

Trabalho Final — Inteligência Artificial · Sistemas de Informação · UniCatólica
Prof. Me. Décio Gonçalves de Aguiar Neto

---

## 1. Equipe

| Nome | Matrícula |
|------|-----------|
| Esteiner Fernandes de Freitas | 2015010349 |
| Gabriel da Silva Braga | 2023020063 |
| Marcos Felipe Arruda Lima | 2023010598 |
| Maria Letícia Pimentel Carlos | 2023010523 |
| Breno Kauê Honório da Silva | 2023010200 |
| Samuel Brito da Silva | 2023020033 |

**Divisão de responsabilidades** _(opcional)_: Etapa 1 (NumPy) — Esteiner;
Etapa 2 (PyTorch e equivalência) — Gabriel; Etapa 3 (CNNs, ViT e transfer
learning) — Marcos; Etapa 4 (explicabilidade / Grad-CAM) — Maria Letícia;
Etapa 5 (otimização final) — Breno; relatório e revisão — toda a equipe.

## 2. O problema

Classificação multiclasse (9 tecidos) de patches histopatológicos de câncer
colorretal corados em H&E, usando o dataset **PathMNIST** (MedMNIST v2).
Resolução oficial deste trabalho: **224×224 RGB**.

Classes: adipose, background, debris, lymphocytes, mucus, smooth muscle,
normal colon mucosa, cancer-associated stroma, colorectal adenocarcinoma
epithelium.

## 3. Principais resultados

| Item | Resultado |
|------|-----------|
| Gradient check (Etapa 1) | diferença relativa **6,13×10⁻¹⁰** (≪ 1e-4) |
| Equivalência NumPy ↔ PyTorch (Etapa 2) | gap **0,00 p.p.** (critério ≤ 2 p.p.) |
| Melhor modelo (validação) | ResNet50 e EfficientNet (fine-tuning) — **99,79%** |
| Melhor custo-benefício | **EfficientNet-B0** (99,79%, ~4M parâmetros) |
| **Acurácia final no teste** (uso único) | **94,09%** |

> A diferença entre validação (~99,8%) e teste (94,09%) reflete o *domain shift*
> do PathMNIST: o conjunto de teste vem de uma fonte clínica independente
> (CRC-VAL-HE-7K) das amostras de treino/validação (NCT-CRC-HE-100K),
> evidenciando generalização real, e não memorização.

## 4. Como reproduzir

```bash
# Ambiente (Etapas 1 e 2 rodam em CPU local)
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt

# Etapa 1 — MLP em NumPy (+ gradient check)
python -m src.etapa1_numpy.train
# Etapa 2 — equivalência NumPy vs PyTorch
python -m src.etapa2_pytorch.equivalence
```

As **Etapas 3 a 5** (224×224, com GPU) foram executadas no **Kaggle**, pelos
notebooks em `notebooks/` (`etapa3_rerun_kaggle.ipynb`, `etapa4_kaggle.ipynb`,
`etapa5_kaggle.ipynb`).

**Seed oficial:** `42` (definida em `src/utils/seeds.py`), chamada no início de
cada experimento.

## 5. Hardware utilizado

- **GPU:** NVIDIA Tesla T4 (16 GB de VRAM) — via Kaggle Notebooks
- **RAM:** ~30 GB (ambiente do Kaggle)
- **Ambiente:** Kaggle (Etapas 3–5, com *mixed precision*); Etapas 1–2 em CPU local
- **Reprodutibilidade:** seeds fixas (numpy, torch, random); dependências em `requirements.txt`

## 6. Estrutura do repositório

```
Trabalho_IA/
├── README.md
├── requirements.txt
├── src/
│   ├── utils/             # seeds, carregamento de dados
│   ├── etapa1_numpy/      # MLP do zero (forward/backprop, SGD+Momentum)
│   ├── etapa2_pytorch/    # mesma MLP em PyTorch + pipeline 224x224
│   ├── etapa3_cnn_vit/    # CNN autoral, 3 CNNs pré-treinadas, ViT
│   ├── etapa4_xai/        # Feature Maps, Grad-CAM
│   └── etapa5_final/      # augmentation, scheduler, regularização, teste final
├── notebooks/             # notebooks de execução no Kaggle
├── results/
│   ├── logs/              # CSVs das comparações
│   └── figures/           # Grad-CAM, feature maps, matriz de confusão
└── relatorio/             # artigo científico + discussão clínica
```

> Os checkpoints `.pt` (>1,7 GB no total) não são versionados (ver `.gitignore`).

## 7. Status das etapas

- [x] Etapa 1 — Rede neural do zero em NumPy (+ gradient checking)
- [x] Etapa 2 — Migração para PyTorch (validação ≤ 2 p.p. vs NumPy)
- [x] Etapa 3 — CNN autoral + 3 CNNs pré-treinadas + ViT
- [x] Etapa 4 — Explicabilidade (Feature Maps, Grad-CAM)
- [x] Etapa 5 — Desafio final (augmentation, scheduler, regularização) + teste

## 8. Uso de IA generativa

> Exigência do enunciado (Seção 9).

A equipe utilizou o **Claude (Anthropic)** como ferramenta de **suporte** ao
longo do projeto: apoio na estruturação do repositório, na implementação e
depuração do código das etapas, e no esclarecimento de conceitos (backpropagation,
transfer learning, Grad-CAM). Todo o código gerado com esse apoio foi **revisado,
testado e compreendido pela equipe**.

| Ferramenta | Etapas | Finalidade |
|------------|--------|------------|
| Claude (Anthropic) | 1 a 5 | Suporte à implementação, depuração e explicação dos conceitos; código revisado pela equipe |

## 9. Referências

- Yang et al. (2023). *MedMNIST v2 — A large-scale lightweight benchmark for
  2D and 3D biomedical image classification.* Scientific Data, 10(1).
  _(referência obrigatória)_
