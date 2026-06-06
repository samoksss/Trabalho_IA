# Do NumPy ao Estado da Arte em Visão Computacional
### Classificação de Tecidos Histopatológicos com PathMNIST

Trabalho Final — Inteligência Artificial · Sistemas de Informação · UniCatólica
Prof. Me. Décio Gonçalves de Aguiar Neto

---

## 1. Equipe

| Nome | Matrícula | GitHub | Responsabilidades principais |
|------|-----------|--------|------------------------------|
| _preencher_ | | | |
| _preencher_ | | | |
| _preencher_ | | | |

> ⚠️ Na defesa oral, qualquer integrante pode ser questionado sobre **qualquer
> linha de código**. Todos devem entender o projeto inteiro.

## 2. O problema

Classificação multiclasse (9 tecidos) de patches histopatológicos de câncer
colorretal corados em H&E, usando o dataset **PathMNIST** (MedMNIST v2).
Resolução oficial deste trabalho: **224×224 RGB**.

Classes: adipose, background, debris, lymphocytes, mucus, smooth muscle,
normal colon mucosa, cancer-associated stroma, colorectal adenocarcinoma
epithelium.

## 3. Como reproduzir

```bash
# 1. Ambiente
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Baixar o dataset (automático na primeira execução)
python -m src.utils.data        # baixa PathMNIST 28x28 e verifica os splits

# 3. Rodar as etapas (exemplos — preencher conforme avançar)
python -m src.etapa1_numpy.train
```

**Seed oficial:** `42` (definida em `src/utils/seeds.py`). Sempre chamada no
início de cada experimento.

## 4. Hardware utilizado

> Exigência do enunciado (6.1). Preencher com a máquina onde os modelos foram treinados.

- **CPU:** _preencher_
- **GPU:** _preencher (modelo, VRAM)_
- **RAM:** _preencher_
- **Ambiente:** _local / Google Colab / Kaggle_

## 5. Estrutura do repositório

```
trabalho-final-ia/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                  # dataset (gitignored, baixado via medmnist)
├── src/
│   ├── utils/             # seeds.py, data.py (carregamento + reprodutibilidade)
│   ├── etapa1_numpy/      # MLP do zero (forward/backprop, SGD+Momentum)
│   ├── etapa2_pytorch/    # mesma MLP em PyTorch + DataLoaders
│   ├── etapa3_cnn_vit/    # CNN autoral, 3 CNNs pré-treinadas, ViT
│   ├── etapa4_xai/        # Feature Maps, Grad-CAM
│   └── etapa5_final/      # data augmentation, schedulers, tuning
├── notebooks/             # exploração / experimentos
├── results/
│   ├── logs/              # logs de treino (CSV / TensorBoard)
│   ├── figures/           # gráficos do relatório
│   └── checkpoints/       # melhor modelo
└── relatorio/             # artigo científico (PDF, máx. 12 páginas)
```

## 6. Status das etapas

- [x] Etapa 1 — Rede neural do zero em NumPy (+ gradient checking)
- [ ] Etapa 2 — Migração para PyTorch (validação ≤ 2 p.p. vs NumPy)
- [ ] Etapa 3 — CNN autoral + 3 CNNs pré-treinadas + ViT
- [ ] Etapa 4 — Explicabilidade (Feature Maps, Grad-CAM)
- [ ] Etapa 5 — Desafio final (augmentation, schedulers, tuning)

## 7. Uso de IA generativa

> Exigência do enunciado (Seção 9). Documentar cada uso relevante de LLMs.

| Ferramenta | Etapa | Finalidade |
|------------|-------|------------|
| _ex.: Claude_ | Setup | Estruturação inicial do repositório e utilitários |
| Claude | Etapa 1 | Implementação da MLP em NumPy (forward/backprop, SGD+Momentum, gradient check) — código revisado e entendido pela equipe |
| | | |

## 8. Referências

- Yang et al. (2023). *MedMNIST v2 — A large-scale lightweight benchmark for
  2D and 3D biomedical image classification.* Scientific Data, 10(1).
  _(referência obrigatória)_
