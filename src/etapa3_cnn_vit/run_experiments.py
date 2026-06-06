"""
Etapa 3 — Runner de experimentos (Seção 4.3).

Dois experimentos:
1) sweep_optim_lr: cruza 2 otimizadores (SGD, AdamW) x 3 learning rates
   (1e-2, 1e-3, 1e-4) num modelo representativo -> escolhe a melhor config.
2) compare_models: roda CNN autoral + 3 CNNs (feature_extraction e
   fine_tuning) + ViT com a melhor config -> tabela acc/tempo/VRAM/parâmetros.

Os resultados vão para results/logs/*.csv (entram direto no relatório).
"""

import csv
import os

import torch

from src.etapa3_cnn_vit.engine import train_model
from src.etapa3_cnn_vit.models import build_model, conta_parametros

LOG_DIR = "results/logs"
CKPT_DIR = "results/checkpoints"
OTIMIZADORES = ["sgd", "adamw"]
LEARNING_RATES = [1e-2, 1e-3, 1e-4]


def make_optimizer(nome, model, lr):
    """Otimizador apenas sobre os parâmetros treináveis."""
    params = [p for p in model.parameters() if p.requires_grad]
    if nome == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if nome == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    raise ValueError(f"Otimizador desconhecido: {nome}")


def _salva_csv(linhas, campos, caminho):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    with open(caminho, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)
    print(f"[ok] salvo: {caminho}")


def sweep_optim_lr(model_name="resnet50", modo="fine_tuning",
                   train_loader=None, val_loader=None, epochs=5, pretrained=True):
    """Cruza otimizadores x LR. Retorna a melhor config e salva o CSV."""
    resultados = []
    for opt_nome in OTIMIZADORES:
        for lr in LEARNING_RATES:
            print(f"\n>>> {model_name}/{modo} | {opt_nome} | lr={lr}")
            model = build_model(model_name, modo=modo, pretrained=pretrained)
            opt = make_optimizer(opt_nome, model, lr)
            r = train_model(model, train_loader, val_loader, optimizer=opt, epochs=epochs)
            resultados.append({
                "modelo": model_name, "modo": modo, "otimizador": opt_nome,
                "lr": lr, "val_acc": round(r["best_val_acc"], 4),
                "tempo_s": r["tempo_s"], "vram_mb": r["vram_mb"],
            })
    _salva_csv(resultados, list(resultados[0].keys()),
               os.path.join(LOG_DIR, "etapa3_sweep_optim_lr.csv"))
    melhor = max(resultados, key=lambda d: d["val_acc"])
    print(f"\n[MELHOR] {melhor['otimizador']} lr={melhor['lr']} "
          f"-> val_acc {melhor['val_acc']}")
    return melhor


def compare_models(train_loader=None, val_loader=None, optimizer="adamw",
                   lr=1e-3, epochs=8, pretrained=True):
    """Tabela comparativa de todos os modelos. Salva CSV."""
    plano = [
        ("custom", "fine_tuning"),
        ("resnet50", "feature_extraction"), ("resnet50", "fine_tuning"),
        ("vgg16", "feature_extraction"), ("vgg16", "fine_tuning"),
        ("efficientnet", "feature_extraction"), ("efficientnet", "fine_tuning"),
        ("vit", "feature_extraction"), ("vit", "fine_tuning"),
    ]
    os.makedirs(CKPT_DIR, exist_ok=True)
    resultados = []
    for nome, modo in plano:
        print(f"\n=== {nome} / {modo} ===")
        model = build_model(nome, modo=modo, pretrained=pretrained)
        opt = make_optimizer(optimizer, model, lr)
        ckpt = os.path.join(CKPT_DIR, f"etapa3_{nome}_{modo}.pt")
        r = train_model(model, train_loader, val_loader, optimizer=opt,
                        epochs=epochs, ckpt_path=ckpt)
        treinaveis, total = conta_parametros(model)
        resultados.append({
            "modelo": nome, "modo": modo,
            "val_acc": round(r["best_val_acc"], 4),
            "tempo_s": r["tempo_s"], "vram_mb": r["vram_mb"],
            "params_treinaveis": treinaveis, "params_total": total,
        })
    _salva_csv(resultados, list(resultados[0].keys()),
               os.path.join(LOG_DIR, "etapa3_comparacao_modelos.csv"))
    return resultados
