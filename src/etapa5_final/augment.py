"""
Etapa 5 — Data augmentation avançado (Seção 4.5).

Justificativa (ponto cobrado pelo enunciado): patches histopatológicos NÃO têm
orientação canônica — um tecido é o mesmo tecido girado ou espelhado. Por isso
flips e rotações de 90° são seguros e aumentam os dados sem distorcer a
morfologia. Já transformações geométricas AGRESSIVAS (cisalhamento, zoom forte,
distorções elásticas) podem deformar núcleos e arranjos celulares, criando
artefatos que não existem na realidade clínica — por isso são evitadas.
O color jitter é mantido leve, para simular variação de coloração H&E entre
lâminas sem descaracterizar os tecidos.
"""

from torchvision import transforms

from src.etapa2_pytorch.data_224 import IMAGENET_MEAN, IMAGENET_STD


# Augmentation de treino apropriado para histologia
train_tf_advanced = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomChoice([                      # rotações múltiplas de 90° (preservam a grade de pixels)
        transforms.RandomRotation((0, 0)),
        transforms.RandomRotation((90, 90)),
        transforms.RandomRotation((180, 180)),
        transforms.RandomRotation((270, 270)),
    ]),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # variação H&E leve
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Avaliação: determinística, sem augmentation
eval_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
