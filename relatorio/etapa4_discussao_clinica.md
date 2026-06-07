# Etapa 4 — Discussão da Explicabilidade (rascunho para o relatório)

> Adapte o texto às suas imagens. Os números (99,82%, confianças) já são os reais do seu experimento.

## Feature Maps da primeira camada convolucional

As ativações da camada `conv1` do ResNet50 evidenciam a especialização precoce
dos filtros, mesmo na profundidade mais rasa da rede. Observou-se que parte dos
filtros responde fortemente a **texturas direcionais e fibrilares** (compatíveis
com músculo liso e estroma associado ao câncer), enquanto outros se especializam
em **pontos e grânulos finos** — consistentes com a detecção de núcleos celulares —
e ainda outros capturam **intensidade e cor globais** (relevantes para a coloração
H&E). Alguns filtros permaneceram pouco ativos para a imagem analisada, indicando
seletividade: eles detectam padrões ausentes naquele patch específico. Esse
comportamento confirma que, já nas primeiras camadas, a rede decompõe a imagem
histológica em bordas, cor e textura, blocos elementares que as camadas profundas
combinam em conceitos morfológicos.

## Grad-CAM — análise dos casos

A análise foi conduzida na última camada convolucional (`layer4`), com o modelo
ResNet50 (fine-tuning) que atingiu **99,82% de acurácia de validação**.

**Acertos com alta confiança.** Nos casos corretos com confiança ~1,00, os mapas
de calor concentraram-se sobre as **estruturas teciduais relevantes** — regiões
celulares densas e arranjos fibrilares — e não sobre espaços vazios ou artefatos.
Isso sugere que a decisão do modelo é guiada por evidência histológica legítima,
e não por atalhos espúrios, o que é desejável em um contexto diagnóstico.

**Erros grosseiros (confiantes, porém incorretos).** Os erros mais informativos
revelam os limites do modelo. No caso *background* classificado como *adipose*
com confiança 1,00, o Grad-CAM delimitou uma ampla região clara e de baixa
textura; o erro é plausível, pois o tecido adiposo também se apresenta claro e
pouco texturizado nas lâminas H&E, levando à confusão por **similaridade visual**.
Nos demais erros, o calor recaiu sobre regiões pequenas e dispersas, indicando
que o modelo se ancorou em **detalhes locais não representativos** da classe
verdadeira. Esse padrão expõe um risco clínico: alta confiança não garante
fundamentação correta.

**Acerto "por sorte".** O patch *normal colon mucosa* foi classificado
corretamente, mas com confiança de apenas 0,50, evidenciando quase empate com uma
segunda classe. Embora os focos de atenção tenham caído sobre aglomerados
celulares, a baixa confiança sugere um patch **morfologicamente ambíguo**,
possivelmente na fronteira entre dois tecidos. O acerto, nesse caso, é frágil e
não deveria ser tratado como uma decisão confiável.

## Conclusão da explicabilidade

O Grad-CAM mostrou que o ResNet50, apesar da acurácia elevada, **não é uma
caixa-preta arbitrária**: nos acertos confiantes, sua atenção é histologicamente
coerente. Por outro lado, os erros confiantes e o acerto por sorte demonstram que
a confiança do softmax, isoladamente, é um indicador insuficiente de
confiabilidade. Em um cenário diagnóstico real, isso reforça a necessidade de
ferramentas de explicabilidade como apoio à decisão humana, e não como
substituição dela.
