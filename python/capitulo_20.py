# -*- coding: utf-8 -*-
# Capítulo 20: exportação automática das células do notebook executável.
# Executar na raiz do projeto: python python/capitulo_20.py
# Células que dependem de contexto de notebook, serviços ou bibliotecas opcionais podem exigir ajustes.

# %% [markdown]
# # Capítulo 20 — Transfer learning
#
# **Execução demonstrativa com dados sintéticos**. O código do bloco 20.1 é preservado integralmente. O arquivo `dados_tuberculose_v2.csv` fornecido é sintético e NÃO contém registros SINAN-TB. Execute no diretório deste notebook.

# %% [markdown]
# ### Bloco de Código 20.1 — Exemplo de aprendizado por transferência com ­CatBoost em dados de tuberculose.
#
# *Página 426 do PDF · Revisão: confirmado*

# %% [markdown]
# ### Preparação dos dados
#
# O CSV sintético foi gerado previamente e incluído no pacote. Mantém os nomes das colunas e os dois domínios geográficos/temporais exigidos pelo exemplo. O conjunto de teste de Sergipe é reservado para avaliação.

# %% [code] célula 4
from pathlib import Path
assert Path("dados_tuberculose_v2.csv").exists(), "CSV sintético ausente"
print("Base sintética disponível")

# %% [markdown]
# ### Código revisado original — bloco 20.1
#
# Abaixo está o código original, **sem modificações**.

# %% [code] célula 6
# Tuberculose pulmonar
# Desfecho: cura (0) vs óbito por tuberculose (1)
# Transferência de aprendizado:
## Pernambuco (domínio fonte) -> Sergipe (domínio destino)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from catboost import CatBoostClassifier

# 1. Leitura e filtragem da base
df = pd.read_csv("dados_tuberculose_v2.csv", low_memory=False)
# ajuste o caminho conforme seu ambiente

# Mantém apenas tuberculose pulmonar e desfechos de interesse
df = df[(df["FORMA"] == 1) & (df["SITUA_ENCE"].isin([1, 3]))].copy()

# Cria variável binária: 0 = cura | 1 = óbito por TB
df["obito_tb"] = (df["SITUA_ENCE"] == 3).astype(int)

# Exclui anos de pandemia
df = df[~df["NU_ANO"].isin([2020, 2021])].copy()

# Restringe a adultos
df["IDADE_ANOS"] = pd.to_numeric(df["IDADE_ANOS"], errors="coerce")
df = df[df["IDADE_ANOS"] >= 18].copy()

# 2. Seleção das variáveis preditoras

vars_modelo = [
    "IDADE_ANOS", "CS_SEXO", "CS_RACA", "CS_ESCOL_N",
    "POP_LIBER", "POP_RUA", "POP_SAUDE", "POP_IMIG",
    "BENEF_GOV", "HIV", "AGRAVAIDS", "AGRAVDIABE",
    "AGRAVDOENC", "AGRAVOUTRA", "AGRAVTABAC",
    "AGRAVALCOO", "AGRAVDROGA",
    "RAIOX_TORA", "BACILOSC_E", "CULTURA_ES",
    "TEST_MOLEC"
]

vars_cat = [v for v in vars_modelo if v != "IDADE_ANOS"]

# Índices das colunas categóricas para o CatBoost
cat_features = [i for i, v in enumerate(vars_modelo) if v != "IDADE_ANOS"]

# 3. Definição da fonte e do destino

# Pernambuco (UF=26) | 2016-2019
fonte = df[(df["SG_UF"] == 26) & (df["NU_ANO"] <= 2019)].copy()

# Sergipe (UF=28) | 2022-2023
destino = df[(df["SG_UF"] == 28) & (df["NU_ANO"] >= 2022)].copy()

# Converte categóricas para string; campos em branco recebem 'Ignorado'
for c in vars_cat:
    fonte[c] = fonte[c].where(fonte[c].notna(), other="Ignorado").astype(str)
    destino[c] = destino[c].where(destino[c].notna(), other="Ignorado").astype(str)

X_fonte = fonte[vars_modelo]
y_fonte = fonte["obito_tb"]

X_destino = destino[vars_modelo]
y_destino = destino["obito_tb"]

# 4. Divisão em treino e teste

Xtreino_fonte, Xteste_fonte, ytreino_fonte, yteste_fonte = train_test_split(
    X_fonte,
    y_fonte,
    test_size=0.30,
    random_state=42,
    stratify=y_fonte
)

Xtreino_destino, Xteste_destino, ytreino_destino, yteste_destino = train_test_split(
    X_destino,
    y_destino,
    test_size=0.30,
    random_state=42,
    stratify=y_destino
)

# 5. Modelo da fonte (Pernambuco)

modelo_fonte = CatBoostClassifier(
    iterations=120,
    depth=5,
    learning_rate=0.08,
    loss_function="Logloss",
    eval_metric="AUC",
    auto_class_weights="Balanced",
    verbose=0,
    random_seed=42
)

modelo_fonte.fit(
    Xtreino_fonte,
    ytreino_fonte,
    cat_features=cat_features
)

pred_fonte = modelo_fonte.predict_proba(Xteste_fonte)[:, 1]

# 6. Modelo com transferência para Sergipe

# Usa o modelo treinado em Pernambuco como ponto de partida (init_model)
# Total de árvores: 120 (fonte) + 60 (fine-tuning) = 180

modelo_transferencia = CatBoostClassifier(
    iterations=60,
    depth=5,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    auto_class_weights="Balanced",
    verbose=0,
    random_seed=42
)

modelo_transferencia.fit(
    Xtreino_destino,
    ytreino_destino,
    cat_features=cat_features,
    init_model=modelo_fonte
)

pred_transferencia = modelo_transferencia.predict_proba(
    Xteste_destino
)[:, 1]

# 7. Modelo treinado apenas com dados de Sergipe
# 180 iterações = 120 fonte + 60 fine-tuning
modelo_sergipe = CatBoostClassifier(
    iterations=180,
    depth=5,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    auto_class_weights="Balanced",
    verbose=0,
    random_seed=42
)

modelo_sergipe.fit(
    Xtreino_destino,
    ytreino_destino,
    cat_features=cat_features
)

pred_sergipe = modelo_sergipe.predict_proba(Xteste_destino)[:, 1]

# 8. Resultados

auroc_fonte = roc_auc_score(yteste_fonte, pred_fonte)
ap_fonte    = average_precision_score(yteste_fonte, pred_fonte)

auroc_tl    = roc_auc_score(yteste_destino, pred_transferencia)
ap_tl       = average_precision_score(yteste_destino, pred_transferencia)

auroc_local = roc_auc_score(yteste_destino, pred_sergipe)
ap_local    = average_precision_score(yteste_destino, pred_sergipe)

print("Pernambuco:", X_fonte.shape[0], "casos | óbitos por TB:", int(y_fonte.sum()))
print("Sergipe:", X_destino.shape[0], "casos | óbitos por TB:", int(y_destino.sum()))

print("\nDesempenho dos modelos:")
print(f"Modelo da fonte (Pernambuco)      - AUROC: {round(auroc_fonte, 3)} | Precisão média: {round(ap_fonte, 3)}")

print(f"Modelo treinado apenas em Sergipe - AUROC: {round(auroc_local, 3)} | Precisão média: {round(ap_local, 3)}")

print(f"Modelo com transferência          - AUROC: {round(auroc_tl, 3)} | Precisão média: {round(ap_tl, 3)}")

print("\nResumo final no domínio destino:")
print(f"Ganho de AUROC (TL - local): {auroc_tl - auroc_local:+.3f}")
print(f"Ganho de precisão média (TL - local): {ap_tl - ap_local:+.3f}")

# %% [markdown]
# ### Verificações complementares
#
# A comparação de AUROC/AP no domínio destino é ilustrativa; a transferência não garante melhora. A comparação com o modelo da fonte usa outro domínio de teste e não é diretamente equivalente.

# %% [code] célula 8
import numpy as np
assert len(Xteste_destino)==len(pred_transferencia)==len(pred_sergipe)
assert np.isfinite(pred_transferencia).all() and np.isfinite(pred_sergipe).all()
assert (pred_transferencia>=0).all() and (pred_transferencia<=1).all()
assert (pred_sergipe>=0).all() and (pred_sergipe<=1).all()
print("Verificações de tamanho e probabilidades: OK")
