# -*- coding: utf-8 -*-
# Capítulo 08: exportação automática das células do notebook executável.
# Executar na raiz do projeto: python python/capitulo_08.py
# Células que dependem de contexto de notebook, serviços ou bibliotecas opcionais podem exigir ajustes.

# %% [markdown]
# # Capítulo 8 — Dados desbalanceados: edição executável
#
# **Importante:** os dados reais do SINAN-TB não foram fornecidos. Este notebook usa **dados sintéticos** para verificar o fluxo computacional; métricas obtidas não reproduzem os resultados do livro. Códigos originais revisados estão em células separadas; preparação, calibração e testes são complementos nossos.

# %% [markdown]
# ## Preparação reprodutível (complemento)
#
# Geração de dados tabulares sintéticos com desbalanceamento e separação estratificada treino/validação/teste. Nenhuma amostra do teste participa do treinamento, SMOTE ou calibração.

# %% [code] célula 3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from collections import Counter

X, y = make_classification(n_samples=900, n_features=12, n_informative=7,
    n_redundant=2, weights=[0.88, 0.12], flip_y=0.01, random_state=42)
X = pd.DataFrame(X, columns=[f"atributo_{i}" for i in range(X.shape[1])])
y = pd.Series(y, name="obito_sintetico")
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42)
print("Treino:", X_train.shape, Counter(y_train))
print("Validação:", X_val.shape, Counter(y_val))
print("Teste:", X_test.shape, Counter(y_test))
assert set(X_train.index).isdisjoint(X_val.index)
assert set(X_train.index).isdisjoint(X_test.index)
assert set(X_val.index).isdisjoint(X_test.index)

# %% [markdown]
# ### Bloco de Código 8.1 — Aplicação de reamostragem sintética (SMOTE) à base
#
# *Página 120 do PDF · Revisão: confirmado*
#
# **Fonte:** código revisado do livro; conteúdo abaixo preservado.

# %% [code] célula 5
from collections import Counter
from imblearn.over_sampling import SMOTE

# 1. FUNCAO DE VISUALIZACAO

# Recebe o vetor de rotulos (y) e exibe a contagem de cada classe.
def visualizar_classes(y, titulo):
    counts = Counter(y)
    df_counts = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    df_counts.columns = ['Classe', 'Quantidade']
    print(f'n--- Distribuicao de Classes: {titulo},')
    print(df_counts.to_string(index=False))

# 2. VERIFICACAO DO DESBALANCEAMENTO

visualizar_classes(y_train, "y_train (Original)")

# 3. APLICACAO DO SMOTE

# IMPORTANTE: o SMOTE deve ser aplicado SOMENTE sobre X_train e y_train.
# X_val e X_test permanecem intocados para garantir avaliacao sem vazamento.
# O X_train neste ponto ja passou por imputacao e OHE (apenas variaveis
# numericas e dummies), pois o SMOTE exige entradas completamente numericas.
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Exibe a contagem após o SMOTE: a classe minoritaria (obito=1) e
# sinteticamente expandida até igual a classe majoritaria (obito=0).
visualizar_classes(y_train_res, "y_train (Resampled)")

# %% [code] célula 6
assert len(X_train_res) > len(X_train)
assert len(set(Counter(y_train_res).values())) == 1
assert len(X_test) == 180
print("SMOTE: somente treino, classes balanceadas")

# %% [markdown]
# ### Bloco de Código 8.2 — Saída do console (output), base SINAN-TB
#
# *Página 120 do PDF · Revisão: confirmado*
#
# **Fonte:** código revisado do livro; conteúdo abaixo preservado.

# %% [markdown]
# **Saída ilustrativa do livro (não executável):** o bloco transcrito contém somente o comentário `# Sem código`. Os resultados numéricos do exemplo real não são reproduzidos com a base sintética.

# %% [code] célula 9
# Sem código

# %% [markdown]
# ### Bloco de Código 8.3 — Modelo 0, Baseline sem correção do desbalanceamento
#
# *Página 122 do PDF · Revisão: confirmado*
#
# **Fonte:** código revisado do livro; conteúdo abaixo preservado.

# %% [code] célula 11
xgb_default = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
xgb_default.fit(X_train, y_train)
y_pred_default = xgb_default.predict(X_test)

# %% [code] célula 12
from sklearn.metrics import classification_report
print("Baseline — teste sintético:")
print(classification_report(y_test, y_pred_default, zero_division=0))

# %% [markdown]
# ### Bloco de Código 8.4 — Saída: Modelo 0, Default
#
# *Página 122 do PDF · Revisão: confirmado*
#
# **Fonte:** código revisado do livro; conteúdo abaixo preservado.

# %% [markdown]
# **Saída ilustrativa do livro (não executável):** o bloco transcrito contém somente o comentário `# Sem código`. Os resultados numéricos do exemplo real não são reproduzidos com a base sintética.

# %% [code] célula 15
# SEM CODIGO

# %% [markdown]
# ### Bloco de Código 8.5 — Modelo 1, Aprendizado sensível a custo
#
# *Página 123 do PDF · Revisão: confirmado*
#
# **Fonte:** código revisado do livro; conteúdo abaixo preservado.

# %% [code] célula 17
# Razão negativos/positivos no conjunto de treino
peso_balanceamento = np.sum(y_train == 0) / np.sum(y_train == 1)

xgb_cost = xgb.XGBClassifier(
    scale_pos_weight=peso_balanceamento,
    random_state=42,
    eval_metric='logloss'
)

xgb_cost.fit(X_train, y_train)
y_pred_cost = xgb_cost.predict(X_test)

# %% [code] célula 18
assert peso_balanceamento > 1
print("Modelo sensível a custo — teste sintético:")
print(classification_report(y_test, y_pred_cost, zero_division=0))

# %% [markdown]
# ### Bloco de Código 8.6 — Saída: Modelo 1, Sensível a custo
#
# *Página 124 do PDF · Revisão: confirmado*
#
# **Fonte:** código revisado do livro; conteúdo abaixo preservado.

# %% [markdown]
# **Saída ilustrativa do livro (não executável):** o bloco transcrito contém somente o comentário `# Sem código`. Os resultados numéricos do exemplo real não são reproduzidos com a base sintética.

# %% [code] célula 21
# Sem código

# %% [markdown]
# ### Bloco de Código 8.7 — Modelo 3, Balanced Random Forest
#
# *Página 125 do PDF · Revisão: confirmado*
#
# **Fonte:** código revisado do livro; conteúdo abaixo preservado.

# %% [code] célula 23
from imblearn.ensemble import BalancedRandomForestClassifier

# sampling_strategy='all': balanceia internamente a cada arvore
brf = BalancedRandomForestClassifier(
    sampling_strategy='all',
    replacement=True,
    random_state=42
)

brf.fit(X_train, y_train)
y_pred_brf = brf.predict(X_test)

# %% [code] célula 24
print("Balanced RF — teste sintético:")
print(classification_report(y_test, y_pred_brf, zero_division=0))

# %% [markdown]
# ### Bloco de Código 8.8 — Saída: Modelo 3, Balanced Random Forest
#
# *Página 125 do PDF · Revisão: confirmado*
#
# **Fonte:** código revisado do livro; conteúdo abaixo preservado.

# %% [markdown]
# **Saída ilustrativa do livro (não executável):** o bloco transcrito contém somente o comentário `# Sem código`. Os resultados numéricos do exemplo real não são reproduzidos com a base sintética.

# %% [code] célula 27
# Sem código

# %% [markdown]
# ### Bloco de Código 8.9 — Varredura de limiares sobre probabilidades calibradas
#
# *Página 127 do PDF · Revisão: confirmado*
#
# **Fonte:** código revisado do livro; conteúdo abaixo preservado.

# %% [markdown]
# **Complemento:** calibração Platt usando exclusivamente o conjunto de validação. Modelo de base treinado no conjunto de treino. As probabilidades do teste não são usadas para ajustar o calibrador.

# %% [code] célula 30
modelo_calibracao = xgb.XGBClassifier(n_estimators=70, max_depth=3, n_jobs=2, random_state=42, eval_metric="logloss")
modelo_calibracao.fit(X_train, y_train)
prob_val = modelo_calibracao.predict_proba(X_val)[:, 1]
platt = LogisticRegression(random_state=42)
platt.fit(prob_val.reshape(-1, 1), y_val)
prob_platt = platt.predict_proba(modelo_calibracao.predict_proba(X_test)[:, 1].reshape(-1, 1))[:, 1]
assert len(prob_platt) == len(y_test)
assert np.isfinite(prob_platt).all()
print("Calibração independente concluída; AUC teste:", round(roc_auc_score(y_test, prob_platt), 3))

# %% [code] célula 31
from sklearn.metrics import precision_score, recall_score, f1_score

print(f"{'Limiar':>8} {'Precision':>10} {'Recall':>8} {'F1':>6}")
for thr in [0.50, 0.30, 0.18, 0.10]:
    y_pred = (prob_platt >= thr).astype(int)
    p = precision_score(y_test, y_pred, zero_division=0)
    r = recall_score(y_test, y_pred)
    f = f1_score(y_test, y_pred)
    print(f'{thr:>8.2f} {p:>10.2f} {r:>8.2f} {f:>6.2f}')

# %% [markdown]
# ### Bloco de Código 8.10 — Saída: tradeoff precision/recall por limiar, base SINAN-TB
#
# *Página 128 do PDF · Revisão: confirmado*
#
# **Fonte:** código revisado do livro; conteúdo abaixo preservado.

# %% [code] célula 33
from sklearn.metrics import precision_score, recall_score, f1_score

print(f"{'LIMIAR':>8} {'PRECISÃO':>10} {'RECALL':>8} {'F1':>6}")

for thr in [0.50, 0.30, 0.18, 0.10]:
    y_pred = (prob_platt >= thr).astype(int)
    p = precision_score(y_test, y_pred, zero_division=0)
    r = recall_score(y_test, y_pred)
    f = f1_score(y_test, y_pred)
    print(f"{thr:>8.2f} {p:>10.2f} {r:>8.2f} {f:>6.2f}")

# %% [markdown]
# ### Bloco de Código 8.11 — Relatório de classificação: limiar padrão vs. limiar ajustado
#
# *Página 129 do PDF · Revisão: confirmado*
#
# **Fonte:** código revisado do livro; conteúdo abaixo preservado.

# %% [code] célula 35
from sklearn.metrics import classification_report

print("--- Limiar padrão: 0.50,")

print(classification_report(y_test, (prob_platt >= 0.50).astype(int)))

print("--- Limiar ajustado: 0.18,")

print(classification_report(y_test, (prob_platt >= 0.18).astype(int)))

# %% [markdown]
# ### Bloco de Código 8.12 — Saída: comparação dos relatórios de classificação
#
# *Página 130 do PDF · Revisão: confirmado*
#
# **Fonte:** código revisado do livro; conteúdo abaixo preservado.

# %% [markdown]
# **Saída ilustrativa do livro (não executável):** o bloco transcrito contém somente o comentário `# Sem código`. Os resultados numéricos do exemplo real não são reproduzidos com a base sintética.

# %% [code] célula 38
# Sem código
