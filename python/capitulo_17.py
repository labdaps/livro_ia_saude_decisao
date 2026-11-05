# -*- coding: utf-8 -*-
# Capítulo 17: exportação automática das células do notebook executável.
# Executar na raiz do projeto: python python/capitulo_17.py
# Células que dependem de contexto de notebook, serviços ou bibliotecas opcionais podem exigir ajustes.

# %% [markdown]
# # Capítulo 17 — Explicabilidade SHAP
#
# Códigos revisados, preservados conforme SQLite e arquivos `.py`. Execução sequencial pode depender de dados externos, bibliotecas ou estado anterior. Não executar sem revisar efeitos colaterais.
#
# **Versão executável:** ajustes identificados em notas específicas; dados simulados, sem validação clínica.

# %% [markdown]
# ### Bloco de Código 17.1 — Ajuste e visualização de importância de variáveis em árvore de decisão intrínseca aplicada ao SINAN-TB.
#
# *Página 363 do PDF · Revisão: confirmado*

# %% [markdown]
# **Adaptação 17.1:** harmonização de `Obito` (coluna criada) com as referências em `X` e `y`. Código original preservado no outro notebook.

# %% [code] célula 4
# 1. SIMULACAO DOS DADOS (SINAN-TB)
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

np.random.seed(42)
n = 2000

# Simula variaveis epidemiologicas do SINAN-TB
idade = np.random.randint(15, 90, n)
sexo_masc = np.random.binomial(1, 0.65, n)
escolaridade = np.random.randint(0, 8, n)
alcoolismo = np.random.binomial(1, 0.30, n)
hiv = np.random.binomial(1, 0.12, n)
drogas = np.random.binomial(1, 0.18, n)
pop_rua = np.random.binomial(1, 0.10, n)
diabetes = np.random.binomial(1, 0.15, n)
forma_pulm = np.random.binomial(1, 0.75, n)

# Desfecho: obito (1) ou cura/outro (0)
logit = (
    -3.0 + 0.04 * idade - 0.08 * escolaridade
    + 0.60 * alcoolismo + 0.90 * hiv + 0.50 * drogas
    + 0.70 * pop_rua + 0.30 * diabetes - 0.20 * forma_pulm
)
prob = 1 / (1 + np.exp(-logit))  # funcao logistica
obito = np.random.binomial(1, prob, n)

df = pd.DataFrame({
    "Idade": idade,
    "Sexo Masculino": sexo_masc,
    "Escolaridade": escolaridade,
    "Alcoolismo": alcoolismo,
    "HIV Positivo": hiv,
    "Drogas": drogas,
    "Pop. Rua": pop_rua,
    "Diabetes": diabetes,
    "Forma Pulmonar": forma_pulm,
    "Obito": obito
})

X = df.drop(columns=["Obito"])
y = df["Obito"]

# 2. DIVISAO TREINO/TESTE
# stratify=y preserva a proporcao de obitos em ambos os conjuntos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. TREINAMENTO - MODELO INTRINSECO
# max_depth=3: arvore legivel; cada caminho da raiz a folha e
# diretamente interpretavel
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)  # ajuste apenas nos dados de treino

# %% [markdown]
# ### Bloco de Código 17.2 — Valores SHAP com TreeSHAP no XGBoost
#
# *Página 375 do PDF · Revisão: confirmado*

# %% [markdown]
# **Validação 17.2:** bibliotecas SHAP e XGBoost locais; os gráficos e as explicações referem-se exclusivamente aos dados simulados.

# %% [code] célula 7
import shap
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# 1. TREINAMENTO DO MODELO XGBOOST
# scale_pos_weight corrige o desbalanceamento entre cura (~95%) e obito (~5%)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model_xgb = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss"
)

model_xgb.fit(X_train, y_train)

# 2. CALCULO DOS VALORES SHAP VIA TREESHAP
# shap_values: array (n x p) - usado em summary_plot e dependence_plot
# shap_exp: objeto Explanation - usado em waterfall e bar por cohort
explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer.shap_values(X_test)
shap_exp = explainer(X_test)

# 3. BEESWARM PLOT - importancia global de cada variavel
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
plt.title("Beeswarm SHAP - Importancia global das variaveis (SINAN-TB)")
plt.tight_layout()
plt.show()

# 4. WATERFALL PLOT - explicacao local para a instancia de maior risco predito
probs = model_xgb.predict_proba(X_test)[:, 1]  # probabilidade de obito
idx = int(np.argmax(probs))  # paciente de maior risco

plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_exp[idx], show=False, max_display=9)
plt.title(f"Waterfall SHAP - Explicacao local (Paciente #{X_test.index[idx]})")
plt.tight_layout()
plt.show()

# 5. DEPENDENCE PLOT - efeito da Idade colorido por HIV Positivo
fig, ax = plt.subplots(figsize=(9, 5))

shap.dependence_plot(
    "Idade",
    shap_values,
    X_test,
    interaction_index="HIV Positivo",
    ax=ax,
    show=False
)

ax.set_title("Dependence Plot - Idade x HIV Positivo (SINAN-TB)")
plt.tight_layout()
plt.show()
