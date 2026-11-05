# -*- coding: utf-8 -*-
# Capítulo 21: exportação automática das células do notebook executável.
# Executar na raiz do projeto: python python/capitulo_21.py
# Células que dependem de contexto de notebook, serviços ou bibliotecas opcionais podem exigir ajustes.

# %% [markdown]
# # Capítulo 21 — versão executável demonstrativa
#
# **Atenção:** os blocos originais são reproduzidos abaixo como texto (não executados), para evitar substituir silenciosamente `river`/`deep_river`. A demonstração usa `sklearn.linear_model.SGDClassifier`, que **não é** Hoeffding Tree nem Deep River. Todos os dados são sintéticos.

# %% [code] célula 2
import importlib.util
for p in ["river", "deep_river", "torch"]:
    print(p, "disponível" if importlib.util.find_spec(p) else "INDISPONÍVEL")

# %% [markdown]
# ### Bloco de Código 21.1 — Classificação incremental com uma Hoeffding Tree
#
# *Página 437 do PDF · Revisão: confirmado*
#
# **Código original preservado, não executado nesta versão:**
#
# ```python
# from river import tree, metrics
#
# # Hoeffding Tree para classificação
# model = tree.HoeffdingTreeClassifier()
# metric = metrics.Accuracy()
#
# for xi, yi in data_stream:
#     y_pred = model.predict_one(xi)
#     metric.update(yi, y_pred)
#     model.learn_one(xi, yi)
#
# print(f'Acurácia: {metric.get():.3f}')
#
# ```

# %% [markdown]
# **Diagnóstico 21.1:** requer `river` e `data_stream`, ambos indisponíveis no ambiente original.

# %% [markdown]
# ### Bloco de Código 21.2 — Rede neural online com Deep River e PyTorch.
#
# *Página 439 do PDF · Revisão: confirmado*
#
# **Código original preservado, não executado nesta versão:**
#
# ```python
# from river import metrics, compose, preprocessing
# from deep_river.classification import Classifier
# from torch import nn
#
# class MyModule(nn.Module):
#     def __init__(self, n_features):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(n_features, 16),
#             nn.ReLU(),
#             nn.Linear(16, 2)
#         )
#     
#     def forward(self, x):
#         return self.net(x)
#
# model = compose.Pipeline(
#     preprocessing.StandardScaler(),
#     Classifier(module=MyModule, n_features=9)
# )
#
# metric = metrics.ROCAUC()
# for xi, yi in data_stream:
#     y_prob = model.predict_proba_one(xi)
#     metric.update(yi, y_prob)
#     model.learn_one(xi, yi)
#
# ```

# %% [markdown]
# **Diagnóstico 21.2:** requer `deep_river`, `river` e `data_stream`. A arquitetura declara 9 atributos, enquanto o bloco 21.4 seleciona 5; são exemplos distintos, não diretamente encadeáveis.

# %% [markdown]
# ### Bloco de Código 21.3 — Importação e filtragem dos dados do SINAN-TB.
#
# *Página 440 do PDF · Revisão: confirmado*
#
# **Código original preservado, não executado nesta versão:**
#
# ```python
# import pandas as pd
#
# df = pd.read_csv('dados_tuberculose_v2.csv')
#
# colunas = ['NU_ANO', 'SG_UF_NOT', 'FORMA',
#            'RAIOX_TORA', 'IDADE_ANOS', 'SITUA_ENCE']
# df = df[colunas].copy()
#
# # Manter apenas cura (1) e abandono (2)
# df = df[df['SITUA_ENCE'].isin([1, 2])]
#
# print(f'Total de casos: {len(df)}')
#
# ```

# %% [markdown]
# **Adaptação para 21.3 e 21.4:** gerar base sintética com o esquema mínimo; os dados não são registros reais do SINAN-TB. Os códigos originais são mantidos acima.

# %% [code] célula 9
import numpy as np, pandas as pd
rng=np.random.default_rng(21)
n=12000
base=pd.DataFrame({
 "NU_ANO":rng.integers(2016,2025,n),
 "SG_UF_NOT":rng.choice([26,28,35],n),
 "FORMA":rng.choice([1,2,3],n),
 "RAIOX_TORA":rng.choice([1,2,3],n),
 "IDADE_ANOS":rng.integers(18,91,n),
})
p=1/(1+np.exp(-(-2+0.025*(base['IDADE_ANOS']-40)+0.25*(base['FORMA']==2)+0.15*(base['SG_UF_NOT']==28))))
base['SITUA_ENCE']=np.where(rng.random(n)<p,2,1)
base.to_csv('dados_tuberculose_v2.csv',index=False)
print('Dados sintéticos:',base.shape)

# %% [markdown]
# ### Bloco de Código 21.4 — Pré-processamento dos dados para aprendizado online.
#
# *Página 440 do PDF · Revisão: confirmado*
#
# **Código original preservado, não executado nesta versão:**
#
# ```python
# # Remover linhas com valores ausentes
# df = df.dropna()
#
# # Criar variável alvo: 1 = abandono, 0 = cura
# df['alvo'] = (df['SITUA_ENCE'] == 2).astype(int)
#
# # Ordenar por ano (simula ordem cronológica)
# df = df.sort_values('NU_ANO').reset_index(drop=True)
#
# # Separar features e alvo
# features = ['SG_UF_NOT', 'FORMA', 'RAIOX_TORA', 'IDADE_ANOS', 'NU_ANO']
# X = df[features]
# y = df['alvo']
#
# print(f'Cura: {(y==0).sum()} | Abandono: {(y==1).sum()}')
#
# ```

# %% [code] célula 11
df=base.copy()
# Remover linhas com valores ausentes
df = df.dropna()

# Criar variável alvo: 1 = abandono, 0 = cura
df['alvo'] = (df['SITUA_ENCE'] == 2).astype(int)

# Ordenar por ano (simula ordem cronológica)
df = df.sort_values('NU_ANO').reset_index(drop=True)

# Separar features e alvo
features = ['SG_UF_NOT', 'FORMA', 'RAIOX_TORA', 'IDADE_ANOS', 'NU_ANO']
X = df[features]
y = df['alvo']

print(f'Cura: {(y==0).sum()} | Abandono: {(y==1).sum()}')

# %% [code] célula 12
assert len(X)==12000
assert set(y.unique())=={0,1}
print('Validação: preparação concluída; amostras=',len(X))

# %% [markdown]
# ### Bloco de Código 21.5 — Treinamento prequential com uma Hoeffding Tree.
#
# *Página 441 do PDF · Revisão: confirmado*
#
# **Código original preservado, não executado nesta versão:**
#
# ```python
# from river import tree, metrics
#
# model = tree.HoeffdingTreeClassifier()
# metric = metrics.Accuracy()
#
# # Guardar a acurácia a cada 10.000 casos
# historico = []
#
# for i in range(len(X)):
#     xi = X.iloc[i].to_dict()
#     yi = y.iloc[i]
#
#     # Test: prediz antes de aprender
#     y_pred = model.predict_one(xi)
#     metric.update(yi, y_pred)
#
#     # Train: aprende com o exemplo
#     model.learn_one(xi, yi)
#
#     if (i + 1) % 10000 == 0:
#         historico.append({
#             'exemplos': i + 1,
#             'acuracia': metric.get()
#         })
#
# print(f'Acurácia final: {metric.get():.3f}')
#
# ```

# %% [markdown]
# **Adaptação 21.5:** demonstração incremental *test-then-train* com `SGDClassifier.partial_fit` (não é Hoeffding Tree). Escalonamento incremental omitido para preservar simplicidade; variáveis normalizadas por constantes fixas, sem aprender estatísticas no teste.

# %% [code] célula 15
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path
model_demo=SGDClassifier(loss='log_loss',random_state=21)
X_demo=X.astype(float).copy()
X_demo['IDADE_ANOS']/=100
X_demo['NU_ANO']=(X_demo['NU_ANO']-2015)/10
X_demo['SG_UF_NOT']/=100
X_demo['FORMA']/=5
X_demo['RAIOX_TORA']/=5
historico=[]; corretos=0; avaliados=0
for i,(xi,yi) in enumerate(zip(X_demo.to_numpy(),y.to_numpy())):
    if i>0:
        pred=int(model_demo.predict(xi.reshape(1,-1))[0])
        corretos+=int(pred==yi); avaliados+=1
    model_demo.partial_fit(xi.reshape(1,-1),np.array([yi]),classes=np.array([0,1]))
    if (i+1)%1000==0:
        historico.append({'exemplos':i+1,'acuracia':corretos/avaliados if avaliados else np.nan})
print('Acurácia prequential demonstrativa:',round(corretos/avaliados,3),'em',avaliados,'predições')
assert len(historico)==12

# %% [markdown]
# ### Bloco de Código 21.6 — Visualização da evolução da acurácia do modelo online.
#
# *Página 442 do PDF · Revisão: confirmado*
#
# **Código original preservado, não executado nesta versão:**
#
# ```python
# import matplotlib.pyplot as plt
#
# hist = pd.DataFrame(historico)
# plt.figure(figsize=(8, 4))
# plt.plot(hist['exemplos'], hist['acuracia'])
# plt.xlabel('Numero de exemplos vistos')
# plt.ylabel('Acuracia acumulada')
# plt.title('Evolucao da acuracia - Hoefdding Tree')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('figuras/evolucao_acuracia.png', dpi=150)
# plt.show()
#
# ```

# %% [markdown]
# **Adaptação 21.6:** a frequência de histórico é 1.000 amostras, não 10.000, e a pasta de saída é criada explicitamente.

# %% [code] célula 18
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
Path('figuras').mkdir(exist_ok=True)
hist=pd.DataFrame(historico)
fig,ax=plt.subplots(figsize=(8,4))
ax.plot(hist['exemplos'],hist['acuracia'],marker='o')
ax.set(xlabel='Exemplos vistos',ylabel='Acurácia prequential acumulada',title='SGDClassifier incremental — demonstração sintética')
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig('figuras/evolucao_acuracia.png',dpi=130)
plt.show()
assert Path('figuras/evolucao_acuracia.png').exists()
print('Figura salva com',len(hist),'pontos')
