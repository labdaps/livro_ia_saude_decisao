# -*- coding: utf-8 -*-
# Capítulo 16: exportação automática das células do notebook executável.
# Executar na raiz do projeto: python python/capitulo_16.py
# Células que dependem de contexto de notebook, serviços ou bibliotecas opcionais podem exigir ajustes.

# %% [markdown]
# # Capítulo 16 — Versão executável para validação técnica
#
# **Dados sintéticos.** A biblioteca `boruta` está ausente: a seleção alternativa abaixo é apenas um teste de integração e **não reproduz o método Boruta**. Os códigos originais estão no notebook separado.

# %% [code] célula 2
import numpy as np, pandas as pd
from copy import deepcopy
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, auc
from sklearn.inspection import permutation_importance
import plotly.graph_objects as go
import importlib.util
seed = 42
BORUTA_DISPONIVEL = importlib.util.find_spec('boruta') is not None
print('Boruta disponível:', BORUTA_DISPONIVEL)

# %% [markdown]
# ## Bloco 16.1 — Instalação das bibliotecas Boruta e XGBoost

# %% [code] célula 4
print('Instalação do Boruta: indisponível neste ambiente; comando original preservado no notebook original.')

# %% [markdown]
# ## Bloco 16.2 — Importação das bibliotecas utilizadas na análise

# %% [code] célula 6
print('Importações disponíveis carregadas na célula de preparação.')

# %% [markdown]
# ## Bloco 16.3 — Importação e organização das bases de treino e teste

# %% [code] célula 8
# Dados sintéticos no lugar dos CSVs externos; separação estratificada.
X, y = make_classification(n_samples=350,n_features=12,n_informative=5,n_redundant=2,weights=[.72,.28],random_state=seed)
X = pd.DataFrame(X,columns=[f'variavel_{i:02d}' for i in range(X.shape[1])])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,stratify=y,random_state=seed)
print('Treino:',X_train.shape,'Teste:',X_test.shape)

# %% [markdown]
# ## Bloco 16.4 — Definição do XGBoost e do espaço de busca de hiperparâmetros

# %% [code] célula 10
xgb = XGBClassifier(objective='binary:logistic',random_state=seed,n_jobs=1,eval_metric='logloss',tree_method='hist')
xgb_dict={'n_estimators':[20,40],'max_depth':[2,3],'learning_rate':[.05,.1]}
print('Espaço de busca reduzido para teste técnico:',xgb_dict)

# %% [markdown]
# ## Bloco 16.5 — Ajuste do XGBoost por busca aleatória com validação

# %% [code] célula 12
xgb_rs=RandomizedSearchCV(xgb,xgb_dict,n_iter=3,scoring='roc_auc',cv=3,random_state=seed,n_jobs=1)
xgb_rs.fit(X_train,y_train)
xgb=xgb_rs.best_estimator_
print('Melhor AUC CV:',round(xgb_rs.best_score_,3))

# %% [markdown]
# ## Bloco 16.6 — Seleção de variáveis com o algoritmo Boruta

# %% [code] célula 14
if BORUTA_DISPONIVEL:
    from boruta import BorutaPy
    boruta = BorutaPy(deepcopy(xgb), n_estimators='auto', random_state=seed, alpha=.05, max_iter=15)
    boruta.fit(X_train.to_numpy(),np.asarray(y_train))
    metodo_selecao='BorutaPy (máximo 15 iterações para teste)'
    suporte=boruta.support_; tentativas=boruta.support_weak_; ranking=boruta.ranking_
else:
    # Método alternativo: importância por permutação no treino; NÃO é Boruta.
    imp=permutation_importance(xgb,X_train,y_train,n_repeats=3,random_state=seed,scoring='roc_auc',n_jobs=1)
    suporte=imp.importances_mean>0
    if not suporte.any(): suporte[np.argmax(imp.importances_mean)]=True
    tentativas=np.zeros(X_train.shape[1],dtype=bool)
    ranking=np.where(suporte,1,2)
    metodo_selecao='IMPORTÂNCIA POR PERMUTAÇÃO — alternativa; Boruta não validado'
print('Método:',metodo_selecao)

# %% [markdown]
# ## Bloco 16.7 — Extração das variáveis confirmadas, rejeitadas e tentativas

# %% [code] célula 16
features_confirmadas=X_train.columns[suporte].to_list()
features_rejeitadas=X_train.columns[~suporte & ~tentativas].to_list()
features_tentativas=X_train.columns[tentativas].to_list()
print('Selecionadas:',features_confirmadas)
print('Rejeitadas:',features_rejeitadas)
print('Tentativas:',features_tentativas)

# %% [markdown]
# ## Bloco 16.8 — Organização do ranking de variáveis gerado pelo Boruta

# %% [code] célula 18
feature_ranking=pd.DataFrame({'Feature':X_train.columns,'Ranking':ranking,'Confirmada':suporte,'Tentativa':tentativas}).sort_values('Ranking')
print(feature_ranking.to_string(index=False))

# %% [markdown]
# ## Bloco 16.9 — Construção das bases com as variáveis selecionadas

# %% [code] célula 20
X_train_novo=X_train[features_confirmadas].copy()
X_test_novo=X_test[features_confirmadas].copy()
print('Variáveis após seleção:',X_train_novo.shape[1])

# %% [markdown]
# ## Bloco 16.10 — Retreinamento do XGBoost com as variáveis confirmadas

# %% [code] célula 22
xgb_novo_rs=RandomizedSearchCV(XGBClassifier(objective='binary:logistic',random_state=seed,n_jobs=1,eval_metric='logloss',tree_method='hist'),xgb_dict,n_iter=3,scoring='roc_auc',cv=3,random_state=seed,n_jobs=1)
xgb_novo_rs.fit(X_train_novo,y_train)
xgb_novo=xgb_novo_rs.best_estimator_
print('Melhor AUC CV reduzido:',round(xgb_novo_rs.best_score_,3))

# %% [markdown]
# ## Bloco 16.11 — Comparação das curvas ROC dos modelos original e reduzido

# %% [code] célula 24
y_prob_xgb_novo=xgb_novo.predict_proba(X_test_novo)[:,1]
y_prob_xgb=xgb.predict_proba(X_test)[:,1]
fpr_xgb_novo,tpr_xgb_novo,_=roc_curve(y_test,y_prob_xgb_novo)
fpr_xgb,tpr_xgb,_=roc_curve(y_test,y_prob_xgb)
roc_auc_xgb_novo=auc(fpr_xgb_novo,tpr_xgb_novo)
roc_auc_xgb=auc(fpr_xgb,tpr_xgb)
fig=go.Figure()
fig.add_trace(go.Scatter(x=fpr_xgb_novo,y=tpr_xgb_novo,mode='lines',name=f'Reduzido (AUC={roc_auc_xgb_novo:.3f})'))
fig.add_trace(go.Scatter(x=fpr_xgb,y=tpr_xgb,mode='lines',name=f'Original (AUC={roc_auc_xgb:.3f})'))
fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='Aleatório',line=dict(dash='dash')))
fig.update_layout(title='ROC — dados sintéticos; seleção: '+metodo_selecao,xaxis_title='FPR',yaxis_title='TPR')
print('AUC original:',round(roc_auc_xgb,3),'AUC reduzido:',round(roc_auc_xgb_novo,3))
# Gráfico gerado sem renderizador interativo obrigatório.
assert len(fpr_xgb)>1 and len(fpr_xgb_novo)>1
