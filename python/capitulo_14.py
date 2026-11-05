# -*- coding: utf-8 -*-
# Capítulo 14: exportação automática das células do notebook executável.
# Executar na raiz do projeto: python python/capitulo_14.py
# Células que dependem de contexto de notebook, serviços ou bibliotecas opcionais podem exigir ajustes.

# %% [markdown]
# # Capítulo 14 — árvores, florestas e boosting (executável)
#
# **Validação técnica com dados sintéticos, não clínicos.** Os códigos recebidos permanecem integralmente no notebook original. As adaptações abaixo não reproduzem resultados do SINAN-TB.

# %% [markdown]
# ### Bloco de Código 14.1 — Criação do conjunto de dados simulado para árvores
#
# *Página 258 do PDF · Revisão: confirmado*
#
# **Código original:** disponível integralmente no notebook `capitulo_14_original.ipynb`. **Abaixo:** implementação para validação técnica.

# %% [code] célula 3
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.inspection import DecisionBoundaryDisplay
SEED=42
simu_df=pd.DataFrame({'IDADE_ANOS':[57.,32.,26.,78.,51.,51.,44.,21.,26.,51.], 'ESCOLARIDADE':['Ensino médio incompleto','Ensino médio incompleto','1a a 4ª série incompleta do EF','5ª a 8ª série incompleta do EF','Ensino fundamental completo','Ensino médio incompleto','5ª a 8ª série incompleta do EF','5ª a 8ª série incompleta do EF','Ensino médio completo','5ª a 8ª série incompleta do EF'], 'OBITO':[1,0,0,1,1,1,0,0,0,1]})
print(simu_df.shape,simu_df['OBITO'].value_counts().to_dict())

# %% [markdown]
# ### Bloco de Código 14.2 — Codificação das variáveis categóricas do conjunto simulado com one-hot encoding
#
# *Página 258 do PDF · Revisão: confirmado*
#
# **Código original:** disponível integralmente no notebook `capitulo_14_original.ipynb`. **Abaixo:** implementação para validação técnica.

# %% [code] célula 5
simu_df=pd.get_dummies(simu_df,columns=['ESCOLARIDADE'],prefix='ESCOLARIDADE')
X_simu=simu_df.drop(columns='OBITO').astype(float);y_simu=simu_df.OBITO.copy()
print('Variáveis:',X_simu.columns.tolist());assert len(X_simu)==len(y_simu)==10

# %% [markdown]
# ### Bloco de Código 14.3 — Ajuste e visualização da árvore de decisão no conjunto
#
# *Página 259 do PDF · Revisão: confirmado*
#
# **Código original:** disponível integralmente no notebook `capitulo_14_original.ipynb`. **Abaixo:** implementação para validação técnica.

# %% [code] célula 7
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

arvore_simu = DecisionTreeClassifier(criterion='gini', random_state=SEED)
arvore_simu.fit(X_simu, y_simu)

plt.figure(figsize=(12, 8))
plot_tree(arvore_simu, filled=True, feature_names=X_simu.columns, class_names=['Nao Obito', 'Obito'])
plt.show()

assert arvore_simu.tree_.node_count>0

# %% [markdown]
# ### Bloco de Código 14.4 — Visualização das regiões de decisão da árvore simulada
#
# *Página 260 do PDF · Revisão: confirmado*
#
# **Código original:** disponível integralmente no notebook `capitulo_14_original.ipynb`. **Abaixo:** implementação para validação técnica.

# %% [code] célula 9
col=[c for c in X_simu if c.startswith('ESCOLARIDADE_')][0]
X_plot=X_simu[['IDADE_ANOS',col]]
arvore_simu_2d=DecisionTreeClassifier(criterion='gini',random_state=SEED).fit(X_plot,y_simu)
fig,ax=plt.subplots(figsize=(7,5))
DecisionBoundaryDisplay.from_estimator(arvore_simu_2d,X_plot,response_method='predict',ax=ax,alpha=.35)
ax.scatter(X_plot.iloc[:,0],X_plot.iloc[:,1],c=y_simu,edgecolor='k');ax.set_title('Regiões de decisão — simulação');plt.show()

# %% [markdown]
# ### Bloco de Código 14.5 — Preparação dos dados reais de tuberculose para o modelo de árvore de decisão
#
# *Página 261 do PDF · Revisão: confirmado*
#
# **Código original:** disponível integralmente no notebook `capitulo_14_original.ipynb`. **Abaixo:** implementação para validação técnica.

# %% [code] célula 11
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split,StratifiedKFold,RandomizedSearchCV
from sklearn.metrics import roc_auc_score
X,y=make_classification(n_samples=220,n_features=10,n_informative=6,n_redundant=1,weights=[.75,.25],random_state=SEED)
X=pd.DataFrame(X,columns=[f'atributo_{j}' for j in range(10)])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,stratify=y,random_state=SEED)
cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED)
print('Dados SINTÉTICOS:',X_train.shape,X_test.shape,'classes:',np.bincount(y_train))

# %% [markdown]
# ### Bloco de Código 14.6 — Busca aleatória de hiperparâmetros para a árvore de decisão com validação cruzada
#
# *Página 262 do PDF · Revisão: confirmado*
#
# **Código original:** disponível integralmente no notebook `capitulo_14_original.ipynb`. **Abaixo:** implementação para validação técnica.

# %% [code] célula 13
search_tree=RandomizedSearchCV(DecisionTreeClassifier(class_weight='balanced',random_state=SEED),{'max_depth':[2,3,5,None],'min_samples_leaf':[1,3,6],'criterion':['gini','entropy']},n_iter=4,cv=cv,scoring='roc_auc',random_state=SEED,n_jobs=1)
search_tree.fit(X_train,y_train);best_clf=search_tree.best_estimator_
print('Árvore:',search_tree.best_params_,'AUC CV:',round(search_tree.best_score_,3))

# %% [markdown]
# ### Bloco de Código 14.7 — Busca aleatória de hiperparâmetros e avaliação da floresta aleatória
#
# *Página 271 do PDF · Revisão: confirmado*
#
# **Código original:** disponível integralmente no notebook `capitulo_14_original.ipynb`. **Abaixo:** implementação para validação técnica.

# %% [code] célula 15
from sklearn.ensemble import RandomForestClassifier
search_rf=RandomizedSearchCV(RandomForestClassifier(class_weight='balanced',random_state=SEED,n_jobs=1),{'n_estimators':[15,25,35],'max_depth':[3,5,None],'min_samples_leaf':[1,3]},n_iter=4,cv=cv,scoring='roc_auc',random_state=SEED,n_jobs=1)
search_rf.fit(X_train,y_train);rf_best_clf=search_rf.best_estimator_
print('Random Forest:',search_rf.best_params_,'AUC CV:',round(search_rf.best_score_,3))

# %% [markdown]
# ### Bloco de Código 14.8 — Busca aleatória de hiperparâmetros e avaliação do modelo XGBoost
#
# *Página 291 do PDF · Revisão: confirmado*
#
# **Código original:** disponível integralmente no notebook `capitulo_14_original.ipynb`. **Abaixo:** implementação para validação técnica.

# %% [code] célula 17
from xgboost import XGBClassifier
search_xgb=RandomizedSearchCV(XGBClassifier(random_state=SEED,eval_metric='logloss',n_jobs=1,tree_method='hist'),{'n_estimators':[15,25,35],'max_depth':[2,3,4],'learning_rate':[.05,.1],'subsample':[.8,1.]},n_iter=4,cv=cv,scoring='roc_auc',random_state=SEED,n_jobs=1)
search_xgb.fit(X_train,y_train);xgb_best_clf=search_xgb.best_estimator_
print('XGBoost:',search_xgb.best_params_,'AUC CV:',round(search_xgb.best_score_,3))

# %% [markdown]
# ### Bloco de Código 14.11 — Construção das curvas ROC para comparação entre
#
# *Página 294 do PDF · Revisão: confirmado*
#
# **Código original:** disponível integralmente no notebook `capitulo_14_original.ipynb`. **Abaixo:** implementação para validação técnica.

# %% [code] célula 19
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_curve,auc
catboost_best_clf=CatBoostClassifier(iterations=25,depth=3,learning_rate=.1,verbose=False,random_seed=SEED,thread_count=1).fit(X_train,y_train)
lgbm_best_clf=LGBMClassifier(n_estimators=25,max_depth=3,num_leaves=7,min_child_samples=5,verbosity=-1,random_state=SEED,n_jobs=1).fit(X_train,y_train)
fig,ax=plt.subplots(figsize=(7,5))
for nome,modelo in [('XGBoost',xgb_best_clf),('CatBoost',catboost_best_clf),('LightGBM',lgbm_best_clf)]:
 prob=modelo.predict_proba(X_test)[:,1];fpr,tpr,_=roc_curve(y_test,prob);score=auc(fpr,tpr)
 ax.plot(fpr,tpr,label=f'{nome} (AUC={score:.3f})');print(nome,'AUC teste:',round(score,3))
ax.plot([0,1],[0,1],'--',label='Aleatório');ax.set(xlabel='FPR',ylabel='TPR',title='ROC — dados sintéticos');ax.legend();plt.show()
