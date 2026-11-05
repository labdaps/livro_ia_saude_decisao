# -*- coding: utf-8 -*-
# Capítulo 13: exportação automática das células do notebook executável.
# Executar na raiz do projeto: python python/capitulo_13.py
# Células que dependem de contexto de notebook, serviços ou bibliotecas opcionais podem exigir ajustes.

# %% [markdown]
# # Capítulo 13 — Hiperparâmetros (versão executável)
#
# **Dados sintéticos**, sem inferência clínica. Os exemplos completos do livro estão preservados no notebook original. Aqui são usadas buscas menores para validar o funcionamento em tempo reduzido.

# %% [code] célula 2
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
seed=42
X,y=make_classification(n_samples=160,n_features=8,n_informative=5,n_redundant=1,weights=[.72,.28],random_state=seed)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,stratify=y,random_state=seed)
cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=seed)
print('Dados sintéticos:',X_train.shape,X_test.shape)

# %% [markdown]
# ### Bloco de Código 13.1 — Dicionário de hiperparâmetros para random forest
#
# *Página 236 do PDF · Revisão: confirmado*
#
# **Nota:** código integral preservado no notebook original; abaixo, adaptação de validação quando necessária.

# %% [code] célula 4
import numpy as np

# Random Forest
rf_dict = {
    'n_estimators': 100,     # Número de árvores
    'max_depth': 5,          # Profundidade máxima de cada árvore
    'min_samples_split': 2,  # Mínimo de amostras para dividir um nó
    'min_samples_leaf': 1,   # Mínimo de amostras em cada folha
    'max_features': 'sqrt'   # Número de atributos sorteados por divisão
}

assert rf_dict["n_estimators"] == 100
print(rf_dict)

# %% [markdown]
# ### Bloco de Código 13.2 — Dicionário de hiperparâmetros para XGBoost
#
# *Página 237 do PDF · Revisão: confirmado*
#
# **Nota:** código integral preservado no notebook original; abaixo, adaptação de validação quando necessária.

# %% [code] célula 6
from xgboost import XGBClassifier
seed=42
# XGBoost
xgb = XGBClassifier(objective='binary:logistic', random_state=seed)

xgb_dict = {
    'n_estimators': 100,     # Número de rodadas de boosting (árvores)
    'learning_rate': 0.1,    # Taxa de aprendizado
    'max_depth': 6,          # Profundidade máxima de cada árvore
    'subsample': 0.8,        # Porcentagem de amostras usadas para treinar cada árvore
    'colsample_bytree': 0.8, # Porcentagem de colunas (features) usadas por árvore
    'gamma': 0.1             # Valores comuns entre 0 e 5. Quanto maior, mais conservador o modelo
}

print(xgb_dict)

# %% [markdown]
# ### Bloco de Código 13.3 — Grid Search com XGBoost e scikit-learn
#
# *Página 239 do PDF · Revisão: confirmado*
#
# **Nota:** código integral preservado no notebook original; abaixo, adaptação de validação quando necessária.

# %% [markdown]
# **Adaptação:** a grade de 192 combinações × 5 folds foi reduzida para 4 combinações × 3 folds; número de árvores reduzido e `n_jobs=1` para teste rápido. Nenhuma informação do conjunto de teste entra na seleção.

# %% [code] célula 9
xgb_grid={'n_estimators':[12,20],'max_depth':[2,3],'learning_rate':[0.1],'subsample':[0.8]}
xgb_gs=GridSearchCV(XGBClassifier(objective='binary:logistic',random_state=seed,n_jobs=1,tree_method='hist'),xgb_grid,scoring='roc_auc',cv=cv,n_jobs=1)
xgb_gs.fit(X_train,y_train)
print('Grid Search:',xgb_gs.best_params_,'AUC CV:',round(xgb_gs.best_score_,3))
assert len(xgb_gs.cv_results_['params'])==4

# %% [markdown]
# ### Bloco de Código 13.4 — Random Search com XGBoost e scikit-learn
#
# *Página 240 do PDF · Revisão: confirmado*
#
# **Nota:** código integral preservado no notebook original; abaixo, adaptação de validação quando necessária.

# %% [markdown]
# **Adaptação:** `n_iter=4` (em vez de 20), três folds, árvores menores e execução serial. Os intervalos principais continuam demonstrados.

# %% [code] célula 12
xgb_dict={'n_estimators':[12,20,30],'max_depth':[2,3,4], 'learning_rate':np.logspace(-2,-0.7,3),'subsample':[.7,1.0], 'colsample_bytree':[.7,1.0],'gamma':[0,.2],'reg_alpha':[.001,.1], 'reg_lambda':[1,2]}
xgb_rs=RandomizedSearchCV(XGBClassifier(objective='binary:logistic',random_state=seed,n_jobs=1,tree_method='hist'),xgb_dict,n_iter=4,scoring='roc_auc',cv=cv,random_state=seed,n_jobs=1)
xgb_rs.fit(X_train,y_train)
print('Random Search:',xgb_rs.best_params_,'AUC CV:',round(xgb_rs.best_score_,3))
assert len(xgb_rs.cv_results_['params'])==4

# %% [markdown]
# ### Bloco de Código 13.5 — Otimização bayesiana com Optuna e XGBoost
#
# *Página 244 do PDF · Revisão: confirmado*
#
# **Nota:** código integral preservado no notebook original; abaixo, adaptação de validação quando necessária.

# %% [markdown]
# **Adaptação:** `n_trials=4` (em vez de 50), três folds, estimadores menores e semente para reprodutibilidade. A função objetivo mantém a otimização da AUC por validação cruzada.

# %% [code] célula 15
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
def objective(trial):
    params={'n_estimators':trial.suggest_int('n_estimators',12,30,step=6), 'max_depth':trial.suggest_int('max_depth',2,4), 'learning_rate':trial.suggest_float('learning_rate',.02,.2,log=True), 'subsample':trial.suggest_float('subsample',.6,1.), 'colsample_bytree':trial.suggest_float('colsample_bytree',.6,1.), 'gamma':trial.suggest_float('gamma',0,.3), 'reg_alpha':trial.suggest_float('reg_alpha',1e-3,1.,log=True), 'reg_lambda':trial.suggest_float('reg_lambda',1.,2.)}
    model=XGBClassifier(objective='binary:logistic',random_state=seed,n_jobs=1,tree_method='hist',**params)
    return cross_val_score(model,X_train,y_train,cv=cv,scoring='roc_auc',n_jobs=1).mean()
study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=seed))
study.optimize(objective,n_trials=4)
print('Optuna:',study.best_params,'AUC CV:',round(study.best_value,3))
assert len(study.trials)==4
xgb=XGBClassifier(objective='binary:logistic',random_state=seed,n_jobs=1,tree_method='hist',**study.best_params)
xgb.fit(X_train,y_train)
print('AUC no teste reservado:',round(roc_auc_score(y_test,xgb.predict_proba(X_test)[:,1]),3))
