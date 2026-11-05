# -*- coding: utf-8 -*-
# Capítulo 12: exportação automática das células do notebook executável.
# Executar na raiz do projeto: python python/capitulo_12.py
# Células que dependem de contexto de notebook, serviços ou bibliotecas opcionais podem exigir ajustes.

# %% [markdown]
# # Capítulo 12 — Treino, teste e validação (versão executável)
#
# Dados **sintéticos**, não clínicos. Blocos do livro preservados em notebook original; neste notebook, substituições são explicitadas. Métricas são demonstrativas, não evidência clínica. A validação temporal e por grupos deve ser escolhida conforme o desenho real do estudo.

# %% [code] célula 2
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
Xraw, yraw = make_classification(n_samples=160,n_features=5,n_informative=4,n_redundant=0,weights=[.65,.35],random_state=42)
rng=np.random.default_rng(42)
df=pd.DataFrame(Xraw,columns=[f'feature_{i}' for i in range(5)])
df['obito']=yraw
df['patient_id']=np.repeat(np.arange(40),4)
df['date']=pd.date_range('2022-01-01',periods=160,freq='D')
print('Base sintética:',df.shape)

# %% [markdown]
# ### Bloco de Código 12.1 — Separação entre treino e teste com scikit-learn
#
# *Página 219 do PDF · Revisão: confirmado*
#
# **Exemplo do livro.** A execução abaixo contém adaptação explicitada.

# %% [markdown]
# **Adaptação:** excluir identificador e data dos preditores, pois não são atributos numéricos destinados ao Random Forest neste exemplo; a divisão é estratificada.

# %% [code] célula 5
X=df.drop(columns=['obito','patient_id','date']); y=df.obito.copy()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
assert set(X_train.index).isdisjoint(set(X_test.index))
print('Treino/teste:',len(X_train),len(X_test))

# %% [markdown]
# ### Bloco de Código 12.2 — Separação temporal de dados com pandas
#
# *Página 221 do PDF · Revisão: confirmado*
#
# **Exemplo do livro.** Executado conforme a transcrição revisada.

# %% [code] célula 7
import pandas as pd

# Supondo um dataframe com coluna de data
# df: patient_id | date | features... | target

df['date'] = pd.to_datetime(df['date'])

# Ordena por data
df = df.sort_values('date')

# Define corte temporal (ex.: 7 anos treino, 3 anos teste)
cutoff_date = df['date'].quantile(0.7)  # ou definir manualmente

train_df = df[df['date'] <= cutoff_date]
test_df = df[df['date'] > cutoff_date]

print("Treino:", train_df['date'].min(), "-", train_df['date'].max())
print("Teste:", test_df['date'].min(), "-", test_df['date'].max())

# %% [markdown]
# ### Bloco de Código 12.3 — Separação em bloco por paciente com ­GroupShuffleSplit
#
# *Página 222 do PDF · Revisão: confirmado*
#
# **Exemplo do livro.** Executado conforme a transcrição revisada.

# %% [code] célula 9
from sklearn.model_selection import GroupShuffleSplit

# Define grupos (pacientes)
groups = df['patient_id']

gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=groups))

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]

# Verificação: não deve haver interseção
intersection = set(train_df['patient_id']).intersection(set(test_df['patient_id']))
print("Pacientes em comum:", len(intersection))

# %% [code] célula 10
assert len(intersection)==0
print('Sem vazamento entre pacientes no GroupShuffleSplit')

# %% [markdown]
# ### Bloco de Código 12.4 — Validação cruzada k-fold com scikit-learn
#
# *Página 226 do PDF · Revisão: confirmado*
#
# **Exemplo do livro.** A execução abaixo contém adaptação explicitada.

# %% [code] célula 12
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Define o modelo
model = RandomForestClassifier(random_state=42)

# Define o esquema de k-fold (k=5)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Avalia o modelo em cada fold
scores = cross_val_score(model, X_train, y_train,
                          cv=kf, scoring='roc_auc')

print('AUC média KFold:',round(float(np.nanmean(scores)),3))

# %% [markdown]
# ### Bloco de Código 12.5 — Leave-One-Out Cross-Validation com scikit-learn
#
# *Página 227 do PDF · Revisão: confirmado*
#
# **Exemplo do livro.** A execução abaixo contém adaptação explicitada.

# %% [markdown]
# **Adaptação computacional:** Leave-One-Out em uma amostra pequena de 24 observações; o original usa o treino completo e pode ser demorado.

# %% [code] célula 15
from sklearn.model_selection import LeaveOneOut
X_loo=X_train.iloc[:24];y_loo=y_train.iloc[:24]
loo=LeaveOneOut();model=RandomForestClassifier(n_estimators=20,random_state=42)
scores=cross_val_score(model,X_loo,y_loo,cv=loo,scoring='accuracy',n_jobs=1)
print('Iterações LOO:',len(scores),'acurácia:',round(scores.mean(),3))

# %% [markdown]
# ### Bloco de Código 12.6 — Leave-P-Out Cross-Validation com scikit-learn
#
# *Página 227 do PDF · Revisão: confirmado*
#
# **Exemplo do livro.** A execução abaixo contém adaptação explicitada.

# %% [markdown]
# **Correção necessária:** o código revisado termina com parêntese ausente no `print`. Além disso, ROC AUC não é definida quando os dois casos de teste pertencem à mesma classe. Para demonstrar Leave-P-Out, usa-se acurácia em 12 observações (66 combinações), com Random Forest reduzida.

# %% [code] célula 18
from sklearn.model_selection import LeavePOut
X_lpo=X_train.iloc[:12];y_lpo=y_train.iloc[:12]
lpo=LeavePOut(p=2);model=RandomForestClassifier(n_estimators=15,random_state=42)
scores=cross_val_score(model,X_lpo,y_lpo,cv=lpo,scoring='accuracy',n_jobs=1)
print('Iterações LPO:',len(scores),'acurácia:',round(scores.mean(),3))
assert len(scores)==66

# %% [markdown]
# ### Bloco de Código 12.7 — Monte Carlo Cross-Validation com scikit-learn
#
# *Página 228 do PDF · Revisão: confirmado*
#
# **Exemplo do livro.** A execução abaixo contém adaptação explicitada.

# %% [markdown]
# **Adaptação computacional:** 20 partições em vez das 100 do livro para um teste rápido; manter 100 em uma execução definitiva.

# %% [code] célula 21
from sklearn.model_selection import ShuffleSplit
mc=ShuffleSplit(n_splits=20,test_size=.2,random_state=42)
model=RandomForestClassifier(n_estimators=25,random_state=42)
scores=cross_val_score(model,X_train,y_train,cv=mc,scoring='roc_auc')
print('AUC Monte Carlo:',round(scores.mean(),3),'±',round(scores.std(),3))

# %% [markdown]
# ### Bloco de Código 12.8 — K-fold Estratificado com scikit-learn
#
# *Página 229 do PDF · Revisão: confirmado*
#
# **Exemplo do livro.** A execução abaixo contém adaptação explicitada.

# %% [code] célula 23
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X_train, y_train,
                         cv=skf, scoring='roc_auc')

print(f"AUC media: {scores.mean():.3f} (+/- {scores.std():.3f})")

# %% [markdown]
# ### Bloco de Código 12.9 — K-fold em Grupo com scikit-learn
#
# *Página 229 do PDF · Revisão: confirmado*
#
# **Exemplo do livro.** A execução abaixo contém adaptação explicitada.

# %% [markdown]
# **Correção:** `groups=df.patient_id` tem 160 linhas, mas `X_train` tem 128. Os grupos precisam ser alinhados aos índices do treino. Para CV agrupada, os pacientes devem ser disjuntos entre treino e validação de cada fold.

# %% [code] célula 26
from sklearn.model_selection import GroupKFold
model=RandomForestClassifier(n_estimators=30,random_state=42)
groups_train=df.loc[X_train.index,'patient_id']
gkf=GroupKFold(n_splits=5)
for ti,vi in gkf.split(X_train,y_train,groups_train):
    assert set(groups_train.iloc[ti]).isdisjoint(set(groups_train.iloc[vi]))
scores=cross_val_score(model,X_train,y_train,cv=gkf,groups=groups_train,scoring='roc_auc')
print('AUC GroupKFold:',round(scores.mean(),3))

# %% [markdown]
# ### Bloco de Código 12.10 — Validação cruzada temporal com scikit-learn
#
# *Página 230 do PDF · Revisão: confirmado*
#
# **Exemplo do livro.** A execução abaixo contém adaptação explicitada.

# %% [markdown]
# **Correção metodológica:** `train_test_split` aleatório não preserva ordem cronológica. Para TimeSeriesSplit, ordenar explicitamente pela data e criar um conjunto de treino temporal. AUC pode ser indefinida se um fold tiver apenas uma classe.

# %% [code] célula 29
from sklearn.model_selection import TimeSeriesSplit
ordem=df.sort_values('date').index
corte=int(len(ordem)*.8)
X_temporal=X.loc[ordem[:corte]];y_temporal=y.loc[ordem[:corte]]
tss=TimeSeriesSplit(n_splits=5)
model=RandomForestClassifier(n_estimators=30,random_state=42)
scores=cross_val_score(model,X_temporal,y_temporal,cv=tss,scoring='accuracy')
print('Acurácia TimeSeriesSplit:',round(scores.mean(),3))
for ti,vi in tss.split(X_temporal):assert ti.max()<vi.min()
