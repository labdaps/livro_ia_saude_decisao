# -*- coding: utf-8 -*-
# Reteste com amostra SINAN: capitulo_21_real.ipynb
# Exportação automática; executar na raiz do projeto.

# %% [code] célula 3
import importlib.util
for p in ["river", "deep_river", "torch"]:
    print(p, "disponível" if importlib.util.find_spec(p) else "INDISPONÍVEL")

# %% [code] célula 10
import numpy as np,pandas as pd
base=pd.read_csv(r"/mnt/data/dados_tuberculose_v2_10000.csv",low_memory=False)
print('Base REAL:',base.shape)

# %% [code] célula 12
df=base.loc[base['SITUA_ENCE'].isin([1,2]), ['SG_UF_NOT','FORMA','RAIOX_TORA','IDADE_ANOS','NU_ANO','DT_NOTIFIC','SITUA_ENCE']].copy()
features=['SG_UF_NOT','FORMA','RAIOX_TORA','IDADE_ANOS','NU_ANO']
for c in features: df[c]=pd.to_numeric(df[c],errors='coerce')
df=df.dropna(subset=features+['SITUA_ENCE']).copy()
df['alvo']=(df['SITUA_ENCE']==2).astype(int)
df['DT_NOTIFIC']=pd.to_datetime(df['DT_NOTIFIC'],errors='coerce')
df=df.sort_values(['DT_NOTIFIC','NU_ANO'],na_position='last').reset_index(drop=True)
X=df[features];y=df['alvo']
print('Cura:',sum(y==0),'Abandono:',sum(y==1),'Total:',len(y))

# %% [code] célula 13
assert len(X)>1000 and set(y.unique())=={0,1}
print('Validação preparação real:',len(X))

# %% [code] célula 16
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
    if (i+1)%1000==0: historico.append({'exemplos':i+1,'acuracia':corretos/max(1,avaliados)})
print('Acurácia prequential demonstrativa:',round(corretos/max(1,avaliados),3),'em',avaliados,'predições')
assert avaliados==len(X)-1

# %% [code] célula 19
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
Path('figuras').mkdir(exist_ok=True)
hist=pd.DataFrame(historico)
fig,ax=plt.subplots(figsize=(8,4))
ax.plot(hist['exemplos'],hist['acuracia'],marker='o')
ax.set(xlabel='Exemplos vistos',ylabel='Acurácia prequential acumulada',title='SGDClassifier incremental — demonstração com amostra real')
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig('figuras/evolucao_acuracia.png',dpi=130)
plt.show()
assert Path('figuras/evolucao_acuracia.png').exists()
print('Figura salva com',len(hist),'pontos')
