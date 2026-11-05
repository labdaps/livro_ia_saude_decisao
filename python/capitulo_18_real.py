# -*- coding: utf-8 -*-
# Reteste com amostra SINAN: capitulo_18_real.ipynb
# Exportação automática; executar na raiz do projeto.

# %% [code] célula 4
import numpy as np,pandas as pd,warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix,roc_auc_score
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')
df=pd.read_csv(r"/mnt/data/dados_tuberculose_v2_10000.csv",low_memory=False)
print('Amostra REAL:',df.shape)

# %% [code] célula 6
df=df[(df['FORMA']==1) & df['SITUA_ENCE'].isin([1,3])].copy()
df['obito_tb']=(df['SITUA_ENCE']==3).astype(int)
df['Grupo_Rua']=df['POP_RUA'].map({1:'Pop. Rua',2:'Pop. Geral'})
df['Grupo_Sexo']=df['CS_SEXO'].map({'M':'Masculino','F':'Feminino'})
vars_pred=['IDADE_ANOS','CS_SEXO','CS_RACA','CS_ESCOL_N','HIV']
df_clean=df.dropna(subset=vars_pred+['obito_tb','Grupo_Rua','Grupo_Sexo']).copy()
tr,te=train_test_split(np.arange(len(df_clean)),test_size=.3,random_state=42,stratify=df_clean['obito_tb'])
enc=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
X_train=pd.DataFrame(enc.fit_transform(df_clean.iloc[tr][vars_pred]),columns=vars_pred)
X_test=pd.DataFrame(enc.transform(df_clean.iloc[te][vars_pred]),columns=vars_pred)
y_train=df_clean.iloc[tr]['obito_tb'].to_numpy();y_test=df_clean.iloc[te]['obito_tb'].to_numpy()
grua_train=df_clean.iloc[tr]['Grupo_Rua'].to_numpy();grua_test=df_clean.iloc[te]['Grupo_Rua'].to_numpy()
gsex_train=df_clean.iloc[tr]['Grupo_Sexo'].to_numpy();gsex_test=df_clean.iloc[te]['Grupo_Sexo'].to_numpy()
peso_bal=float(np.sum(y_train==0))/max(1,np.sum(y_train==1))
modelo_orig=XGBClassifier(random_state=42,eval_metric='logloss',max_depth=4,n_estimators=35,n_jobs=1,scale_pos_weight=peso_bal)
modelo_orig.fit(X_train,y_train);probs_orig=modelo_orig.predict_proba(X_test)[:,1]
def auditar(y_true,y_prob,grupos,threshold=.5):
    y_pred=(y_prob>=threshold).astype(int); rows={}
    for g in np.unique(grupos):
        m=grupos==g; yt=y_true[m]; yp=y_pred[m]
        tn,fp,fn,tp=confusion_matrix(yt,yp,labels=[0,1]).ravel()
        rows[g]={'n':int(m.sum()),'prevalencia':float(np.mean(yt)),'TPR':tp/(tp+fn) if tp+fn else np.nan,'FPR':fp/(fp+tn) if fp+tn else np.nan,'PPR':float(np.mean(yp)),'AUC':roc_auc_score(yt,y_prob[m]) if len(np.unique(yt))==2 else np.nan}
    return pd.DataFrame.from_dict(rows,orient='index')
perf_orig=auditar(y_test,probs_orig,grua_test)
print('AUC global:',round(roc_auc_score(y_test,probs_orig),3));display(perf_orig.round(3))

# %% [code] célula 8
import importlib.util
fairlearn_disponivel=importlib.util.find_spec('fairlearn') is not None
print('Fairlearn disponível:',fairlearn_disponivel)
if not fairlearn_disponivel:
    print('BLOQUEADO: instale fairlearn para validar o CorrelationRemover original.')

# %% [code] célula 9
if fairlearn_disponivel:
    from fairlearn.preprocessing import CorrelationRemover
    from sklearn.model_selection import GridSearchCV
    X_train_mitig=X_train.copy();X_train_mitig['SITUACAO_RUA_NUM']=(grua_train=='Pop. Rua').astype(float)
    X_test_mitig=X_test.copy();X_test_mitig['SITUACAO_RUA_NUM']=(grua_test=='Pop. Rua').astype(float)
    cr=CorrelationRemover(sensitive_feature_ids=['SITUACAO_RUA_NUM'],alpha=.7)
    X_train_filtered=cr.fit_transform(X_train_mitig);X_test_filtered=cr.transform(X_test_mitig)
    counts=pd.DataFrame({'g':grua_train,'y':y_train}).groupby(['g','y']).size()
    pesos_fair=np.array([len(y_train)/(4*counts[(g,y)]) for g,y in zip(grua_train,y_train)])
    gs=GridSearchCV(XGBClassifier(random_state=42,eval_metric='logloss',n_estimators=30,n_jobs=1),{'max_depth':[3],'learning_rate':[.05,.1]},cv=3,scoring='roc_auc',n_jobs=1)
    gs.fit(X_train_filtered,y_train,sample_weight=pesos_fair)
    probs_mitig=gs.best_estimator_.predict_proba(X_test_filtered)[:,1]
    perf_mitig=auditar(y_test,probs_mitig,grua_test)
    gaps=pd.DataFrame({'Original':(perf_orig.loc['Pop. Geral',['TPR','FPR','PPR']]-perf_orig.loc['Pop. Rua',['TPR','FPR','PPR']]).abs(),'Mitigado':(perf_mitig.loc['Pop. Geral',['TPR','FPR','PPR']]-perf_mitig.loc['Pop. Rua',['TPR','FPR','PPR']]).abs()})
    display(gaps.round(3))
else:
    print('Etapa 18.2 não executada: falta a dependência fairlearn.')
