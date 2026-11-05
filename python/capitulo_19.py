# -*- coding: utf-8 -*-
# Capítulo 19: exportação automática das células do notebook executável.
# Executar na raiz do projeto: python python/capitulo_19.py
# Células que dependem de contexto de notebook, serviços ou bibliotecas opcionais podem exigir ajustes.

# %% [markdown]
# # Capítulo 19 — multicalibração
#
# Validação técnica com dados **sintéticos**. O bloco original foi mantido sem alterações. O exemplo não implementa multicalibração plena.
#
# **Atenção:** o código original usa `y_true` para corrigir as previsões. Nunca aplicar essa correção sobre os rótulos do conjunto de teste usado na avaliação.

# %% [markdown]
# ### Bloco de Código 19.1 — Exemplo didático de correção iterativa do erro médio
#
# *Página 412 do PDF · Revisão: confirmado*

# %% [code] célula 3
def multicalibration_postprocess(y_pred, X_test_df, y_true,
                                n_iterations=5, learning_rate=0.1):
    """
    Exemplo didático de ajuste iterativo por grupo.
    Não implementa multicalibração plena nem PMC.
    Corrigi apenas o erro médio absoluto do pior subgrupo marginal.
    """
    y_pred_mc = y_pred.copy()
    
    subgroups = {
        'age_<50': X_test_df['age_group'] == '<50',
        'age_50-60': X_test_df['age_group'] == '50-60',
        'age_60+': X_test_df['age_group'] == '>60',
        'female': X_test_df['sex_label'] == 'Feminino',
        'male': X_test_df['sex_label'] == 'Masculino',
        'low_risk': X_test_df['risk_group'] == 'Baixo',
        'medium_risk': X_test_df['risk_group'] == 'Médio',
        'high_risk': X_test_df['risk_group'] == 'Alto',
    }
    
    for _ in range(n_iterations):
        max_error = 0
        worst_group = None
        
        for group_name, mask in subgroups.items():
            if mask.sum() > 0:
                error = abs(y_pred_mc[mask].mean() - y_true[mask].mean())
                if error > max_error:
                    max_error = error
                    worst_group = (group_name, mask)
        
        if worst_group is None or max_error <= 0.01:
            break  # Convergência antecipada
        group_name, mask = worst_group
        
        observed_rate = y_true[mask].mean()
        current_rate = y_pred_mc[mask].mean()
        adjustment = learning_rate * (observed_rate - current_rate)
        y_pred_mc[mask] = (y_pred_mc[mask] + adjustment).clip(0, 1)
    
    return y_pred_mc

# %% [markdown]
# ## Preparação de dados sintéticos e separação de calibração/teste
#
# Os rótulos de calibração são usados para aprender ajustes. Os rótulos de teste são reservados exclusivamente para avaliação.

# %% [code] célula 5
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
rng = np.random.default_rng(19)
n = 1800
X = pd.DataFrame({
    'age_group': rng.choice(['<50', '50-60', '>60'], n, p=[.4,.35,.25]),
    'sex_label': rng.choice(['Feminino','Masculino'], n),
    'risk_group': rng.choice(['Baixo','Médio','Alto'], n, p=[.45,.35,.20])
})
risk_effect = X['risk_group'].map({'Baixo':-.15,'Médio':.05,'Alto':.24}).to_numpy()
age_effect = X['age_group'].map({'<50':-.07,'50-60':.02,'>60':.09}).to_numpy()
p_true = np.clip(.27 + risk_effect + age_effect + (X['sex_label']=='Masculino').to_numpy()*.03, .02, .95)
y = pd.Series(rng.binomial(1,p_true),index=X.index)
# Probabilidades deliberadamente mal calibradas para demonstrar a correção
pred = pd.Series(np.clip(.17 + .65*p_true, .01, .99),index=X.index)
idx_cal,idx_test = train_test_split(np.arange(n),test_size=.4,random_state=19,stratify=y)
X_cal, X_test = X.iloc[idx_cal].reset_index(drop=True), X.iloc[idx_test].reset_index(drop=True)
y_cal, y_test = y.iloc[idx_cal].reset_index(drop=True), y.iloc[idx_test].reset_index(drop=True)
p_cal, p_test = pred.iloc[idx_cal].reset_index(drop=True), pred.iloc[idx_test].reset_index(drop=True)
print('Calibração:',len(idx_cal),'Teste:',len(idx_test))

# %% [markdown]
# ## Teste da função original no conjunto de calibração
#
# A função original ajusta previsões **somente nos exemplos cujos rótulos foram fornecidos**. Ela não aprende nem devolve uma transformação transferível para novas observações.

# %% [code] célula 7
p_cal_adjusted = multicalibration_postprocess(p_cal, X_cal, y_cal, n_iterations=30, learning_rate=.5)
assert len(p_cal_adjusted)==len(p_cal)
assert p_cal_adjusted.between(0,1).all()
assert np.isfinite(p_cal_adjusted).all()
print('Brier calibração antes:',round(brier_score_loss(y_cal,p_cal),4))
print('Brier calibração depois:',round(brier_score_loss(y_cal,p_cal_adjusted),4))

# %% [markdown]
# ## Adaptação independente: aprender correções na calibração e aplicá-las no teste
#
# Esta é uma **extensão didática**, não o algoritmo do livro nem uma implementação formal de multicalibração. A regra aprende deslocamentos por subgrupo apenas na calibração e os aplica, na mesma ordem, ao teste sem consultar seus rótulos.

# %% [code] célula 9
def learn_group_adjustments(p, X, y, n_iterations=30, learning_rate=.5, tol=.01):
    p = np.asarray(p,dtype=float).copy()
    y = np.asarray(y,dtype=float)
    groups = {
        'age_<50': X['age_group'].to_numpy() == '<50',
        'age_50-60': X['age_group'].to_numpy() == '50-60',
        'age_60+': X['age_group'].to_numpy() == '>60',
        'female': X['sex_label'].to_numpy() == 'Feminino',
        'male': X['sex_label'].to_numpy() == 'Masculino',
        'low_risk': X['risk_group'].to_numpy() == 'Baixo',
        'medium_risk': X['risk_group'].to_numpy() == 'Médio',
        'high_risk': X['risk_group'].to_numpy() == 'Alto',
    }
    steps=[]
    for _ in range(n_iterations):
        candidates=[(abs(p[m].mean()-y[m].mean()),name,m) for name,m in groups.items() if m.any()]
        if not candidates: break
        error,name,mask=max(candidates,key=lambda t:t[0])
        if error<=tol:break
        adjustment=float(learning_rate*(y[mask].mean()-p[mask].mean()))
        p[mask]=np.clip(p[mask]+adjustment,0,1)
        steps.append((name,adjustment))
    return steps,p

def apply_group_adjustments(p,X,steps):
    p=np.asarray(p,dtype=float).copy()
    masks={
        'age_<50': X['age_group'].to_numpy() == '<50',
        'age_50-60': X['age_group'].to_numpy() == '50-60',
        'age_60+': X['age_group'].to_numpy() == '>60',
        'female': X['sex_label'].to_numpy() == 'Feminino',
        'male': X['sex_label'].to_numpy() == 'Masculino',
        'low_risk': X['risk_group'].to_numpy() == 'Baixo',
        'medium_risk': X['risk_group'].to_numpy() == 'Médio',
        'high_risk': X['risk_group'].to_numpy() == 'Alto',
    }
    for name,delta in steps:
        p[masks[name]]=np.clip(p[masks[name]]+delta,0,1)
    return p
steps,p_cal_extended=learn_group_adjustments(p_cal,X_cal,y_cal)
p_test_adjusted=apply_group_adjustments(p_test,X_test,steps)
assert len(p_test_adjusted)==len(y_test)
assert np.all((p_test_adjusted>=0)&(p_test_adjusted<=1))
print('Ajustes aprendidos:',len(steps))
print('Brier teste antes:',round(brier_score_loss(y_test,p_test),4))
print('Brier teste depois:',round(brier_score_loss(y_test,p_test_adjusted),4))

# %% [markdown]
# ## Auditoria por subgrupo no teste
#
# A tabela permite inspecionar a diferença entre frequência observada e probabilidade prevista por grupo. O ajuste pode melhorar alguns grupos e piorar outros; não há garantia de melhora geral.

# %% [code] célula 11
rows=[]
for col in ['age_group','sex_label','risk_group']:
    for value in sorted(X_test[col].unique()):
        mask=(X_test[col]==value).to_numpy()
        rows.append({'atributo':col,'grupo':value,'n':int(mask.sum()),
                     'taxa_observada':round(float(y_test.to_numpy()[mask].mean()),3),
                     'erro_abs_antes':round(abs(float(p_test.to_numpy()[mask].mean()-y_test.to_numpy()[mask].mean())),3),
                     'erro_abs_depois':round(abs(float(p_test_adjusted[mask].mean()-y_test.to_numpy()[mask].mean())),3)})
resultados=pd.DataFrame(rows)
print(resultados.to_string(index=False))
print('Maior erro por grupo antes/depois:',resultados['erro_abs_antes'].max(),resultados['erro_abs_depois'].max())

# %% [markdown]
# ## Conclusão
#
# - O bloco 19.1 foi executado com dados sintéticos.
# - O exemplo original corrige o próprio conjunto rotulado; **não fornece inferência independente**.
# - A extensão separa calibração e teste e aplica ajustes aprendidos sem acesso aos rótulos do teste.
# - A comparação é exclusivamente ilustrativa e não comprova multicalibração formal nem utilidade clínica.
