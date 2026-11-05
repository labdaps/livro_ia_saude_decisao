# -*- coding: utf-8 -*-
# Reteste com amostra SINAN: capitulo_11_real.ipynb
# Exportação automática; executar na raiz do projeto.

# %% [code] célula 4
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
fonte = pd.read_csv(r"/mnt/data/dados_tuberculose_v2_10000.csv", low_memory=False)
fonte.to_csv('dados_tuberculose.csv',index=False)
print('Base REAL:',fonte.shape)

# %% [code] célula 6
import pandas as pd
import numpy as np
df = pd.read_csv('dados_tuberculose.csv',low_memory=False)
sudeste=[31,32,33,35]
# O filtro original 2022+ gera apenas 1 registro nesta amostra.
original=df.loc[df.SG_UF.isin(sudeste)&(df.IDADE_ANOS>=60)&(df.NU_ANO>=2022)]
print('Filtro original:',len(original))
# Reteste exploratório: relaxamento SOMENTE do filtro temporal.
df_filtered=df.loc[df.SG_UF.isin(sudeste)&(df.IDADE_ANOS>=60)].copy()
print('Filtro alternativo:',df_filtered.shape)

# %% [code] célula 7
assert len(df_filtered)>0
assert set(df_filtered.SG_UF.unique()).issubset({31,32,33,35})
assert df_filtered.IDADE_ANOS.min()>=60
print('Filtro alternativo validado com dados reais')

# %% [code] célula 9
# Dimensões da base filtrada
print(df_filtered.shape)

# Tipos de variáveis e presença de ausentes
df_filtered.info()

# Visualização das primeiras linhas
df_filtered.head()

# %% [code] célula 11
# Verificação se existem e quantos são os valores duplicados
df_filtered.duplicated().sum()

# Verificação de valores duplicados
df_filtered.loc[df_filtered.duplicated()]

# Remoção dos Valores duplicados
df_filtered.drop_duplicates(inplace=True)

print("Observações após remoção:", df_filtered.shape[0])

# %% [code] célula 13
# Variáveis selecionadas com base em conhecimento clínico
variaveis_independentes = [
    "IDADE_ANOS", "CS_SEXO", "CS_RACA", "CS_ESCOL_N",
    "TRATSUP_AT", "TRATAMENTO", "AGRAVDROGA", "POP_RUA",
    "POP_LIBER", "BACILOSC_E", "AGRAVALCOO", "FORMA",
    "CS_GESTANT", "AGRAVTABAC"
]

variavel_desfecho = ["SITUA_ENCE"]

df_filtered = df_filtered[variaveis_independentes + variavel_desfecho]

# Nova dimensão da base:
print("Shape DF filtrado:", df_filtered.shape)

# Verificação das colunas da base
df_filtered.columns

# %% [code] célula 15
# Correção de tipos: categoricas
categoricas = [
    "CS_SEXO", "CS_RACA", "CS_ESCOL_N", "TRATSUP_AT",
    "TRATAMENTO", "AGRAVDROGA", "POP_RUA", "POP_LIBER",
    "BACILOSC_E", "AGRAVALCOO", "FORMA", "CS_GESTANT",
    "AGRAVTABAC", "SITUA_ENCE"
]

df_filtered[categoricas] = df_filtered[categoricas].astype("category")

# Correção de tipos: numérica contínua
df_filtered["IDADE_ANOS"] = df_filtered["IDADE_ANOS"].astype(float)

# Verificação
df_filtered.info()

# %% [code] célula 17
# Percentual de ausentes por variável, em ordem decrescente

percentual_ausentes = (
    df_filtered.isna().mean() * 100
).sort_values(ascending=False)

percentual_ausentes = percentual_ausentes[percentual_ausentes > 0]

print(percentual_ausentes)

# %% [code] célula 19
# Estatística descritiva da variável numérica
print(df_filtered[['IDADE_ANOS']].describe().T)

# Distribuição das variáveis categóricas
print(df_filtered[['CS_SEXO', 'CS_RACA']].describe().T)

# %% [code] célula 22
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8,4))
sns.boxplot(data=df_filtered,y='IDADE_ANOS',color='steelblue')
plt.title('Distribuição da idade (anos) — dados sintéticos')
plt.show()
plt.figure(figsize=(8,4))
sns.countplot(data=df_filtered,x='CS_RACA',color='steelblue')
plt.title('Distribuição por raça/cor — dados sintéticos')
plt.show()

# %% [code] célula 24
# Exemplo de remoção de idades muito avançadas
df_filtered = df_filtered.loc[df_filtered.IDADE_ANOS <= 105]

# Verificação da nova dimensão da base
df_filtered.shape

# %% [code] célula 27
X=df_filtered.drop(columns=['SITUA_ENCE']).copy()
y=df_filtered['SITUA_ENCE'].copy()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
X_train=X_train.copy();X_test=X_test.copy()
print('Treino/teste:',X_train.shape,X_test.shape)
assert X_train.index.intersection(X_test.index).empty

# %% [code] célula 28
# Cálculo da correlação entre variáveis numéricas do conjunto de treino
X_train.corr(numeric_only=True)

# %% [code] célula 30
# Verificação de valores ausentes na base:
print("Quantidade de missings no treino: ", X_train.isna().sum().sum())
print("Quantidade de missings no teste: ", X_test.isna().sum().sum())

# %% [code] célula 33
X_train.columns=X_train.columns.str.lower();X_test.columns=X_test.columns.str.lower()
assert X_train.columns.equals(X_test.columns)
print('Colunas:',list(X_train.columns))

# %% [code] célula 34
# Lista de variáveis numéricas
lista_variaveis_numericas = ["idade_anos"]

# Lista de variáveis categóricas
lista_variaveis_categóricas = ["sg_uf", "cs_sexo", "cs_raca", "cs_escol_n",
                               "tratsup_at", "tratamento", "agravdroga",
                               "pop_rua", "pop_liber", "bacilosc_e",
                               "agravalcoo", "forma", "cs_gestant", "agravtabac"]

# %% [code] célula 35
lista_variaveis_categóricas=[c for c in lista_variaveis_categóricas if c in X_train.columns and c in X_test.columns]
assert 'sg_uf' not in lista_variaveis_categóricas
print('Variáveis categóricas presentes:',lista_variaveis_categóricas)

# %% [code] célula 37
# Cálculo das medianas apenas no conjunto de treino
medianas_treino = X_train[lista_variaveis_numericas].median()

# Aplicação das medianas do treino para imputacao em treino e teste
X_train[lista_variaveis_numericas] = X_train[lista_variaveis_numericas].fillna(medianas_treino)
X_test[lista_variaveis_numericas] = X_test[lista_variaveis_numericas].fillna(medianas_treino)

# %% [code] célula 40
for coluna in lista_variaveis_categóricas:
    for frame in (X_train,X_test):
        frame[coluna]=frame[coluna].cat.add_categories([9]) if 9 not in frame[coluna].cat.categories else frame[coluna]
print('Categoria 9 habilitada nas variáveis categóricas.')

# %% [code] célula 41
# Preenchimento das categorias ausentes com o código 9 (categoria sintética de "não informado")
X_train.loc[:, lista_variaveis_categóricas] = X_train.loc[:, lista_variaveis_categóricas].fillna(9)
X_test.loc[:, lista_variaveis_categóricas] = X_test.loc[:, lista_variaveis_categóricas].fillna(9)

# %% [code] célula 44
# Obtenção da moda nas bases de treino
modas_treino = X_train[lista_variaveis_categóricas].mode().iloc[0]

# Aplicação da moda do treino nas base de treino e de teste
X_train[lista_variaveis_categóricas] = X_train[lista_variaveis_categóricas].fillna(modas_treino)
X_test[lista_variaveis_categóricas] = X_test[lista_variaveis_categóricas].fillna(modas_treino)

# %% [code] célula 46
from sklearn.preprocessing import StandardScaler

# Criação do Scaler
standard_scaler = StandardScaler()

# Treino do Scaler na base de treino
standard_scaler.fit(X_train[lista_variaveis_numericas])

# Aplicação do Scaler na base de treino e de teste
X_train[lista_variaveis_numericas] = standard_scaler.transform(X_train[
    lista_variaveis_numericas])
X_test[lista_variaveis_numericas] = standard_scaler.transform(X_test[
    lista_variaveis_numericas])

# Verificar Média e Desvio Padrão
X_train.describe().round(2)

# %% [code] célula 49
from sklearn.preprocessing import OneHotEncoder

# Criação do one hot encoder
one_hot_encoder = OneHotEncoder(drop="if_binary", sparse_output=False,
                              handle_unknown="ignore")

# Treino do one hot encoding na base de treino
one_hot_encoder.fit(X_train[lista_variaveis_categóricas])

# Aplicação na base de treino e de teste
X_train_encoded = pd.DataFrame(one_hot_encoder.transform(X_train[
    lista_variaveis_categóricas]),
                                columns=one_hot_encoder.get_feature_names_out(),
                                index=X_train.index)

X_test_encoded = pd.DataFrame(one_hot_encoder.transform(X_test[
    lista_variaveis_categóricas]),
                                columns=one_hot_encoder.get_feature_names_out(),
                                index=X_test.index)

# %% [code] célula 50
assert X_train_encoded.shape[1]==X_test_encoded.shape[1]
assert X_train_encoded.index.equals(X_train.index)
print('One-hot: mesmas colunas e alinhamento preservado')

# %% [code] célula 53
from sklearn.preprocessing import OrdinalEncoder
escolaridade_treino=pd.DataFrame({'escolaridade':['sem instrução','ensino fundamental','ensino médio','ensino superior']})
escolaridade_teste=pd.DataFrame({'escolaridade':['ensino médio','não informado']})
ordem_escolaridade=['sem instrução','ensino fundamental','ensino médio','ensino superior']
ordinal_encoder_demo=OrdinalEncoder(categories=[ordem_escolaridade],handle_unknown='use_encoded_value',unknown_value=-1)
encoded_treino=ordinal_encoder_demo.fit_transform(escolaridade_treino[['escolaridade']])
encoded_teste=ordinal_encoder_demo.transform(escolaridade_teste[['escolaridade']])
print('Ordinal treino:',encoded_treino.ravel(),'teste:',encoded_teste.ravel())
assert encoded_teste[1,0]==-1

# %% [code] célula 56
# Demonstração isolada de target encoding; nunca usar rótulos do teste para estimar médias.
municipios_treino=pd.DataFrame({'id_municip':[101,101,102,103,103,103], 'obito':[0,1,0,1,0,0]})
municipios_teste=pd.DataFrame({'id_municip':[101,102,999]})
medias=municipios_treino.groupby('id_municip')['obito'].mean()
media_global=municipios_treino['obito'].mean()
municipios_teste['id_municip_te']=municipios_teste['id_municip'].map(medias).fillna(media_global)
print(municipios_teste)
assert np.isfinite(municipios_teste['id_municip_te']).all()
print('Cautela: médias calculadas diretamente no treino podem gerar vazamento dentro do treino; para modelagem recomenda-se codificação out-of-fold.')

# %% [code] célula 58
# Remoção das variáveis categóricas originais
X_train.drop(columns=lista_variaveis_categóricas, inplace=True)
X_test.drop(columns=lista_variaveis_categóricas, inplace=True)

# União das variáveis numéricas com as colunas geradas pelo encoding
X_train = pd.concat([X_train, X_train_encoded], axis=1)
X_test = pd.concat([X_test, X_test_encoded], axis=1)

# %% [code] célula 60
# Reunindo preditores e desfecho na base de treino
df_train = pd.concat([X_train, y_train], axis=1)
df_train.to_csv("dados_treino_preprocessados.csv", index=False)

# Reunindo preditores e desfecho na base de teste
df_test = pd.concat([X_test, y_test], axis=1)
df_test.to_csv("dados_teste_preprocessados.csv", index=False)

# %% [code] célula 61
assert X_train.columns.equals(X_test.columns)
assert len(df_train)==len(y_train) and len(df_test)==len(y_test)
assert not df_train.isna().any().any() and not df_test.isna().any().any()
print('Bases sintéticas exportadas, colunas compatíveis e sem ausentes')
