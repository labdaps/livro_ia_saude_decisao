# -*- coding: utf-8 -*-
# Capítulo 11: exportação automática das células do notebook executável.
# Executar na raiz do projeto: python python/capitulo_11.py
# Células que dependem de contexto de notebook, serviços ou bibliotecas opcionais podem exigir ajustes.

# %% [markdown]
# # Capítulo 11 — Pré-processamento (versão executável)
#
# **Dados sintéticos, não SINAN-TB.** Blocos originais preservados em células identificadas. Complementos são separados; exemplos hipotéticos 11.18 e 11.19 são demonstrados em datasets próprios. Saídas não têm validade clínica.

# %% [markdown]
# ## Preparação: dados sintéticos e verificações
#
# Este complemento cria `dados_tuberculose.csv` **somente no diretório deste notebook** para viabilizar a leitura literal do bloco 11.1. A base real não foi disponibilizada.

# %% [code] célula 3
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
rng=np.random.default_rng(42)
n=360
cols=['CS_SEXO','CS_RACA','CS_ESCOL_N','TRATSUP_AT','TRATAMENTO','AGRAVDROGA','POP_RUA','POP_LIBER','BACILOSC_E','AGRAVALCOO','FORMA','CS_GESTANT','AGRAVTABAC']
df_sintetico=pd.DataFrame({'SG_UF':rng.choice([31,32,33,35,41],n,p=[.24,.20,.24,.22,.10]),'IDADE_ANOS':rng.integers(40,101,n),'NU_ANO':rng.choice([2021,2022,2023,2024],n)})
for c in cols:df_sintetico[c]=rng.choice([1,2,3,np.nan],n,p=[.42,.39,.13,.06])
df_sintetico['SITUA_ENCE']=rng.choice([1,2,3],n,p=[.75,.16,.09])
df_sintetico.to_csv('dados_tuberculose.csv',index=False)
print('Base SINTÉTICA criada:',df_sintetico.shape)

# %% [markdown]
# ### Bloco de Código 11.1 — Importação e filtragem da base SINAN-TB para a população de análise
#
# *Página 191 do PDF · Revisão: confirmado*
#
# **Código revisado do livro.**

# %% [code] célula 5
import pandas as pd
import numpy as np

# Importação da base
df = pd.read_csv("dados_tuberculose.csv", encoding="latin1", sep=",")

# UFs da região Sudeste
sudeste = [31, 32, 33, 35]

# Cópia do DataFrame Original
df_filtered = df.copy()

# Filtro: Sudeste, idosos (>= 60 anos), notificações a partir de 2022
df_filtered = df_filtered.loc[
    (df["SG_UF"].isin(sudeste)) & 
    (df["IDADE_ANOS"] >= 60) & 
    (df["NU_ANO"] >= 2022)
].copy()

print("Observações após filtragem:", df_filtered.shape[0])
print("Variáveis disponíveis:", df_filtered.shape[1])

# %% [code] célula 6
assert len(df_filtered)>0
assert set(df_filtered.SG_UF.unique()).issubset({31,32,33,35})
assert df_filtered.IDADE_ANOS.min()>=60
print('Filtro validado com dados sintéticos')

# %% [markdown]
# ### Bloco de Código 11.2 — Inspeção inicial da estrutura e dos tipos de variáveis
#
# *Página 192 do PDF · Revisão: confirmado*
#
# **Código revisado do livro.**

# %% [code] célula 8
# Dimensões da base filtrada
print(df_filtered.shape)

# Tipos de variáveis e presença de ausentes
df_filtered.info()

# Visualização das primeiras linhas
df_filtered.head()

# %% [markdown]
# ### Bloco de Código 11.3 — Verificação e remoção de registros duplicados
#
# *Página 192 do PDF · Revisão: confirmado*
#
# **Código revisado do livro.**

# %% [code] célula 10
# Verificação se existem e quantos são os valores duplicados
df_filtered.duplicated().sum()

# Verificação de valores duplicados
df_filtered.loc[df_filtered.duplicated()]

# Remoção dos Valores duplicados
df_filtered.drop_duplicates(inplace=True)

print("Observações após remoção:", df_filtered.shape[0])

# %% [markdown]
# ### Bloco de Código 11.4 — Seleção de variáveis
#
# *Página 193 do PDF · Revisão: confirmado*
#
# **Código revisado do livro.**

# %% [code] célula 12
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

# %% [markdown]
# ### Bloco de Código 11.5 — Correção de tipos das variáveis
#
# *Página 194 do PDF · Revisão: confirmado*
#
# **Código revisado do livro.**

# %% [code] célula 14
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

# %% [markdown]
# ### Bloco de Código 11.6 — Mapeamento de valores ausentes por variável
#
# *Página 194 do PDF · Revisão: confirmado*
#
# **Código revisado do livro.**

# %% [code] célula 16
# Percentual de ausentes por variável, em ordem decrescente

percentual_ausentes = (
    df_filtered.isna().mean() * 100
).sort_values(ascending=False)

percentual_ausentes = percentual_ausentes[percentual_ausentes > 0]

print(percentual_ausentes)

# %% [markdown]
# ### Bloco de Código 11.7 — Estatística descritiva por tipo de variável
#
# *Página 195 do PDF · Revisão: confirmado*
#
# **Código revisado do livro.**

# %% [code] célula 18
# Estatística descritiva da variável numérica
print(df_filtered[['IDADE_ANOS']].describe().T)

# Distribuição das variáveis categóricas
print(df_filtered[['CS_SEXO', 'CS_RACA']].describe().T)

# %% [markdown]
# ### Bloco de Código 11.8 — Visualização da distribuição de variáveis numéricas e categóricas
#
# *Página 195 do PDF · Revisão: confirmado*
#
# **Código revisado do livro.**

# %% [markdown]
# **Exemplo hipotético isolado:** o código original é reproduzido abaixo como referência, mas não executado diretamente porque depende de variáveis não presentes no fluxo principal e contém incompatibilidades de API/estrutura. A célula seguinte demonstra a técnica com dados próprios.
#
# ```python
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Variável numérica: distribuição da idade
# plt.figure(figsize=(8, 4))
# sns.boxplot(y="IDADE_ANOS", data=df_filtered)
# plt.title("Distribuição da idade (anos)")
# plt.ylabel("Idade")
# plt.show()
#
# # Variável categórica: contagem por raça/cor
# plt.figure(figsize=(8, 4))
# sns.countplot(data=df_filtered, x="CS_RACA")
# plt.title("Distribuição por raça/ cor (CS_RACA)")
# plt.xlabel("Raca/ cor")
# plt.ylabel("Contagem")
# plt.xticks(rotation=45)
# plt.show()
#
# ```

# %% [code] célula 21
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

# %% [markdown]
# ### Bloco de Código 11.9 — Tratamento de outliers
#
# *Página 196 do PDF · Revisão: confirmado*
#
# **Código revisado do livro.**

# %% [code] célula 23
# Exemplo de remoção de idades muito avançadas
df_filtered = df_filtered.loc[df_filtered.IDADE_ANOS <= 105]

# Verificação da nova dimensão da base
df_filtered.shape

# %% [markdown]
# ### Bloco de Código 11.10 — Correlação entre variáveis numéricas no conjunto de treino
#
# *Página 198 do PDF · Revisão: confirmado*
#
# **Código revisado do livro.**

# %% [markdown]
# **Complemento:** divisão estratificada antes da análise do conjunto de treino. O desfecho permanece fora dos preditores.

# %% [code] célula 26
X=df_filtered.drop(columns=['SITUA_ENCE']).copy()
y=df_filtered['SITUA_ENCE'].copy()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
X_train=X_train.copy();X_test=X_test.copy()
print('Treino/teste:',X_train.shape,X_test.shape)
assert X_train.index.intersection(X_test.index).empty

# %% [code] célula 27
# Cálculo da correlação entre variáveis numéricas do conjunto de treino
X_train.corr(numeric_only=True)

# %% [markdown]
# ### Bloco de Código 11.11 — Verificação de valores ausentes após divisão treino/ teste
#
# *Página 199 do PDF · Revisão: confirmado*
#
# **Código revisado do livro.**

# %% [code] célula 29
# Verificação de valores ausentes na base:
print("Quantidade de missings no treino: ", X_train.isna().sum().sum())
print("Quantidade de missings no teste: ", X_test.isna().sum().sum())

# %% [markdown]
# ### Bloco de Código 11.12 — Organização das variáveis por tipo
#
# *Página 199 do PDF · Revisão: confirmado*
#
# **Código revisado do livro.**

# %% [markdown]
# **Complemento:** harmonização de nomes conforme a seleção real do bloco 11.4. A coluna `sg_uf` não existe mais. Mantemos os nomes minúsculos usados nos blocos seguintes.

# %% [code] célula 32
X_train.columns=X_train.columns.str.lower();X_test.columns=X_test.columns.str.lower()
assert X_train.columns.equals(X_test.columns)
print('Colunas:',list(X_train.columns))

# %% [code] célula 33
# Lista de variáveis numéricas
lista_variaveis_numericas = ["idade_anos"]

# Lista de variáveis categóricas
lista_variaveis_categóricas = ["sg_uf", "cs_sexo", "cs_raca", "cs_escol_n",
                               "tratsup_at", "tratamento", "agravdroga",
                               "pop_rua", "pop_liber", "bacilosc_e",
                               "agravalcoo", "forma", "cs_gestant", "agravtabac"]

# %% [code] célula 34
lista_variaveis_categóricas=[c for c in lista_variaveis_categóricas if c in X_train.columns and c in X_test.columns]
assert 'sg_uf' not in lista_variaveis_categóricas
print('Variáveis categóricas presentes:',lista_variaveis_categóricas)

# %% [markdown]
# ### Bloco de Código 11.13 — Imputação de valores ausentes em variáveis numéricas (mediana)
#
# *Página 200 do PDF · Revisão: confirmado*
#
# **Código revisado do livro.**

# %% [code] célula 36
# Cálculo das medianas apenas no conjunto de treino
medianas_treino = X_train[lista_variaveis_numericas].median()

# Aplicação das medianas do treino para imputacao em treino e teste
X_train[lista_variaveis_numericas] = X_train[lista_variaveis_numericas].fillna(medianas_treino)
X_test[lista_variaveis_numericas] = X_test[lista_variaveis_numericas].fillna(medianas_treino)

# %% [markdown]
# ### Bloco de Código 11.14 — Imputação de valores ausentes em variáveis categórias (categoria sintética de não informado)
#
# *Página 201 do PDF · Revisão: confirmado*
#
# **Código revisado do livro.**

# %% [markdown]
# **Complemento:** as colunas foram convertidas para `category` em 11.5; a categoria sintética 9 precisa ser adicionada antes do `fillna(9)`. Também é necessário que as colunas sejam tratadas de modo compatível com a atribuição.

# %% [code] célula 39
for coluna in lista_variaveis_categóricas:
    for frame in (X_train,X_test):
        frame[coluna]=frame[coluna].cat.add_categories([9]) if 9 not in frame[coluna].cat.categories else frame[coluna]
print('Categoria 9 habilitada nas variáveis categóricas.')

# %% [code] célula 40
# Preenchimento das categorias ausentes com o código 9 (categoria sintética de "não informado")
X_train.loc[:, lista_variaveis_categóricas] = X_train.loc[:, lista_variaveis_categóricas].fillna(9)
X_test.loc[:, lista_variaveis_categóricas] = X_test.loc[:, lista_variaveis_categóricas].fillna(9)

# %% [markdown]
# ### Bloco de Código 11.15 — Imputação de valores ausentes em variáveis categórias (moda)
#
# *Página 201 do PDF · Revisão: confirmado*
#
# **Código revisado do livro.**

# %% [markdown]
# **Observação metodológica:** como o bloco 11.14 já preencheu todos os ausentes, a imputação por moda abaixo não terá efeito. Para comparar os métodos seria necessário executar pipelines alternativos desde o mesmo ponto inicial.

# %% [code] célula 43
# Obtenção da moda nas bases de treino
modas_treino = X_train[lista_variaveis_categóricas].mode().iloc[0]

# Aplicação da moda do treino nas base de treino e de teste
X_train[lista_variaveis_categóricas] = X_train[lista_variaveis_categóricas].fillna(modas_treino)
X_test[lista_variaveis_categóricas] = X_test[lista_variaveis_categóricas].fillna(modas_treino)

# %% [markdown]
# ### Bloco de Código 11.16 — Escalonamento por StandardScaler
#
# *Página 203 do PDF · Revisão: confirmado*
#
# **Código revisado do livro.**

# %% [code] célula 45
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

# %% [markdown]
# ### Bloco de Código 11.17 — One-hot encoding nas variáveis categóricas
#
# *Página 205 do PDF · Revisão: confirmado*
#
# **Código revisado do livro.**

# %% [markdown]
# **Compatibilidade:** a API `sparse_output=False` requer versão recente do scikit-learn.

# %% [code] célula 48
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

# %% [code] célula 49
assert X_train_encoded.shape[1]==X_test_encoded.shape[1]
assert X_train_encoded.index.equals(X_train.index)
print('One-hot: mesmas colunas e alinhamento preservado')

# %% [markdown]
# ### Bloco de Código 11.18 — Exemplo hipotético de aplicação do ordinal encoding
#
# *Página 207 do PDF · Revisão: confirmado*
#
# **Código revisado do livro.**

# %% [markdown]
# **Exemplo hipotético isolado:** o código original é reproduzido abaixo como referência, mas não executado diretamente porque depende de variáveis não presentes no fluxo principal e contém incompatibilidades de API/estrutura. A célula seguinte demonstra a técnica com dados próprios.
#
# ```python
# from sklearn.preprocessing import OrdinalEncoder
#
# # Exemplo hipotético: variável de escolaridade armazenada como texto
# # Ordem natural definida explicitamente com base no conhecimento da variável
# ordem_escolaridade = [
#     "sem instrução",
#     "ensino fundamental",
#     "ensino médio",
#     "ensino superior"
# ]
#
# # Criação do encoder com a ordem explícita
# ordinal_encoder = OrdinalEncoder(
#     categories=ordem_escolaridade,
#     handle_unknown="use_encoded_value",
#     unknown_value=-1
# )
#
# # Ajuste no treino e transformação em treino e teste
# X_train["escolaridade"] = ordinal_encoder.fit_transform(X_train["escolaridade"])
# X_test["escolaridade"] = ordinal_encoder.transform(X_test["escolaridade"])
#
# ```

# %% [code] célula 52
from sklearn.preprocessing import OrdinalEncoder
escolaridade_treino=pd.DataFrame({'escolaridade':['sem instrução','ensino fundamental','ensino médio','ensino superior']})
escolaridade_teste=pd.DataFrame({'escolaridade':['ensino médio','não informado']})
ordem_escolaridade=['sem instrução','ensino fundamental','ensino médio','ensino superior']
ordinal_encoder_demo=OrdinalEncoder(categories=[ordem_escolaridade],handle_unknown='use_encoded_value',unknown_value=-1)
encoded_treino=ordinal_encoder_demo.fit_transform(escolaridade_treino[['escolaridade']])
encoded_teste=ordinal_encoder_demo.transform(escolaridade_teste[['escolaridade']])
print('Ordinal treino:',encoded_treino.ravel(),'teste:',encoded_teste.ravel())
assert encoded_teste[1,0]==-1

# %% [markdown]
# ### Bloco de Código 11.19 — Exemplo hipotético de aplicação do target encoding
#
# *Página 209 do PDF · Revisão: confirmado*
#
# **Código revisado do livro.**

# %% [markdown]
# **Exemplo hipotético isolado:** o código original é reproduzido abaixo como referência, mas não executado diretamente porque depende de variáveis não presentes no fluxo principal e contém incompatibilidades de API/estrutura. A célula seguinte demonstra a técnica com dados próprios.
#
# ```python
# # Exemplo hipotético: target encoding para ID_MUNICIP
# # Calculando a proporção de óbitos por município no conjunto de treino
# target_means = X_train.join(y_train).groupby("id_municip")["obito"].mean()
#
# # Aplicando no treino e no teste
# X_train["id_municip_te"] = X_train["id_municip"].map(target_means)
# X_test["id_municip_te"] = X_test["id_municip"].map(target_means)
#
# # Municípios que aparecem no teste mas não no treino
# # recebem a média global do desfecho no treino
# media_global = y_train.mean()
# X_train["id_municip_te"] = X_train["id_municip_te"].fillna(media_global)
# X_test["id_municip_te"] = X_test["id_municip_te"].fillna(media_global)
#
# # Removendo a coluna original
# X_train.drop(columns=["id_municip"], inplace=True)
# X_test.drop(columns=["id_municip"], inplace=True)
#
# ```

# %% [code] célula 55
# Demonstração isolada de target encoding; nunca usar rótulos do teste para estimar médias.
municipios_treino=pd.DataFrame({'id_municip':[101,101,102,103,103,103], 'obito':[0,1,0,1,0,0]})
municipios_teste=pd.DataFrame({'id_municip':[101,102,999]})
medias=municipios_treino.groupby('id_municip')['obito'].mean()
media_global=municipios_treino['obito'].mean()
municipios_teste['id_municip_te']=municipios_teste['id_municip'].map(medias).fillna(media_global)
print(municipios_teste)
assert np.isfinite(municipios_teste['id_municip_te']).all()
print('Cautela: médias calculadas diretamente no treino podem gerar vazamento dentro do treino; para modelagem recomenda-se codificação out-of-fold.')

# %% [markdown]
# ### Bloco de Código 11.20 — Consolidação da base de dados
#
# *Página 210 do PDF · Revisão: confirmado*
#
# **Código revisado do livro.**

# %% [code] célula 57
# Remoção das variáveis categóricas originais
X_train.drop(columns=lista_variaveis_categóricas, inplace=True)
X_test.drop(columns=lista_variaveis_categóricas, inplace=True)

# União das variáveis numéricas com as colunas geradas pelo encoding
X_train = pd.concat([X_train, X_train_encoded], axis=1)
X_test = pd.concat([X_test, X_test_encoded], axis=1)

# %% [markdown]
# ### Bloco de Código 11.21 — Salvamento da base de dados
#
# *Página 210 do PDF · Revisão: confirmado*
#
# **Código revisado do livro.**

# %% [code] célula 59
# Reunindo preditores e desfecho na base de treino
df_train = pd.concat([X_train, y_train], axis=1)
df_train.to_csv("dados_treino_preprocessados.csv", index=False)

# Reunindo preditores e desfecho na base de teste
df_test = pd.concat([X_test, y_test], axis=1)
df_test.to_csv("dados_teste_preprocessados.csv", index=False)

# %% [code] célula 60
assert X_train.columns.equals(X_test.columns)
assert len(df_train)==len(y_train) and len(df_test)==len(y_test)
assert not df_train.isna().any().any() and not df_test.isna().any().any()
print('Bases sintéticas exportadas, colunas compatíveis e sem ausentes')
