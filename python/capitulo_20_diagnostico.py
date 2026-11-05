# -*- coding: utf-8 -*-
# Reteste com amostra SINAN: capitulo_20_diagnostico.ipynb
# Exportação automática; executar na raiz do projeto.

# %% [code] célula 2
import pandas as pd
df=pd.read_csv('/mnt/data/dados_tuberculose_v2_10000.csv',low_memory=False)
df=df[(df.FORMA==1)&df.SITUA_ENCE.isin([1,3])&(df.IDADE_ANOS>=18)&~df.NU_ANO.isin([2020,2021])]
fonte=df[(df.SG_UF==26)&(df.NU_ANO<=2019)]
destino=df[(df.SG_UF==28)&(df.NU_ANO>=2022)]
print("Pernambuco:",len(fonte),"Sergipe:",len(destino))
assert len(destino)>0, "Não há observações no domínio destino: transferência não pode ser avaliada."
