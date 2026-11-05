# -*- coding: utf-8 -*-
# Capítulo 07: exportação automática das células do notebook executável.
# Executar na raiz do projeto: python python/capitulo_07.py
# Células que dependem de contexto de notebook, serviços ou bibliotecas opcionais podem exigir ajustes.

# %% [markdown]
# # Capítulo 7 — Versão executável e auditável
#
# Os códigos do livro são preservados em células próprias. Complementos e verificações estão separados. A célula 7.2 requer `pydicom` e os dados de teste DICOM correspondentes.

# %% [markdown]
# ## Preparação do ambiente
#
# Dependências: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `pydicom`.
#
# No ambiente local, execute no terminal `python -m pip install -r requirements_capitulo_07.txt`. Não são instalados pacotes automaticamente durante a execução.

# %% [code] célula 3
import importlib.util
modulos = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "pydicom"]
disponibilidade = {m: importlib.util.find_spec(m) is not None for m in modulos}
print("Disponibilidade:", disponibilidade)

# %% [markdown]
# ### Bloco de Código 7.1 — Exploração de dados estruturados clínicos
#
# *Página 88 do PDF · Revisão: confirmado*
#
# **Código original revisado:** sem modificações.

# %% [code] célula 5
import pandas as pd
import numpy as np

# 1. Simulacao de conjunto de dados clinico (n=200)
np.random.seed(42)
n = 200

df = pd.DataFrame({
    "paciente_id": range(1, n + 1),
    "idade": np.random.randint(18, 95, n),
    "sexo": np.random.choice(["F", "M"], n),
    "pressao_sistolica": np.random.normal(130, 20, n).round().astype(int).clip(80, 220),
    "hemoglobina_g_dL": np.random.normal(13.5, 2.0, n).round(1).clip(4.0, 20.0),
    "glicemia_mg_dL": np.random.normal(110, 35, n).round(1).clip(30),
    "tipo_sanguineo": np.random.choice(["A+", "A-", "B+", "B-", "O+", "O-"], n),
    "estagio_tumor": pd.Categorical(
        np.random.choice(["I", "II", "III", "IV"], n),
        categories=["I", "II", "III", "IV"],
        ordered=True
    ),
    "obito": np.random.choice([0, 1], n, p=[0.85, 0.15]),
    "data_admissao": pd.date_range("2023-01-01", periods=n, freq="2D"),
})

# Inserindo ~7% de valores ausentes em colunas laboratoriais
for col in ["hemoglobina_g_dL", "glicemia_mg_dL"]:
    df.loc[np.random.choice(n, int(n * 0.07), replace=False), col] = np.nan

# 2. Panorama geral do DataFrame
print(df.dtypes, end="\n\n")
print(df.isnull().sum()[lambda s: s > 0], end="\n\n")  # so colunas com NaN

# 3. Estatisticas descritivas por desfecho
numericas = ["idade", "pressao_sistolica", "hemoglobina_g_dL", "glicemia_mg_dL"]

print(
    df.groupby("obito")[numericas]
      .agg(["mean", "median", "std"])
      .round(2),
    end="\n\n"
)

# 4. Tabela cruzada - estadiamento x obito (contagem e proporcao)
print(pd.crosstab(df["estagio_tumor"], df["obito"], margins=True))

print(
    pd.crosstab(
        df["estagio_tumor"],
        df["obito"],
        normalize="index"
    ).round(3)
)

# %% [code] célula 6
assert len(df) == 200
assert df["paciente_id"].is_unique
assert df["obito"].isin([0, 1]).all()
print("Verificação 7.1: OK — 200 registros e estrutura consistente")

# %% [markdown]
# ### Bloco de Código 7.2 — Fluxo básico de pré-processamento de imagem médica
#
# *Página 93 do PDF · Revisão: confirmado*
#
# **Código original revisado:** sem modificações.

# %% [markdown]
# **Pré-condição:** esta célula somente será executada quando `pydicom` estiver instalado e o DICOM de teste estiver acessível. Caso contrário, a célula informa o bloqueio sem impedir os outros blocos.

# %% [code] célula 9
if disponibilidade["pydicom"]:
    import pydicom
    from pydicom.data import get_testdata_file
    import numpy as np
    import matplotlib.pyplot as plt

    # 1. Carregar um DICOM real
    caminho = get_testdata_file("CT_small.dcm")
    ds = pydicom.dcmread(caminho)

    # 2. Extrair metadados estruturados do cabecalho DICOM
    meta = {
        "modalidade": ds.Modality,
        "espacamento_pixel_mm": [round(float(x), 2) for x in ds.PixelSpacing],
        "slice_thickness_mm": float(ds.SliceThickness),
        "rescale_slope": float(ds.RescaleSlope),
        "rescale_intercept": float(ds.RescaleIntercept),
    }
    print("Metadados do cabecalho DICOM:")
    for chave, valor in meta.items():
        print(f" {chave}: {valor}")

    # 3. Converter pixeis brutos -> Unidades Hounsfield (HU)
    pixeis = ds.pixel_array.astype(np.float64)
    hu = pixeis * meta["rescale_slope"] + meta["rescale_intercept"]
    print(f"\nEstatisticas HU: min={hu.min():.0f}, max={hu.max():.0f}")

else:
    print("BLOQUEADO (ambiente): instale pydicom e disponibilize CT_small.dcm para executar o bloco 7.2")

# %% [markdown]
# ### Bloco de Código 7.3 — Processamento de linguagem natural clínica em ação:
#
# *Página 98 do PDF · Revisão: confirmado*
#
# **Código original revisado:** sem modificações.

# %% [code] célula 11
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Laudos radiológicos simulados
laudos = [
    "Paciente sem sinais de pneumonia aguda.",  # Negativo
    "Extensa pneumonia aguda no lobo inferior.",  # Positivo
    "Ausência de consolidação ou pneumonia."  # Negativo (duplicado)
]

# Vectorização TF-IDF
vectorizer = TfidfVectorizer()
matriz_tfidf = vectorizer.fit_transform(laudos)

# Exibir pesos para termos clínicamente relevantes
vocab = vectorizer.get_feature_names_out()
df_tfidf = pd.DataFrame(
    matriz_tfidf.toarray(),
    columns=vocab,
    index=["Laudo 1 (neg)", "Laudo 2 (pos)", "Laudo 3 (neg)"]
)

# Foco nos termos críticos para a tarefa
termos_interesse = ["pneumonia", "sem", "sinais", "ausência", "extensa"]
termos_disponíveis = [t for t in termos_interesse if t in df_tfidf.columns]
print("Pesos TF-IDF para termos selecionados:")
print(df_tfidf[termos_disponíveis].round(3))
print()

# --- Observação didática ---
# Nota que "pneumonia" recebe peso significativo em três laudos, tanto no positivo quanto no negativo. O TF-IDF não distingue "sem pneumonia" de "extensa pneumonia" porque trata cada unidade textual de forma independente (saco de palavras). Essa é a limitação fundamental que motivou o desenvolvimento de representações vetoriais contextuais como BERT, onde a representação de "pneumonia" muda conforme o contexto sintático em que aparece.

# %% [code] célula 12
assert matriz_tfidf.shape[0] == 3
assert "pneumonia" in vocab
print("Verificação 7.3: OK — matriz TF-IDF com 3 documentos")

# %% [markdown]
# ### Bloco de Código 7.4 — Filtragem de sinal ECG
#
# *Página 100 do PDF · Revisão: confirmado*
#
# **Código original revisado:** sem modificações.

# %% [code] célula 14
import numpy as np
from scipy.signal import butter, filtfilt

def filtro_passa_banda(sinal, fs, f_low=0.5, f_high=40, ordem=4):
    """
    Aplica filtro Butterworth passa-banda ao sinal ECG.

    Parametros
    ---------
    sinal : np.ndarray
        Vetor 1D com o sinal bruto de uma derivacao.
    fs : float
        Frequencia de amostragem em Hz (ex: 500).
    f_low, f_high : float
        Frequencias de corte inferior e superior em Hz.
    ordem : int
        Ordem do filtro Butterworth.

    Retorna
    --------
    sinal_filtrado : np.ndarray
        Sinal com ruído fora da faixa 0.5-40 Hz atenuado.

    """
    nyq = fs / 2
    b, a = butter(ordem, [f_low / nyq, f_high / nyq], btype='band')
    return filtfilt(b, a, sinal)

# Exemplo: sinal simulado com ruído de 60 Hz
fs = 500  # 500 Hz de amostragem
t = np.arange(0, 2, 1/fs)  # 2 segundos
sinal_limpo = np.sin(2 * np.pi * 1 * t)  # componente cardíaco ~1 Hz
ruído_60hz = 0.3 * np.sin(2 * np.pi * 60 * t)  # interferência da rede
sinal_ruidoso = sinal_limpo + ruído_60hz

sinal_filtrado = filtro_passa_banda(sinal_ruidoso, fs)
print(f"Amplitude do ruído antes: {np.std(sinal_ruidoso):.3f}")
print(f"Amplitude do ruído depois: {np.std(sinal_filtrado):.3f}")

# %% [markdown]
# **Complemento de validação:** mede especificamente a amplitude da componente de 60 Hz, que o desvio-padrão total não isola.

# %% [code] célula 16
def amplitude_frequencia(sinal, fs, hz):
    frequencias = np.fft.rfftfreq(len(sinal), 1/fs)
    espectro = np.abs(np.fft.rfft(sinal)) * 2 / len(sinal)
    return float(espectro[np.argmin(abs(frequencias-hz))])
antes_60 = amplitude_frequencia(sinal_ruidoso, fs, 60)
depois_60 = amplitude_frequencia(sinal_filtrado, fs, 60)
print(f"60 Hz: antes={antes_60:.5f}, depois={depois_60:.5f}")
assert depois_60 < antes_60
print("Verificação 7.4: OK — componente de 60 Hz atenuada")

# %% [markdown]
# ### Bloco de Código 7.5 — Extração de atributos longitudinais com proteção contra vazamento temporal
#
# *Página 102 do PDF · Revisão: confirmado*
#
# **Código original revisado:** sem modificações.

# %% [code] célula 18
import pandas as pd
import numpy as np

# Historico laboratorial de um paciente renal crônico
# com visitas ambulatoriais irregulares - padrão típico em nefrologia
historico = pd.DataFrame({
    'data': pd.to_datetime([
        '2022-01-10',  # visita 1
        '2022-08-09',  # visita 2 - hemoglobina glicada não realizada
        '2023-05-15',  # visita 3
        '2024-06-01'   # visita 4 - DEPOIS do ponto de predicao
    ]),
    'hemoglobina_glicada': [6.5, np.nan, 7.8, 8.1],
    'creatinina': [1.1, 1.3,     1.8, 3.2],
})

# Ponto de predicao: queremos estimar risco em 1º de janeiro de 2024.
# Regra: nenhuma informacao após esta data pode entrar como atributo.
data_predicao = pd.to_datetime('2024-01-01')

# %% [markdown]
# **Complemento proposto (não faz parte da transcrição):** aplica o corte temporal e calcula atributos usando apenas observações anteriores à data de predição.

# %% [code] célula 20
passado = historico.loc[historico["data"] <= data_predicao].sort_values("data").copy()
assert not passado.empty
assert passado["data"].max() <= data_predicao
atributos = {
    "ultima_creatinina": float(passado["creatinina"].dropna().iloc[-1]),
    "ultima_hemoglobina_glicada": float(passado["hemoglobina_glicada"].dropna().iloc[-1]),
    "numero_visitas": int(len(passado)),
}
print("Atributos até a data de predição:", atributos)
assert atributos["ultima_creatinina"] == 1.8
assert atributos["numero_visitas"] == 3
print("Verificação 7.5: OK — sem vazamento temporal")
