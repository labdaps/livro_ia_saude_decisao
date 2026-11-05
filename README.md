# Exemplos de IA em Saúde — notebooks e scripts locais

Projeto consolidado dos 14 capítulos revisados (07, 08, 11–21, 23). Os notebooks de `notebooks/capitulo_XX.ipynb` correspondem às versões **executáveis/adaptadas** das validações anteriores. As versões originais e saídas históricas ficam em `relatorios/`. Os retestes com a amostra real estão nomeados `capitulo_XX_real.ipynb` (e diagnóstico do capítulo 20).

## Estrutura

- `notebooks/`: um notebook executável por capítulo e notebooks adicionais de reteste com dados reais.
- `python/`: exportação das células de código para scripts `.py` com marcações `# %%`; preserva a ordem, mas nem todo script é independente de estado, arquivos ou bibliotecas.
- `dados/`: amostra SINAN-TB fornecida pelo usuário, 9.999 linhas; tratar como amostra e observar políticas de privacidade e acesso.
- `relatorios/`: relatórios por capítulo, versões originais, execuções anteriores, retestes e manifesto de compilação.
- `resultados/`: destino de notebooks executados localmente.
- `scripts/`: verificador de ambiente e executor de notebooks.
- `TAREFAS_VALIDACAO.md`: checklist para homologação completa.

## Instalação (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/verificar_ambiente.py
```

## Instalação (Linux/macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/verificar_ambiente.py
```

## Uso no Jupyter (sem Colab)

```bash
jupyter lab
```

Abra `notebooks/capitulo_XX.ipynb` e use **Run All** com kernel limpo. Para executar por linha de comando:

```bash
python scripts/executar_notebook.py notebooks/capitulo_11.ipynb
```

## Uso em Python puro

```bash
python python/capitulo_11.py
```

Scripts são exportações fiéis das **células da versão executável**, não reimplementações independentes. Alguns exemplos dependem de caminhos locais, pacotes opcionais, execução anterior ou recursos externos; confira o checklist. Os comandos mágicos de notebook são comentados e precisam de adaptação manual quando aplicáveis. Os scripts foram verificados quanto à sintaxe, **não executados todos de ponta a ponta**.

## Dados reais e limitações

A amostra `dados/dados_tuberculose_v2_10000.csv` não substitui automaticamente os dados sintéticos em todos os capítulos. Para retestes já realizados, prefira os notebooks `_real`. O experimento original do capítulo 20 requer Sergipe 2022–2023, ausente nesta amostra. Os capítulos 15, 16, 18, 21 e 23 têm validações originais pendentes por dependências e/ou infraestrutura. A compatibilidade exata de versões não foi fixada; registrar um `pip freeze` após instalar em ambiente funcional.

Veja `TAREFAS_VALIDACAO.md` para a lista completa.
