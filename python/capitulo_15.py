# -*- coding: utf-8 -*-
# Capítulo 15: exportação automática das células do notebook executável.
# Executar na raiz do projeto: python python/capitulo_15.py
# Células que dependem de contexto de notebook, serviços ou bibliotecas opcionais podem exigir ajustes.

# %% [markdown]
# # Capítulo 15 — Deep learning (versão executável)
#
# **Nota:** os três blocos originais estão preservados integralmente no notebook `capitulo_15_original.ipynb`. Os complementos são demonstrações técnicas e não resultados clínicos. Os rótulos de pacientes 9 e 10 são conhecidos nos comentários do código, mas não constam do vetor `y` revisado; não são completados silenciosamente.

# %% [markdown]
# ### Bloco de Código 15.1 — Exemplo de uma rede neural de três camadas ocultas
#
# *Página 300 do PDF · Revisão: confirmado*
#
# **Diagnóstico:** o trecho extraído é incompleto: cria 10 registros e 8 rótulos, e define somente a primeira camada. A seguir, valida-se a arquitetura 4–16–8–4–1 descrita no bloco seguinte, usando **apenas os oito pares rotulados**, com PyTorch como implementação demonstrativa, não como execução do TensorFlow original.

# %% [code] célula 3
import numpy as np
import torch
from torch import nn
np.random.seed(42)
torch.manual_seed(42)
X_pacientes_historico = np.array([[1,1,1,.9],[0,0,0,.1],[1,0,0,.2],[0,1,0,.8],[1,0,1,.4],[1,1,0,.7],[0,0,0,.2],[0,0,0,.8],[1,1,1,.6],[1,0,0,.1]],dtype='float32')
y_diagnostico_real=np.array([1,0,0,1,0,1,1,0],dtype='float32')
assert len(X_pacientes_historico)==10 and len(y_diagnostico_real)==8
print('Inconsistência original: 10 registros e 8 rótulos. Treino demonstrativo usa apenas 8 pares.')
X=torch.tensor(X_pacientes_historico[:len(y_diagnostico_real)])
y=torch.tensor(y_diagnostico_real).reshape(-1,1)
modelo=nn.Sequential(nn.Linear(4,16),nn.ReLU(),nn.Linear(16,8),nn.ReLU(),nn.Linear(8,4),nn.ReLU(),nn.Linear(4,1),nn.Sigmoid())
opt=torch.optim.Adam(modelo.parameters(),lr=.02)
loss_fn=nn.BCELoss()
for epoch in range(60):
 opt.zero_grad();loss=loss_fn(modelo(X),y);loss.backward();opt.step()
with torch.no_grad():
 p=modelo(X)
 assert p.shape==(8,1) and torch.isfinite(p).all()
 print('Arquitetura: 4→16→8→4→1; perda final:',round(loss.item(),4))
 print('Probabilidades (treino demonstrativo):',np.round(p.numpy().ravel(),3))

# %% [markdown]
# ### Bloco de Código 15.2 — Classificação com um modelo fundacional TabPFN
#
# *Página 301 do PDF · Revisão: confirmado*
#
# **Diagnóstico:** o trecho original somente configura tamanhos, cores e rótulos. O desenho abaixo é um complemento visual, não uma figura já existente no bloco extraído.

# %% [code] célula 5
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
sizes=[4,16,8,4,1]
fig,ax=plt.subplots(figsize=(10,6))
for k,count in enumerate(sizes):
 ys=np.linspace(-2.3,2.3,count)
 for yy in ys: ax.add_patch(Circle((k*2.0,yy),radius=.11,facecolor=['#3498db','#9b59b6','#2ecc71','#f1c40f','#e74c3c'][k],edgecolor='black',lw=.3))
 ax.text(k*2.0,-2.8,['Entrada (4)','Oculta 1 (16)','Oculta 2 (8)','Oculta 3 (4)','Saída (1)'][k],ha='center',fontsize=9)
ax.set_xlim(-.6,8.6);ax.set_ylim(-3.2,2.8);ax.set_aspect('equal');ax.axis('off');ax.set_title('Arquitetura ilustrativa 4–16–8–4–1');plt.show()
assert len(sizes)==5 and sizes[0]==4 and sizes[-1]==1

# %% [markdown]
# ### Bloco de Código 15.3 — Visualizacao de uma rede neural de 3 camadas ocultas
#
# *Página 320 do PDF · Revisão: confirmado*
#
# **Diagnóstico:** o código usa `tabfn` (nome incompatível com o pacote TabPFN conhecido) e possui 10 linhas de atributos para apenas 8 rótulos. A célula abaixo verifica a dependência sem instalar pacotes, baixar pesos ou substituir o modelo. Caso TabPFN esteja disponível, realiza um teste real com os 8 pares rotulados.

# %% [code] célula 7
import importlib.util
import numpy as np
X=np.array([[1,1,1,.9],[0,0,0,.1],[1,0,0,.2],[0,1,0,.8],[1,0,1,.4],[1,1,0,.7],[0,0,1,.2],[0,0,0,.8],[1,1,1,.6],[1,0,0,.1]])
y=np.array([1,0,0,1,0,1,1,0])
assert X.shape==(10,4) and y.shape==(8,)
if importlib.util.find_spec('tabpfn') is None:
 print('BLOQUEADO: pacote tabpfn não instalado. Nenhum resultado TabPFN foi simulado.')
else:
 from tabpfn import TabPFNClassifier
 try:
  modelo_tabpfn=TabPFNClassifier()
  modelo_tabpfn.fit(X[:len(y)],y)
  probabilidades=modelo_tabpfn.predict_proba(np.array([[1,0,1,.75]]))
  assert probabilidades.shape==(1,2)
  print('Probabilidade estimada (TabPFN real):',probabilidades[0,1])
 except Exception as exc:
  print('BLOQUEADO: execução real do TabPFN falhou:',type(exc).__name__,str(exc)[:400])
