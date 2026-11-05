# -*- coding: utf-8 -*-
# Capítulo 23: exportação automática das células do notebook executável.
# Executar na raiz do projeto: python python/capitulo_23.py
# Células que dependem de contexto de notebook, serviços ou bibliotecas opcionais podem exigir ajustes.

# %% [markdown]
# # Capítulo 23 — Aprendizado federado
#
# **Validação local segura.** Os blocos originais são preservados como referência, sem executar conexões RPC. A simulação NumPy abaixo não é uma execução do Flower ou TensorFlow.

# %% [markdown]
# ### Bloco de Código 23.1 — Implementação do cliente federado (hospital) utilizando a biblioteca Flower e TensorFlow.
#
# *Página 482 do PDF · Revisão: confirmado*
#
# **Código original (referência; não executado):**

# %% [markdown]
# ```python
# import tensorflow as tf
# import flwr as fl
#
# # 1. Definimos uma rede neural simples (Parâmetros 'w')
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(10, activation="relu", input_shape=(8,)),  # Entrada: 8 features
#     tf.keras.layers.Dense(1, activation="sigmoid")  # Saída: Probabilidade
# ])
# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["auc"])
#
# # 2. Implementação do Cliente Flower
# class HospitalClient(fl.client.NumPyClient):
#     def get_parameters(self, config):
#         # Extrai os pesos (w) da rede neural como matrizes NumPy
#         return model.get_weights()
#
#     def fit(self, parameters, config):
#         # Atualiza a rede local com os pesos globais do servidor
#         model.set_weights(parameters)
#
#         # Treina localmente (Minimiza Fk(w))
#         # Simulamos o uso de nk = 200 amostras
#         model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
#
#         # Retorna os pesos ajustados e o valor de nk para a média ponderada
#         return model.get_weights(), 200, {}
#
#     def evaluate(self, parameters, config):
#         # Avalia o modelo global nos dados do hospital k
#         model.set_weights(parameters)
#         loss, auc = model.evaluate(X_test, y_test, verbose=0)
#         return loss, 50, {"auc": auc}  # nk_test = 50
#
# # Conecta o hospital ao servidor orquestrador via RPC
# fl.client.start_numpy_client(server_address="[...]:8080", client=HospitalClient())
#
# ```

# %% [markdown]
# ### Adaptação 23.1 — Dados locais e atualização de parâmetros, sem TensorFlow/Flower
#
# Simulação de atualizações locais com gradiente logístico e diferentes tamanhos de amostra. **Não representa o treinamento original da rede neural.**

# %% [code] célula 5
import numpy as np
rng = np.random.default_rng(23)
num_hospitais, n_features = 5, 8
sizes = [100, 150, 200, 250, 300]
w_global = np.zeros(n_features)

def treino_local(w_inicial, X, y, taxa=0.2):
    logits = np.clip(X @ w_inicial, -30, 30)
    p = 1 / (1 + np.exp(-logits))
    grad = X.T @ (p - y) / len(y)
    return w_inicial - taxa * grad

hospitais = []
for n in sizes:
    X = rng.normal(size=(n, n_features))
    logits = X @ np.array([1.1, -.8, .4, 0, 0, .6, -.2, .3])
    p = 1/(1+np.exp(-np.clip(logits,-30,30)))
    y = rng.binomial(1, p)
    hospitais.append((X,y))
assert len(hospitais)==5 and sum(map(lambda h: len(h[1]),hospitais))==1000
print('Hospitais:',len(hospitais),'| Amostras locais:',sizes)

# %% [markdown]
# ### Bloco de Código 23.2 — Configuração do servidor federado e definição da estratégia de agregação FedAvg.
#
# *Página 483 do PDF · Revisão: confirmado*
#
# **Código original (referência; não executado):**

# %% [markdown]
# ```python
# import flwr as fl
#
# # Define a estratégia FedAvg: o servidor calculará (nk/n) * pesos_locais
# # Configurado para exigir o mínimo de 5 hospitais para o aprendizado
# strategy = fl.server.strategy.FedAvg(
#     min_fit_clients=5,
#     min_available_clients=5,
# )
#
# # Inicia o processo de treinamento federado
# fl.server.start_server(
#     server_address="[...]:8080",
#     config=fl.server.ServerConfig(num_rounds=5),
#     strategy=strategy
# )
#
# ```

# %% [markdown]
# ### Adaptação 23.2 — Agregação FedAvg com NumPy (simulação matemática)
#
# Verifica a média ponderada por número de amostras e a consistência com o cálculo explícito. Não há servidor, comunicação ou orquestração Flower.

# %% [code] célula 9
def fedavg(pesos_locais, quantidades):
    if len(pesos_locais) != len(quantidades) or not pesos_locais:
        raise ValueError('Listas de clientes incompatíveis')
    ns = np.asarray(quantidades, dtype=float)
    if np.any(ns <= 0):
        raise ValueError('Cada cliente deve ter amostras')
    pesos = np.stack(pesos_locais)
    return np.average(pesos, axis=0, weights=ns)

historico=[]
for rodada in range(1,6):
    locais=[treino_local(w_global, X,y) for X,y in hospitais]
    novo=fedavg(locais,sizes)
    esperado=sum(n*w for n,w in zip(sizes,locais))/sum(sizes)
    assert np.allclose(novo,esperado)
    assert np.isfinite(novo).all() and novo.shape==(8,)
    variacao=float(np.linalg.norm(novo-w_global))
    historico.append((rodada,variacao))
    w_global=novo
    print(f'Rodada {rodada}: variação L2 = {variacao:.6f}')
assert len(historico)==5
assert np.allclose(fedavg([np.ones(8),np.zeros(8)],[1,3]),np.ones(8)*.25)
print('Testes FedAvg: OK; pesos finais:',np.round(w_global,4))
