# Importando bibliotecas necessárias
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression


# Exemplo de dados de transações simuladas.
"""
    Variaveis que podem influenciar fraude:
    - valor_transacao (em dólars)
    - tempo_conta (em messes desde criação da conta do usuario)
    - num_transacoes_ult_30_dias (quantidade de compras nos últimos 30 dias)
    - pias_origem (codificacao numericamente, ex: 0 - Brasil, 1 - EUA, 2 - Outros)
    Alvo (fraude): 1 - fraude, 0 - legitimo
"""

data = {
    'valor_transacao': [15, 200, 350, 60, 500, 9000, 30, 850, 4999],
    'tempo_conta': [2, 24, 1, 12, 36, 0, 6, 48, 3],    
    'num_transacoes_ult_30_dias': [1, 5, 0, 3, 10, 20, 2, 15, 4],
    'pais_origem': [0, 0, 2, 2, 1, 1, 0, 1, 2], 
    'fraude': [0, 0, 1, 0, 0, 1, 0, 0, 1]  # 0 - legitimo, 1 - fraude
}

df = pd.DataFrame(data)

# Separando as variáveis independentes (X) e a variável dependente (y)
X = df[['valor_transacao', 'tempo_conta', 'num_transacoes_ult_30_dias', 'pais_origem']]
y = df['fraude']

# Criando o modelo de regressão logística
model = LogisticRegression()    

# Treinando o modelo
model.fit(X, y)

# Salvando o modelo treinado em um arquivo
with open('modelo_fraude.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Modelo treinado e salvo como 'modelo_fraude.pkl'")