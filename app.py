import streamlit as st
import pickle
import numpy as np

# Carregando o modelo treinado
with open('modelo_fraude.pkl', 'rb') as file:
    model = pickle.load(file)

# Título do aplicativo
st.title("Detecção de Fraude em Transações")
st.write("Este aplicativo utiliza um modelo de aprendizado de máquina para detectar fraudes em transações financeiras.")


# Seção de entrada de dados
st.header("Detalhes de Transação")

# Exemplo de sliders para capturar variáveis (simulando features de um dataset)
valor_transacao = st.number_input("Valor da Transação (em dólares)", min_value=0, max_value=10000, value=50)
tempo_conta = st.number_input("Tempo de Conta (em meses)", min_value=0, max_value=120, value=6)        
num_transacoes_ult_30_dias = st.number_input("Número de Transações nos Últimos 30 Dias", min_value=0, max_value=1000, value=3)


# Exemplo de seleção para país de origem
pais_origem_options = {
    "Brasil": 0,
    "EUA": 1,
    "Outros": 2
}
pais_origem_escolhido = st.selectbox("País de Origem", options=list(pais_origem_options.keys()))
pais_origem = pais_origem_options[pais_origem_escolhido]

# Botão para realizar a predição
if st.button("Verificar Fraude"):
    # Constroi o array/reshape adequado para o modelo
    input_array = np.array([[valor_transacao, tempo_conta, num_transacoes_ult_30_dias, pais_origem]])

    # Realiza a predição
    prediction = model.predict(input_array)
    # Realiza a predição de probabilidade
    prediction_proba = model.predict_proba(input_array)

    # Exibe o resultado
    st.write("Resultado da Predição:")
    if prediction[0] == 1:
        st.error("Fraude detectada!")
    else:
        st.success("Transação legítima.")

    # Exibe a probabilidade
    st.write("Probabilidades:")
    st.write(f"**Probabilidade de fraude:** {prediction_proba[0][1]:.2f}")
    st.write(f"**Probabilidade de transação legítima:** {prediction_proba[0][0]:.2f}")
else:
    st.write("Clique em 'Verificar Fraude' para ver o resultado.")

# Exibe informações adicionais
st.header("Informações Adicionais")
st.write("Este modelo foi treinado com dados simulados e pode não refletir a realidade.")
