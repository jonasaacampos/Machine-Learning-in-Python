import math
import numpy as np  # converte dados para uma matriz
import pandas as pd  # converte a matriz em um dataframe
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression  # cria modelo
from sklearn.model_selection import train_test_split  # treina modelo

# para filtrar ruídos do modelo
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings("ignore")


def load_data(fileData):
    df = pd.read_csv(fileData)
    print(
        f"""
         \n
         Dados carregados com sucesso!
         Shape = {df.shape}
         ({df.shape[0]} linhas X {df.shape[1]} colunas)
         """
    )
    print(df.head())
    return df


def chart_generate(df, x, y, title):
    """
    Gera um gráfico de dispersão e salva-o como arquivo de imagem.
    O objetivo é ter uma visão geral dos dados
    
    Args:
        df (pandas.DataFrame): dataframe com os dados
        x (pandas.Series): Coluna do dataframe com o eixo X
        y (pandas.Series): Coluna do dataframe com o eixo y
        title (str): Título do gráfico
    """    
    df.plot(x=x, y=y, style="o")
    plt.title(title)
    plt.xlabel(x)
    plt.xticks(rotation=90)
    plt.ylabel(y)
    plt.show()
    plt.savefig("img/pt01-fig01.png")


def data_preparation(df):
    """
    - Recebe um DataFrame oriundo do tratamento da função load_data()
    - separa e retorna as variáveis X e y
    X => variável independente ('eu controlo')
    y => variável dependente ('muda em relação à alteração da variável controle')

    'a variável independente é controlada pelo experimentador, 
    enquanto o valor da variável dependente só muda em resposta à variável independente'
    Saiba mais em:
    https://datascience.eu/pt/matematica-e-estatistica/uma-atualizacao-na-analise-de-regressao/

    Args:
        df (pandas.Series): Colunas X e y de um DataFrame pandas

    Returns:
        tuple: Valores e X e y
    """    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, 1].values
    return (X, y)


def show_coef_intercept():
    """Coeficiente angular e interceptação em y de uma equação
    Returns:
        str: valores formatados para coeficientes B0 e B1
    """    
    return f"B1 (coef_)     : {modelo.coef_}\nB0 (intercept_): {modelo.intercept_}"


arquivo = "Projeto-02/data/dataset.csv"
df = load_data(arquivo)
chart_generate(df, "Investimento", "Retorno", "Investimento X Retorno")

X, y = data_preparation(df)

TEST_SIZE = 0.3
RANDOM_STATE = 0

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

## Construção do modelo
modelo = LinearRegression()

# treinamento do modelo
modelo.fit(X_treino, y_treino)
print("\nModelo treinado com sucesso!")

print(show_coef_intercept())

# Plot Regressão Linear
# y = B1 * X + B0
regression_line = modelo.coef_ * X + modelo.intercept_
plt.scatter(X, y)
plt.title("Investimento x Retorno")
plt.xlabel("Investimento")
plt.ylabel("Retorno")
plt.plot(X, regression_line, color="black")
plt.savefig("img/parte1-linha-regressao.png")
plt.show()

# Previsão com dados de teste
y_pred = modelo.predict(X_teste)

# Real X Previsto

print(
    """
      --- Resultado da previsão dos investimentos ---
      """
)

df_valores = pd.DataFrame({"Real": y_teste, "Previsto": y_pred})
print(df_valores)

# plot
fig, ax = plt.subplots()
index = np.arange(len(X_teste))
bar_width = 0.35
actual = plt.bar(index, df_valores["Real"], bar_width, label="Valor Real")
plt.xlabel("Investimento")
plt.ylabel("Retorno")
plt.title("Valor Real X Previsto")
<<<<<<< HEAD
plt.xticks(index + bar_width, X_teste, rotation=90)
=======
#plt.xticks(index + bar_width, X_teste, rotation=90)
>>>>>>> 361bf57 (🐛 rotação de eixo x atualizado toDo readme)
plt.legend()
plt.savefig("img/part01-atual-vs-previsto.png")
plt.show()

def avaliacao_do_modelo(y_teste, y_pred):
    print(f"""                      Avaliação do modelo
            .......................................................................
            
            MAE.........(Mean Abolute Error): {mean_absolute_error(y_teste, y_pred)}
            MSE.........(Mean Squared Error): {mean_squared_error(y_teste, y_pred)}
            RMSE...(Root Mean Squared Error): {math.sqrt(mean_squared_error(y_teste, y_pred))}
            R2 Score........................: {r2_score(y_teste, y_pred)}

            .......................................................................
          """)


avaliacao_do_modelo(y_teste, y_pred)

## inserindo dados desconhecidos...
print("------------------------")
print("Previsão para retorno com novos dados")

## assim como treinamos o modelo, entregamos a ele os dados no mesmo formato
## os mesmos usados na transformação e análise dos dados de teste para treino do modelo
input_investimento = float(input("Digite o valor do investimento: ").strip())
investimento = np.array([input_investimento])
investimento = investimento.reshape(-1, 1)


# previsões
pred_score = modelo.predict(investimento)
print(
    f"""
        Investimento realizado.....: {input_investimento:.2f}
        Retorno previsto...........: {pred_score[0]:.2f}
      """
)
