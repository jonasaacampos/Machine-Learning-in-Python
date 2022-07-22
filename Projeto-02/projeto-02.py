import math
import numpy as np  # converte dados para uma matriz
import pandas as pd  # converte a matriz em um dataframe
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression  # cria modelo
from sklearn.model_selection import train_test_split  # treina modelo

# para filtrar ruídos do modelo
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score

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
    df = df
    df.plot(x=x, y=y, style="o")
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.savefig("img/pt01-fig01.png")


def data_preparation(df):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, 1].values
    return (X, y)


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

# coeficientes B0 e B1
print(
    f"""
      B1 (coef_)     : {modelo.coef_}
      B0 (intercept_): {modelo.intercept_}
      """
)

# Plot Regressão Linear
# y = B1 * X + B0
regression_line = modelo.coef_ * X + modelo.intercept_
plt.scatter(X, y)
plt.title("Investimento x Retorno")
plt.xlabel("Investimento")
plt.ylabel("Retorno")
plt.plot(X, regression_line, color="black")
plt.savefig("img/parte1-linha-regressao.png")
# plt.show()

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
