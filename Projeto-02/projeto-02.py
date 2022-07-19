import math
import numpy as np                                    # converte dados para uma matriz
import pandas as pd                                   # converte a matriz em um dataframe
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression     # cria modelo
from sklearn.model_selection import train_test_split  # treina modelo
# para filtrar ruídos do modelo
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings("ignore")


def load_data(fileData):
   df = pd.read_csv(fileData)
   print(f'''
         \n
         Dados carregados com sucesso!
         Shape = {df.shape}
         ({df.shape[0]} linhas X {df.shape[1]} colunas)
         ''')
   print(df.head())


arquivo = 'Projeto-02/data/dataset.csv'


load_data(arquivo)
