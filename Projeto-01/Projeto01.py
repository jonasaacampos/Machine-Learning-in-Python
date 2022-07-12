# Projeto 1 - Classificação de Texto com Aprendizagem Supervisionada

# Pacotes
import re                                                      # para expressão regular
import praw                                                    # para conexão com a API do reddit
import config
import numpy as np
from sklearn.model_selection import train_test_split           # dividir dados de treino e teste
from sklearn.feature_extraction.text import TfidfVectorizer    # preparar matriz com dados de texto
from sklearn.decomposition import TruncatedSVD                 # reduzir a dimensionalidade
from sklearn.neighbors import KNeighborsClassifier             # algoritmo de machine learning
from sklearn.ensemble import RandomForestClassifier            # algoritmo de machine learning
from sklearn.linear_model import LogisticRegressionCV          # algoritmo de machine learning
from sklearn.metrics import classification_report              # imprimir as métricas de classficação dos modelos
from sklearn.pipeline import Pipeline                          # sequência de desenvolvimento
from sklearn.metrics import confusion_matrix                   # imprimir modelo e avaliar performance
import matplotlib.pyplot as plt                                # visualização dos dados
import seaborn as sns                                          # visualização dos dados

## Carregando os Dados

# Lista de temas que usaremos para buscas no Reddit. 
# Essas serão as classes que usaremos como variável target
assuntos = ['politica', 'partidos', 'lula', 'bolsonaro', 'conspiração', 'comunismo']
tamanho_min_post = 100
quantidade_post_minerar = 100       #limite de 1000 de acordo com termos de uso da api

def apiRedditConection():
   api_reddit = praw.Reddit(client_id = "8ZXmCNi4cHYXh5FyVCZI7Q", 
                        client_secret = "VuRph69q7ai4m9CC-qrXh9YRag3mtg",
                        password = "bele2012",
                        user_agent = "jaac-script-app",
                        username = "Vegetable-Carrot7306")
   #verifica se conexão foi realizada com sucesso
   usuario = api_reddit.user.me()
   return(f"Usuário " + {usuario} + 'logado com sucesso')
   
   
def carregaDados():

   apiRedditConection()
   
  """  #conta caracteres  e dígitos dos posts
   char_count = lambda post: len(re.sub('\W|\d', '', post.selftext))
   #filtrar filtros para posts com mais de x caracteres
   mask = lambda post: char_count >= tamanho_min_post
   
   #resultadodos armazenados em listas
   data =[]
   labels = []
   
   for i, assunto in enumerate(assuntos):

      # Extrai os posts
      subreddit_data = api_reddit.subreddit(assunto).new(limit = 10)

      # Filtra os posts que não satisfazem nossa condição
      posts = [post.selftext for post in filter(mask, subreddit_data)]

      # Adiciona posts e labels às listas
      data.extend(posts)
      labels.extend([i] * len(posts))

   return data, labels """


carregaDados()

   
   

