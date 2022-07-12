# Projeto 1 - Classificação de Texto com Aprendizagem Supervisionada

# Pacotes
# para expressão regular
import re
# para conexão com a API do reddit
import praw
import config
import numpy as np
# dividir dados de treino e teste
from sklearn.model_selection import train_test_split
# preparar matriz com dados de texto
from sklearn.feature_extraction.text import TfidfVectorizer
# reduzir a dimensionalidade
from sklearn.decomposition import TruncatedSVD
# algoritmo de machine learning
from sklearn.neighbors import KNeighborsClassifier
# algoritmo de machine learning
from sklearn.ensemble import RandomForestClassifier
# algoritmo de machine learning
from sklearn.linear_model import LogisticRegressionCV
# imprimir as métricas de classficação dos modelos
from sklearn.metrics import classification_report
# sequência de desenvolvimento
from sklearn.pipeline import Pipeline
# imprimir modelo e avaliar performance
from sklearn.metrics import confusion_matrix
# visualização dos dados
import matplotlib.pyplot as plt
# visualização dos dados
import seaborn as sns

# Carregando os Dados

# Lista de temas que usaremos para buscas no Reddit.
# Essas serão as classes que usaremos como variável target
assuntos = ['politica', 'partidos', 'lula',
            'bolsonaro', 'conspiração', 'comunismo']
tamanho_min_post = 100
quantidade_post_minerar = 1000  # limite de 1000 de acordo com termos de uso da api


def apiRedditConection():
    api_reddit = praw.Reddit(client_id="8ZXmCNi4cHYXh5FyVCZI7Q",
                             client_secret="VuRph69q7ai4m9CC-qrXh9YRag3mtg",
                             password="bele2012",
                             user_agent="jaac-script-app",
                             username="Vegetable-Carrot7306")
    # verifica se conexão foi realizada com sucesso
    print('Usuário logado!')
    print(api_reddit.user.me())
    return(api_reddit)


def carregaDados():

    # Contamos o número de caracteres usando expressões regulares
    def char_count(post): return len(re.sub('\W|\d', '', post.selftext))

    # Definimos a condição para filtrar os posts (retornaremos somente posts com 100 ou mais caracteres)
    def mask(post): return char_count(post) >= tamanho_min_post

    # Listas para os resultados
    data = []
    labels = []

    # Loop
    for i, assunto in enumerate(assuntos):

        # Extrai os posts
        subreddit_data = apiRedditConection().subreddit(assunto).new(limit = quantidade_post_minerar)

        # Filtra os posts que não satisfazem nossa condição
        posts = [post.selftext for post in filter(mask, subreddit_data)]

        # Adiciona posts e labels às listas
        data.extend(posts)
        labels.extend([i] * len(posts))

        # Print
        print(f"Número de posts do assunto {assunto}: {len(posts)}", f"\nUm dos posts extraídos: {posts[0][:80]}...\n",
              "_" * 80 + '\n')

    return data, labels


carregaDados()

print("done!")