# Projeto 1 - Classificação de Texto com Aprendizagem Supervisionada

# Pacotes
# para expressão regular
import re

# para conexão com a API do reddit
import praw
import config
import numpy as np
from sklearn.model_selection import train_test_split        # dividir dados de treino e teste
from sklearn.feature_extraction.text import TfidfVectorizer # preparar matriz com dados de texto
from sklearn.decomposition import TruncatedSVD              # reduzir a dimensionalidade
from sklearn.neighbors import KNeighborsClassifier          # algoritmo de machine learning
from sklearn.ensemble import RandomForestClassifier         # algoritmo de machine learning
from sklearn.linear_model import LogisticRegressionCV       # algoritmo de machine learning
from sklearn.metrics import classification_report           # imprimir as métricas de classficação dos modelos
from sklearn.pipeline import Pipeline                       # sequência de desenvolvimento
from sklearn.metrics import confusion_matrix                # imprimir modelo e avaliar performance
import matplotlib.pyplot as plt                             # visualização dos dados
import seaborn as sns                                       # visualização dos dados

# Carregando os Dados

# Lista de temas que usaremos para buscas no Reddit.
# Essas serão as classes que usaremos como variável target
assuntos = ["datascience", "machinelearning", "physics", "astrology", "conspiracy"]
tamanho_min_post = 100
quantidade_post_minerar = 100  # limite de 1000 de acordo com termos de uso da api
tamanho_amostra_post = 80


def api_reddit_connection():
    api_reddit = praw.Reddit(
        client_id="8ZXmCNi4cHYXh5FyVCZI7Q",
        client_secret="VuRph69q7ai4m9CC-qrXh9YRag3mtg",
        password="bele2012",
        user_agent="jaac-script-app",
        username="Vegetable-Carrot7306",
    )
    # verifica se conexão foi realizada com sucesso
    print("Usuário logado!")
    print(api_reddit.user.me())
    return api_reddit

# Carregamento de dados
def load_data():
    # Contamos o número de caracteres usando expressões regulares
    def char_count(post):
        return len(re.sub("\W|\d", "", post.selftext))

    def filtroPostsMinChar(post):
        return char_count(post) >= tamanho_min_post

    # Listas para os resultados
    data = []
    labels = []

    # Loop
    for i, assunto in enumerate(assuntos):

        # Extrai os posts
        subreddit_data = (
            api_reddit_connection().subreddit(assunto).new(limit=quantidade_post_minerar)
        )

        # Filtra os posts que não satisfazem nossa condição
        posts = [post.selftext for post in filter(filtroPostsMinChar, subreddit_data)]

        # Adiciona posts e labels às listas
        data.extend(posts)
        labels.extend([i] * len(posts))

        # Print
        print(
            f"Número de posts do assunto {assunto}: {len(posts)}",
            f"\nUm dos posts extraídos: {posts[0][:tamanho_amostra_post]}...\n",
            "_" * 80 + "\n",
        )

    return data, labels


# Divisão de dados de treino e teste
# variáveis de controle para gerar o mesmo padrão de aletoriedade
TEST_SIZE = 0.2
RANDOM_STATE = 0

def split_data():

    print(
        f"Dividir {100 * TEST_SIZE}% dos dados para treinamento e avaliação do modelo..."
    )

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        data, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    quantidadeAmostrasDeTeste = len(y_teste)
    print(f"Foram criadas {quantidade_post_minerar} amostras de teste")

    return X_treino, X_teste, y_treino, y_teste


# Pré processamento de dados e Extração de atributos
# - Remove símbolos, números e strings semelhantes a url com pré-processador personalizado
# - Vetoriza texto usando o termo frequência inversa de frequência de documento
# - Reduz para valores principais usando decomposição de valor singular
# - Particiona dados e rótulos em conjuntos de treinamento / validação

# Variáveis de controle
MIN_DOC_FREQ = 1
N_COMPONENTS = 1000
N_ITER = 30


def pre_process_pipeline():

    pattern = r"\W|\d|http.*\s+|www.*\s+"

    def clean_patterns(text):
        return re.sub(pattern, "", text)

    # vetorização para matriz TF-IDF (faz a tokenização do texto, mas faz um rankeamento de importância)
    # TfidfVectorizer combina as opões de CountVectorizer e TfidfTransformer em um único modelo
    # TfidfVectorizer faz a tokenização, a contagem de palavras, normaliza, e retorna uma matriz
    vectorizer = TfidfVectorizer(
        preprocessor = clean_patterns, stop_words="english", min_df=MIN_DOC_FREQ
    )

    # Redução da dimensionalidade da matriz TF-IDF (maldição da dimensionalidade)
    # O PCA não seria ideal neste caso, pois aqui a tendência é da matriz ficar esparsa (cheia de zeros)
    # Truncade Singular Value Decomposition é o ideal para matrizes esparsas
    decomposition = TruncatedSVD(n_components=N_COMPONENTS, n_iter=N_ITER)

    pipeline = [("tfId", vectorizer), ("svd", decomposition)]

    return pipeline


## Seleção do Modelo
# Variáveis de controle
N_NEIGHBORS = 4
CV = 3

def make_models():

    """_summary_

    Returns:
        _type_: _description_
    """    

    modelo_1 = KNeighborsClassifier(n_neighbors = N_NEIGHBORS)
    modelo_2 = RandomForestClassifier(random_state = RANDOM_STATE)
    modelo_3 = LogisticRegressionCV(cv = CV, random_state = RANDOM_STATE)

    modelos = [("KNN", modelo_1), ("RandomForest", modelo_2), ("LogReg", modelo_3)]
    
    return modelos


