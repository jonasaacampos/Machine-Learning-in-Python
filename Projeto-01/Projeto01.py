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
assuntos = ["datascience", "machinelearning", "physics", "astrology", "conspiracy"]
tamanho_min_post = 100
quantidade_post_minerar = 1000  # limite de 1000 de acordo com termos de uso da api
tamanho_amostra_post = 120


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
            api_reddit_connection()
            .subreddit(assunto)
            .new(limit=quantidade_post_minerar)
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
        f"Dividir {100 * TEST_SIZE}% dos dados para teste e avaliação do modelo..."
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
        preprocessor=clean_patterns, stop_words="english", min_df=MIN_DOC_FREQ
    )

    # Redução da dimensionalidade da matriz TF-IDF (maldição da dimensionalidade)
    # O PCA não seria ideal neste caso, pois aqui a tendência é da matriz ficar esparsa (cheia de zeros)
    # Truncade Singular Value Decomposition é o ideal para matrizes esparsas
    decomposition = TruncatedSVD(n_components=N_COMPONENTS, n_iter=N_ITER)

    pipeline = [("tfId", vectorizer), ("svd", decomposition)]

    return pipeline


# Seleção do Modelo
# Variáveis de controle
N_NEIGHBORS = 4
CV = 3
MAX_ITER = 3000


def make_models():
    """_summary_

    Returns:
        _type_: _description_
    """

    modelo_1 = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    modelo_2 = RandomForestClassifier(random_state=RANDOM_STATE)
    modelo_3 = LogisticRegressionCV(cv=CV, random_state=RANDOM_STATE, max_iter = MAX_ITER)

    modelos = [("KNN", modelo_1), ("RandomForest", modelo_2), ("LogReg", modelo_3)]

    return modelos


def train_test(modelos, pipeline, X_treino, X_teste, y_treino, y_teste):
    """_summary_"""

    resultados = []

    for nome, modelo in modelos:
        # treinamento
        pipe = Pipeline(pre_process_pipeline() + [(nome, modelo)])
        print(f"Treinando o modelo {nome}...")
        pipe.fit(X_treino, y_treino)

        # previsões
        y_pred = pipe.predict(X_teste)

        # cálculo das métricas
        report = classification_report(y_teste, y_pred)
        print(f"Relatório de Classificação\n{report}")

        resultados.append(
            [modelo, {"modelo": nome, "previsões": y_pred, "report": report}]
        )
    return resultados


# Executando o Pipeline Para Todos os Modelos
# Pipeline de Machine Learning


if __name__ == "__main__":

    data, labels = load_data()
    X_treino, X_teste, y_treino, y_teste = split_data()
    pipeline = pre_process_pipeline()
    all_models = make_models()
    resultados = train_test(all_models, pipeline, X_treino, X_teste, y_treino, y_teste)

    print("Finalizado com sucesso!")

# Visualização dos resultados


def plot_distribution():
    _, counts = np.unique(labels, return_counts=True)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 6), dpi=120)
    plt.title("Quantidade de postagens por assunto")
    sns.barplot(x = assuntos, y = counts)
    plt.legend(' '.join([f.title(), f' - {c} posts']) for f,c in zip(assuntos, counts))
    plt.show()


def plot_confusion(resultado):
    print(f'Relatório de Classificação\n', resultado[-1]['report'])
    y_pred = resultado[-1]['predictions']
    _, test_counts = np.unique(y_teste, return_counts = True)
    conf_matrix_percent = conf_matrix / test_counts.transpose() * 100
    plt.figure(figsize = (9,8), dpi = 120)
    plt.title(resultado[-1]['modelo'].upper() + " Resultados")
    plt.xlabel("Valor Real")
    plt.ylabel("Previsão do Modelo")
    ticklabels = [f"r/{sub}" for sub in assuntos]
    sns.heatmap(data = conf_matrix_percent, xticklabels = ticklabels, yticklabels = ticklabels, annot = True, fmt = '.2f')
    plt.show()
    

# Gráfico de avaliação
plot_distribution()

# Resultado do KNN
plot_confusion(resultados[0])

# Resultado do RandomForest
plot_confusion(resultados[1])

# Resultado da Regressão Logística
plot_confusion(resultados[2])