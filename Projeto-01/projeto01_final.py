# Projeto 1 - Classificação de Texto com Aprendizagem Supervisionada

import re                                                       # para expressão regular
import praw                                                     # para conexão com a API do reddit   
import config
import numpy as np
from sklearn.model_selection import train_test_split            # dividir dados de treino e teste
from sklearn.feature_extraction.text import TfidfVectorizer     # preparar matriz com dados de texto
from sklearn.decomposition import TruncatedSVD                  # reduzir a dimensionalidade
# algoritmos de machine learning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV

from sklearn.metrics import classification_report               # imprimir as métricas de classficação dos modelos
from sklearn.pipeline import Pipeline                           # sequência de desenvolvimento
from sklearn.metrics import confusion_matrix                    # imprimir modelo e avaliar performance
# visualização dos dados
import matplotlib.pyplot as plt
import seaborn as sns

## Carregando os Dados

# Lista de temas que usaremos para buscas no Reddit. 
# Essas serão as classes que usaremos como variável target
assuntos = ["datascience", "machinelearning", "physics", "astrology", "conspiracy"]
quantidade_min_char_post = 100
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
    # print("Usuário logado!")
    # print(api_reddit.user.me())
    
    return api_reddit


def carrega_dados():
    
    # Contamos o número de caracteres usando expressões regulares
    char_count = lambda post: len(re.sub('\W|\d', '', post.selftext))

    # Definimos a condição para filtrar os posts para posts apenas com mais de X caracteres
    mask = lambda post: char_count(post) >= quantidade_min_char_post

    # Listas para os resultados
    data = []
    labels = []

    for i, assunto in enumerate(assuntos):

        # Extrai os posts
        subreddit_data = api_reddit_connection().subreddit(assunto).new(limit = quantidade_post_minerar)

        # Filtra os posts que não satisfazem nossa condição definida em 'mask'
        posts = [post.selftext for post in filter(mask, subreddit_data)]

        data.extend(posts)
        labels.extend([i] * len(posts))

        # Print
        print(f"Número de posts do assunto r/{assunto}: {len(posts)}",
              f"\nUm dos posts extraídos: {posts[0][:tamanho_amostra_post]}...\n",
              "_" * 80 + '\n')
    
    return data, labels

## Divisão em Dados de Treino e Teste

# Variáveis de controle
TEST_SIZE = .2
RANDOM_STATE = 0


# Função para split dos dados
def split_data():

    print(f"Split {100 * TEST_SIZE}% dos dados para teste e avaliação do modelo...")
    
    # Split dos dados
    X_treino, X_teste, y_treino, y_teste = train_test_split(data, 
                                                            labels, 
                                                            test_size = TEST_SIZE, 
                                                            random_state = RANDOM_STATE)

    print(f"{len(y_teste)} amostras de teste.")
    
    return X_treino, X_teste, y_treino, y_teste

## Pré-Processamento de Dados e Extração de Atributos

# - Remove símbolos, números e strings semelhantes a url com pré-processador personalizado
# - Vetoriza texto usando o termo frequência inversa de frequência de documento
# - Reduz para valores principais usando decomposição de valor singular
# - Particiona dados e rótulos em conjuntos de treinamento / validação

# Variáveis de controle
MIN_DOC_FREQ = 1
N_COMPONENTS = 1000
N_ITER = 30

# Função para o pipeline de pré-processamento
def preprocessing_pipeline():
    
    # Remove caracteres não "alfabéticos"
    pattern = r'\W|\d|http.*\s+|www.*\s+'
    preprocessor = lambda text: re.sub(pattern, ' ', text)

    # Vetorização TF-IDF
    vectorizer = TfidfVectorizer(preprocessor = preprocessor, stop_words = 'english', min_df = MIN_DOC_FREQ)

    # Reduzindo a dimensionalidade da matriz TF-IDF 
    decomposition = TruncatedSVD(n_components = N_COMPONENTS, n_iter = N_ITER)
    
    # Pipeline
    pipeline = [('tfidf', vectorizer), ('svd', decomposition)]

    return pipeline

## Seleção do Modelo

# Variáveis de controle
N_NEIGHBORS = 4
CV = 3

# Função para criar os modelos
def cria_modelos():

    modelo_1 = KNeighborsClassifier(n_neighbors = N_NEIGHBORS)
    modelo_2 = RandomForestClassifier(random_state = RANDOM_STATE)
    modelo_3 = LogisticRegressionCV(cv = CV, random_state = RANDOM_STATE)

    modelos = [("KNN", modelo_1), ("RandomForest", modelo_2), ("LogReg", modelo_3)]
    
    return modelos

## Treinamento e Avaliação dos Modelos

# Função para treinamento e avaliação dos modelos
def treina_avalia(modelos, pipeline, X_treino, X_teste, y_treino, y_teste):
    
    resultados = []
    
    # Loop
    for name, modelo in modelos:

        # Pipeline
        pipe = Pipeline(pipeline + [(name, modelo)])

        # Treinamento
        print(f"Treinando o modelo {name} com dados de treino...")
        pipe.fit(X_treino, y_treino)

        # Previsões com dados de teste
        y_pred = pipe.predict(X_teste)

        # Calcula as métricas
        report = classification_report(y_teste, y_pred)
        print("Relatório de Classificação\n", report)

        resultados.append([modelo, {'modelo': name, 'previsoes': y_pred, 'report': report,}])           

    return resultados

## Executando o Pipeline Para Todos os Modelos

# Pipeline de Machine Learning
if __name__ == "__main__":
    
    # Carrega os dados
    data, labels = carrega_dados()
    
    # Faz a divisão
    X_treino, X_teste, y_treino, y_teste = split_data()
    
    # Pipeline de pré-processamento
    pipeline = preprocessing_pipeline()
    
    # Cria os modelos
    all_models = cria_modelos()
    
    # Treina e avalia os modelos
    resultados = treina_avalia(all_models, pipeline, X_treino, X_teste, y_treino, y_teste)

print("Concluído com Sucesso!")

## Visualizando os Resultados

def plot_distribution():
    _, counts = np.unique(labels, return_counts = True)
    sns.set_theme(style = "whitegrid")
    plt.figure(figsize = (15, 6), dpi = 120)
    plt.title("Número de Posts Por Assunto")
    sns.barplot(x = assuntos, y = counts)
    plt.legend([' '.join([f.title(),f"- {c} posts"]) for f,c in zip(assuntos, counts)])
    plt.show()

def plot_confusion(result):
    print("Relatório de Classificação\n", result[-1]['report'])
    y_pred = result[-1]['previsoes']
    conf_matrix = confusion_matrix(y_teste, y_pred)
    _, test_counts = np.unique(y_teste, return_counts = True)
    conf_matrix_percent = conf_matrix / test_counts.transpose() * 100
    plt.figure(figsize = (9,8), dpi = 120)
    plt.title(result[-1]['modelo'].upper() + " Resultados")
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


# Fim


