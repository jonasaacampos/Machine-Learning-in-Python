<p align="center">
	  <a href='https://jonasaacampos.github.io/portfolio/'>
      <img alt="Engenheiro de Machine Learning - Badge" src="https://img.shields.io/static/v1?color=red&label=Engenieer&message=Machine-Learning&style=for-the-badge&logo=ia"/>
      </a>
</p>

<img alt="brain" src="img/brain.png" width=150 align=right>

<h1>Machine Learning em Python e C++</h1>

Anotações e projetos do curso de formação em Engenharia de Machine Learning.

## O papel do Engenheiro de Machine Learning na contrução de modelos

Normalmente o trabalho do Engnheiro de Machine Learning está mais relacionado com a publicação do que com a construção do modelo. Para tanto é necessário algum conhecimento para criar modelos, para que consiga fazer o deploy do modelo.

A concepção do modelo, geralmente fica a cargo do Cientista de Dados.

## Exemplo de problemas de Machine Learning

Marque abaixa quais problemas podemos solucionar com Machine Learning:

- [ ] Calcular total de vendas a partir de dados de transações financeiras
  - basta uma operação matemática. Não é necessária
- [x] Prever o toral de vendas a partir de dados de transações financeiras
  - meu modelo fará uma compreensão dos dados envolvidos (modelagem preditiva)
- [x] Detectar padrões nas vendas a partir de transações financeiras
  - Aqui meu modelo fará agrupamentos de tipos de conas para determinar padrões.

> Machine Learning é aplicável somente não existe uma operação matemática para prever o resultado
> 
> Machine Learning não CRIA padrões, mas os detecta

## Definição de modelo de machine learning

Todo modelo de machine learning é uma formulação matemática que tenta explicar o relacionamento entre dados.

Usa-se um algoritmo com um rol de hipóteses de resposta, e durante o treinamento com os dados, o modelo descobre parâmetros que confirmam ou explicam o relacionamento dos dados caso exista. (Função alvo > espaço de hipóteses > dados > verificação de relacionamentos)

Existem algoritmos diferentes para problemas diferentes e para dados diferentes.

## Riscos e Oportunidade

Riscos
- Dados Insuficientes
- Não possuir desempenho computacional
- Presissas inválidas

Oportunidades
- Quanto maior o risco, maior as oportunidades
- Computação Quântica vai ocasionar um novo salto na área

## Framework

> utilizar framework facilita a vida. Crie um modelo do zero somente se não existir nenhum algoritmo que lhe atenda,

### Principais frameworks de Machine Learning

- TensorFlow e Keras
- Scikit-Learn (o mais amigável)
- Pytorch (pesquisa)
- Spark MLLib (computação distribuída)
- mxnet (nuvem aws)
- mlpack (C++)
- DeepLearning4j (Java)

## Modelos de MAchine Learning

### Modelos de Classificação:

Objetivo de classificar os dados em pelo menos 2 categorias
- Classificação Binária
- Classificação Multiclasse
- Multi-Label Classification (marcar elementos em imagem)
- Classificação Desbalanceada

### Modelos de Regressão

Servem para prever um valor numérico

- Regressão linear simples
- Regressão linear múltipla
- Regressão polinomial
- Regressão Ridge
- Regressão LASSO


### Modelos de Clustering

Diferentemente dos outros modelos, neste caso não temos dados de saída. Servem para agrupar melhor os dados.

### Modelos de Associação

Serve para encontrar padrões entre os dados. **Associação serve para recomendação**.

### Modelos de Deep Learning

IA é um nome comercial. Deep Learning é o resultado de análises estatísticas e expressões matemáticas.

## Para saber mais

- Machine Learning, Tom Michael
- Machine Learning with Python for Everyone, Mark Fenne
- The Hundred-Page Machine Learning Book, Andriy Burkov

