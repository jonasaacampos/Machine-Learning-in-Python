<p align="center">
	  <a href='https://jonasaacampos.github.io/portfolio/'>
      <img alt="Engenheiro de Machine Learning - Badge" src="https://img.shields.io/static/v1?color=red&label=Engenieer&message=Machine-Learning&style=for-the-badge&logo=ia"/>
      </a>
</p>

<img alt="brain" src="img/brain.png" width=150 align=right>

<h1>Prevendo o retorno financeiro de investimentos em títulos públicos</h1>

Construção de modelos de machine learning

- 01 modelo utilizando framework
- 01 modelo construído do zero

## Definição do projeto

Dado o valor de um investimento, qual o retorno possível?

$$ Taxa De Retorno={Valor Atual - Valor Original\over{ValorOriginal}} * 100 $$

Este é um exemplo de que NÃO PRECISAMOS de machine learning. Se eu tenho os dados e a fórmula matemática, basta usá-los.

Se eu não souber a fórmula matemática que determina a relação entre os dados, então o aprendizado de máquina É IDEAL, pois ela descobrirá de forma aproximada qual função matemática melhor de aplica para o mesmo resultado de uma fórmula matemática conhecida.

## Regressão Linear

> A análise de regressão linear é usada para prever o valor de uma variável com base no valor de outra. A variável que deseja prever é chamada de variável dependente. A variável que é usada para prever o valor de outra variável é chamada de variável independente. (IBM, 2022)

$$ Y = b0 + b1 * x $$

```
Y = variável dependente
X = variável independente
b0 = intercept
b1 = coeficiente

b0 e b1 são estimados durante o treinamento com dados históricos
```
## To do
- [ ] Fazer funções para organizar código
- [ ] 
## Para saber mais

- IBM, 2022. [Regressão linear - Gere previsões usando uma fórmula matemática facilmente interpretada.](https://www.ibm.com/br-pt/analytics/learn/linear-regression)
- 