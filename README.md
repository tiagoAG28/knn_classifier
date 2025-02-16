
# knn_classifier
=======
KNN Classifier

##Descrição
Este projeto implementa um classificador KNN (K-Nearest Neighbors) em Python.  
O algoritmo KNN classifica novos pontos com base nos **k** vizinhos mais próximos dos dados de treino, utilizando a votação maiotária.  
Além disso, o projeto permite:

- **Definir o número de vizinhos (neighbors)** a serem considerados;
- **Indicar se os dados devem ser normalizados** (usando escalonamento min-max) antes da classificação;
- **Definir a quantidade de pontos** a serem gerados para cada classe num dataset sintético;
- **Escolher entre dois exemplos** de aplicação:
  1. Exemplo com duas classes;
  2. Exemplo com três classes;
- **Visualizar gráficos** que demonstram:
  - Os dados gerados (antes do modelo);
  - A fronteira de decisão resultante após a aplicação do modelo KNN;
- **Observar um exemplo adicional de previsão** para novos pontos, com exibição das coordenadas e da classe prevista.

## Funcionalidades

- **Implementação do algoritmo KNN:**  
  A classe `KNNClassifier` permite treinar o modelo com dados de entrada e prever as labels de novos pontos com base nos vizinhos mais próximos.  
  _Opção de normalização:_ Os dados podem ser normalizados via min-max scaling, para que todas as features tenham a mesma influência no cálculo das distâncias.

- **Geração de dados sintéticos:**  
  Duas funções auxiliares geram conjuntos de dados com:

  - Duas classes, ou
  - Três classes.

- **Visualização dos resultados:**  
  São exibidos gráficos de dispersão dos dados gerados e da fronteira de decisão calculada pelo modelo.  
  Também é apresentada uma previsão para novos pontos, mostrando suas coordenadas e a classe atribuída.

## Estrutura do Código

- **KNNClassifier**

  - `__init__(self, k=3, normalize=True)`: Inicializa o modelo definindo o número de vizinhos e a opção de normalização.
  - `fit(self, X, y)`: Armazena os dados de treino (realizando normalização se necessário).
  - `predict(self, X)`: Prediz as labels para novos pontos, calculando distâncias, selecionando os k vizinhos e votando pela classe maioritária.

- **Funções Auxiliares:**
  - `generate_two_class_data(n_points)`: Gera um dataset sintético com duas classes (ex.: pontos centrados em (0,0) e (5,5)).
  - `generate_three_class_data(n_points)`: Gera um dataset sintético com três classes (ex.: pontos centrados em (0,0), (5,5) e (0,5)).
  - `plot_data(X, y, title)`: Plota os dados gerados em um gráfico de dispersão.
  - `plot_decision_boundary(knn, X, y, title)`: Plota a fronteira de decisão do modelo KNN, sobrepondo os dados.

## Requisitos

- **Python 3.x**
- **NumPy**
- **Matplotlib**

## Instalação

Para instalar as bibliotecas necessárias, execute:

```bash
pip install numpy matplotlib
```

#Como usar
###Clonar este repositório para a sua máquina local

git clone https://github.com/tiagoAG28/knn-classifier.git
cd knn-classifier

###Executar o script knn_classifier.py

python knn_classifier.py

O script irá solicitar ao utilizador os seguintes parâmetros:

Número de pontos para cada classe: Define a quantidade de pontos a serem gerados para cada classe.
Número de vizinhos (k): Define o número de vizinhos a serem considerados no KNN.
Normalizar dados?: Se yes, os dados serão normalizados.
Escolha do exemplo: Se 1, gera dados com duas classes; se 2, gera dados com três classes.

#Exemplo de Output
Gráficos Gerados:
Dados Gerados: Gráfico com a distribuição dos pontos de cada classe.
Fronteira de Decisão: Gráfico mostrando a fronteira de decisão do modelo KNN.
Previsões para Novos Pontos: Visualização das previsões do modelo para novos pontos.

Exemplo de previsões:

Previsões para novos pontos: [0 1 0 1]

