import numpy as np  # Importa a biblioteca numpy para operações com arrays e cálculos numéricos
import matplotlib.pyplot as plt  # Importa o pyplot do matplotlib para criação e exibição de gráficos


class KNNClassifier:
    """
    Classe que implementa o algoritmo KNN para classificação.

    O classificador armazena os dados de treino e permite a previsão ds classes de novos pontos
    com base na votação dos k vizinhos mais próximos. Pode também normalizar os dados antes de calcular
    as distâncias, garantindo que todas as features tenham a mesma influência.
    """

    def __init__(self, k=3, normalize=True):
        """
        Inicia a instância do classificador KNN.

        Parametros:
        -----------
        k : int
            Número de vizinhos a serem considerados na classificação.
        normalize : bool
            Se True, os dados serão normalizados (min-max scaling) antes da classificação.
        """
        self.k = k
        self.normalize = normalize
        self.X_train = None
        self.y_train = None
        self.min = None  # Valores mínimos de cada feature para normalização
        self.max = None  # Valores máximos de cada feature para normalização

    def fit(self, X, y):
        """
        Treina o classificador armazenando os dados de treino.
        Se a normalização estiver ativada, os dados são transformados utilizando o min-max scaling.

        Parâmetros:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Dados de treino.
        y : array-like, shape = [n_samples]
            labels correspondentes aos dados de treino.
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        if self.normalize:
            # Calcula os valores mínimos e máximos de cada feature
            self.min = self.X_train.min(axis=0)
            self.max = self.X_train.max(axis=0)
            diff = self.max - self.min
            diff[diff == 0] = 1  # Evita divisão por zero para features constantes
            # Normaliza os dados para o intervalo [0, 1]
            self.X_train = (self.X_train - self.min) / diff

    def predict(self, X):
        """
        Prevê as labels para os dados de teste com base nos dados de treino.

        Parâmetros:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Dados para os quais se deseja prever as labels.

        Retorna:
        --------
        predictions : numpy array, shape = [n_samples]
            Vetor com labels preditas para cada amostra.
        """
        X = np.array(X)  # Dados para os quais se deseja prever as labels
        if self.normalize:
            # Normaliza os dados de teste utilizando os mesmos parâmetros dos dados de treino
            diff = self.max - self.min
            diff[diff == 0] = 1
            X = (X - self.min) / diff

        predictions = []
        for x in X:
            # Calcula a distância Euclidiana entre o ponto x e todos os pontos de treino
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            # Obtém os índices dos k vizinhos mais próximos (ordenando as distâncias)
            # Suponha que 'distances' seja uma lista ou um array contendo as distâncias calculadas.
            indices = list(
                range(len(distances))
            )  # Cria uma lista de índices de 0 até o tamanho de distances - 1.
            indices.sort(
                key=lambda i: distances[i]
            )  # Ordena os índices com base nos valores correspondentes em 'distances'.
            k_indices = indices[
                : self.k
            ]  # Seleciona os 'k' primeiros índices (os que correspondem às menores distâncias).
            # Recolhe as labels dos k vizinhos
            k_nearest_labels = self.y_train[k_indices]
            # Realiza a votação maioritária: a label que aparece com maior frequência é escolhido
            # Inicializa um dicionário para contar as ocorrências de cada label em k_nearest_labels
            count_dict = {}
            for label in k_nearest_labels:
                if label in count_dict:
                    count_dict[
                        label
                    ] += 1  # Incrementa a contagem se a label já estiver presente
                else:
                    count_dict[label] = (
                        1  # Adiciona a label com contagem 1 se for a primeira ocorrência
                    )

            # Encontra a label com a maior contagem manualmente
            max_label = None
            max_count = -1
            for label, count in count_dict.items():
                if count > max_count:
                    max_count = count  # Atualiza a contagem máxima
                    max_label = label  # Atualiza a label correspondente

            # Adiciona a label com a maior contagem à lista de predições
            predictions.append(max_label)
        return np.array(predictions)


def generate_two_class_data(n_points):
    """
    Gera dados sintéticos para um exemplo com duas classes.

    Parâmetros:
    -----------
    n_points : int
        Número de pontos a serem gerados para cada classe.

    Retorna:
    --------
    X : numpy array, shape = [2*n_points, 2]
        Dados gerados para as duas classes.
    y : numpy array, shape = [2*n_points]
        labels dos dados (0 para a classe 0 e 1 para a classe 1).
    """
    # Classe 0: Pontos centrados em (0, 0)
    X0 = np.random.randn(n_points, 2) + np.array([0, 0])
    y0 = np.zeros(n_points)
    # Classe 1: Pontos centrados em (5, 5)
    X1 = np.random.randn(n_points, 2) + np.array([5, 5])
    y1 = np.ones(n_points)
    # Combina os dados e as labels de ambas as classes
    # Supondo que X0, X1, y0 e y1 são arrays do NumPy já definidos

    # Para os dados (X):
    # Converte X0 e X1 para listas
    X0_list = X0.tolist()  # Converte o array X0 numa lista de listas
    X1_list = X1.tolist()  # Converte o array X1 numaa lista de listas
    # Concatena as listas (juntando os elementos de X0_list e X1_list)
    X_combined = X0_list + X1_list
    # Converte a lista concatenada de volta para um array do NumPy
    X = np.array(X_combined)

    # Para as labels (y):
    # Converte y0 e y1 para listas
    y0_list = y0.tolist()  # Converte o array y0 numa lista
    y1_list = y1.tolist()  # Converte o array y1 enuma lista
    # Concatena as listas (juntando os elementos de y0_list e y1_list)
    y_combined = y0_list + y1_list
    # Converte a lista concatenada de volta para um array do NumPy
    y = np.array(y_combined)
    return X, y


def generate_three_class_data(n_points):
    """
    Gera dados sintéticos para um exemplo com três classes.

    Parâmetros:
    -----------
    n_points : int
        Número de pontos a serem gerados para cada classe.

    Retorna:
    --------
    X : numpy array, shape = [3*n_points, 2]
        Dados gerados para as três classes.
    y : numpy array, shape = [3*n_points]
        labels dos dados (0, 1 e 2 para as três classes).
    """
    # Classe 0: Pontos centrados em (0, 0)
    X0 = np.random.randn(n_points, 2) + np.array([0, 0])
    y0 = np.zeros(n_points)
    # Classe 1: Pontos centrados em (5, 5)
    X1 = np.random.randn(n_points, 2) + np.array([5, 5])
    y1 = np.ones(n_points)
    # Classe 2: Pontos centrados em (0, 5)
    X2 = np.random.randn(n_points, 2) + np.array([0, 5])
    y2 = np.full(n_points, 2)  # Cria um array com o valor 2 para as labels da classe 2
    # Supondo que X0, X1, X2, y0, y1 e y2 já estão definidos como arrays do NumPy

    # Para os dados (X):
    # Converte cada array numa lista de listas
    X0_list = X0.tolist()  # Converte o array X0 numa lista
    X1_list = X1.tolist()  # Converte o array X1 numa lista
    X2_list = X2.tolist()  # Converte o array X2 numa lista

    # Concatena as listas (juntando os elementos de X0_list, X1_list e X2_list)
    X_combined = X0_list + X1_list + X2_list

    # Converte a lista concatenada de volta para um array do NumPy
    X = np.array(X_combined)

    # Para as labels (y):
    # Converte cada array de labels numa lista
    y0_list = y0.tolist()  # Converte o array y0 numa lista
    y1_list = y1.tolist()  # Converte o array y1 numa lista
    y2_list = y2.tolist()  # Converte o array y2 numa lista

    # Concatena as listas de labels
    y_combined = y0_list + y1_list + y2_list

    # Converte a lista concatenada de volta para um array do NumPy
    y = np.array(y_combined)
    return X, y


def plot_data(X, y, title):
    """
    Plota os dados num gráfico de dispersão.

    Parâmetros:
    -----------
    X : numpy array, shape = [n_samples, 2]
        Dados a serem plotados.
    y : numpy array, shape = [n_samples]
        labels dos dados, que determinam as cores dos pontos.
    title : str
        Título do gráfico.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor="k", cmap=plt.cm.RdYlBu)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


def plot_decision_boundary(knn, X, y, title):
    """
    Plota a fronteira de decisão do modelo KNN juntamente com os dados.

    Parâmetros:
    -----------
    knn : objeto KNNClassifier
        O modelo treinado.
    X : numpy array, shape = [n_samples, 2]
        Dados utilizados para gerar a grade de pontos.
    y : numpy array, shape = [n_samples]
        labels dos dados, para sobrepor os pontos ao gráfico.
    title : str
        Título do gráfico.
    """
    h = 0.1  # Tamanho do passo para a grade
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    # Obtém as previsões para cada ponto da grade
    Z = knn.predict(grid_points)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor="k", cmap=plt.cm.RdYlBu)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


def main():
    """
    Função principal para execução dos exemplos do modelo KNN.

    Esta função solicita os parâmetros ao usuário, gera os dados sintéticos,
    treina o modelo KNN e exibe os gráficos dos dados gerados e da fronteira de decisão.
    O utilizador pode escolher entre um exemplo com duas classes ou com três classes.
    Além disso, é apresentado um exemplo adicional de previsão para novos pontos.
    """
    # --------------------- Entrada de Parâmetros ---------------------
    n_points_input = input("Entre com a quantidade de pontos para cada classe: ")
    try:
        n_points = int(n_points_input)
    except ValueError:
        print("Entrada inválida. utilizando 50 pontos por classe como padrão.")
        n_points = 50

    k_input = input("Indique o número de vizinhos (k): ")
    try:
        k = int(k_input)
    except ValueError:
        print("Entrada inválida. Usando k = 3 como padrão.")
        k = 3

    norm_input = input("Normalizar dados? (yes/no): ").strip().lower()
    if norm_input in ["yes", "y"]:
        normalize = True
    elif norm_input in ["no", "n"]:
        normalize = False
    else:
        print("Entrada inválida. Utilizando normalização por padrão.")
        normalize = True

    # Escolha do exemplo: 1 para duas classes ou 2 para três classes
    example_choice = input(
        "Escolha o exemplo (1 - duas classes, 2 - três classes): "
    ).strip()
    if example_choice == "2":
        X, y = generate_three_class_data(n_points)
        example_title = "Exemplo com Três Classes"
    else:
        X, y = generate_two_class_data(n_points)
        example_title = "Exemplo com Duas Classes"

    # --------------------- Exibição dos Gráficos ---------------------
    # 1. Gráfico dos dados gerados (antes do modelo)
    plot_data(X, y, f"{example_title} - Dados Gerados (antes do modelo)")

    # 2. Treino do modelo KNN e plot da fronteira de decisão
    knn = KNNClassifier(k=k, normalize=normalize)
    knn.fit(X, y)
    plot_decision_boundary(
        knn,
        X,
        y,
        f"{example_title} - Fronteira de Decisão (k = {k}, Normalização = {normalize})",
    )

    # --------------------- Exemplo Adicional de Previsão ---------------------
    # Para demonstrar a versatilidade do modelo, preveja as labels para os novos pontos arbitrários.
    new_points = np.array([[1, 1], [3, 3], [0, 4], [6, 4]])
    predictions = knn.predict(new_points)
    print("Previsões para novos pontos:", predictions)

    # Plota os novos pontos sobre a fronteira de decisão para visualização
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = knn.predict(grid_points)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor="k", cmap=plt.cm.RdYlBu)

    # Plota os novos pontos em destaque (marcados com "x" e cor preta)
    plt.scatter(
        new_points[:, 0],
        new_points[:, 1],
        c="red",
        s=100,
        marker="x",
        label="Novos Pontos",
    )
    plt.title("Exemplo Adicional: Previsão para Novos Pontos")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()  # Executa a função principal quando o script é executado diretamente
