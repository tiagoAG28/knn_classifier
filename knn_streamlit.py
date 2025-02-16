import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


# --------------------- Implementação do KNN ---------------------
class KNNClassifier:
    """
    Classe que implementa o algoritmo KNN para classificação.

    O classificador armazena os dados de treino e permite a previsão das classes de novos pontos
    com base na votação dos k vizinhos mais próximos. Pode também normalizar os dados antes de calcular
    as distâncias, garantindo que todas as features tenham a mesma influência.
    """

    def __init__(self, k=3, normalize=True):
        self.k = k
        self.normalize = normalize
        self.X_train = None
        self.y_train = None
        self.min = None  # Valores mínimos de cada feature para normalização
        self.max = None  # Valores máximos de cada feature para normalização

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        if self.normalize:
            self.min = self.X_train.min(axis=0)
            self.max = self.X_train.max(axis=0)
            diff = self.max - self.min
            diff[diff == 0] = 1  # Evita divisão por zero para features constantes
            self.X_train = (self.X_train - self.min) / diff

    def predict(self, X):
        X = np.array(X)
        if self.normalize:
            diff = self.max - self.min
            diff[diff == 0] = 1
            X = (X - self.min) / diff

        predictions = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            indices = list(range(len(distances)))
            indices.sort(key=lambda i: distances[i])
            k_indices = indices[: self.k]
            k_nearest_labels = self.y_train[k_indices]
            count_dict = {}
            for label in k_nearest_labels:
                if label in count_dict:
                    count_dict[label] += 1
                else:
                    count_dict[label] = 1

            max_label = None
            max_count = -1
            for label, count in count_dict.items():
                if count > max_count:
                    max_count = count
                    max_label = label
            predictions.append(max_label)
        return np.array(predictions)


# --------------------- Funções para Gerar Dados ---------------------
def generate_two_class_data(n_points):
    """
    Gera dados sintéticos para um exemplo com duas classes.
    """
    # Classe 0: Pontos centrados em (0, 0)
    X0 = np.random.randn(n_points, 2) + np.array([0, 0])
    y0 = np.zeros(n_points)
    # Classe 1: Pontos centrados em (5, 5)
    X1 = np.random.randn(n_points, 2) + np.array([5, 5])
    y1 = np.ones(n_points)

    X = np.concatenate((X0, X1), axis=0)
    y = np.concatenate((y0, y1), axis=0)
    return X, y


def generate_three_class_data(n_points):
    """
    Gera dados sintéticos para um exemplo com três classes.
    """
    # Classe 0: Pontos centrados em (0, 0)
    X0 = np.random.randn(n_points, 2) + np.array([0, 0])
    y0 = np.zeros(n_points)
    # Classe 1: Pontos centrados em (5, 5)
    X1 = np.random.randn(n_points, 2) + np.array([5, 5])
    y1 = np.ones(n_points)
    # Classe 2: Pontos centrados em (0, 5)
    X2 = np.random.randn(n_points, 2) + np.array([0, 5])
    y2 = np.full(n_points, 2)

    X = np.concatenate((X0, X1, X2), axis=0)
    y = np.concatenate((y0, y1, y2), axis=0)
    return X, y


# --------------------- Funções de Plot ---------------------
def plot_data(X, y, title):
    """
    Plota os dados num gráfico de dispersão e retorna a figura.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor="k", cmap=plt.cm.RdYlBu)
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    return fig


def plot_decision_boundary(knn, X, y, title):
    """
    Plota a fronteira de decisão do modelo KNN juntamente com os dados e retorna a figura.
    """
    h = 0.1  # tamanho do passo para a grade
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = knn.predict(grid_points)
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor="k", cmap=plt.cm.RdYlBu)
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    return fig


# --------------------- Aplicativo Streamlit ---------------------
def main():
    st.title("Classificador KNN com Streamlit")

    st.sidebar.header("Configurações")
    n_points = st.sidebar.number_input(
        "Número de pontos para cada classe:", min_value=10, value=50, step=1
    )
    k = st.sidebar.number_input("Número de vizinhos (k):", min_value=1, value=3, step=1)
    normalize_option = st.sidebar.selectbox("Normalizar dados?", options=["Sim", "Não"])
    normalize = True if normalize_option == "Sim" else False
    example_choice = st.sidebar.radio(
        "Escolha o exemplo", options=["Duas classes", "Três classes"]
    )

    # Gera os dados de acordo com a escolha do exemplo
    if example_choice == "Três classes":
        X, y = generate_three_class_data(n_points)
        example_title = "Exemplo com Três Classes"
    else:
        X, y = generate_two_class_data(n_points)
        example_title = "Exemplo com Duas Classes"

    # --------------------- Plot dos Dados Gerados ---------------------
    st.subheader(f"{example_title} - Dados Gerados (antes do modelo)")
    fig_data = plot_data(X, y, f"{example_title} - Dados Gerados (antes do modelo)")
    st.pyplot(fig_data)

    # --------------------- Treinamento do Modelo e Fronteira de Decisão ---------------------
    knn = KNNClassifier(k=k, normalize=normalize)
    knn.fit(X, y)
    fig_boundary = plot_decision_boundary(
        knn,
        X,
        y,
        f"{example_title} - Fronteira de Decisão (k = {k}, Normalização = {normalize})",
    )
    st.subheader("Fronteira de Decisão")
    st.pyplot(fig_boundary)

    # --------------------- Exemplo Adicional de Previsão ---------------------
    new_points = np.array([[1, 1], [3, 3], [0, 4], [6, 4]])
    predictions = knn.predict(new_points)
    st.subheader("Previsões para Novos Pontos")
    st.write("Novos pontos:", new_points)
    st.write("Previsões:", predictions)

    # Plot dos novos pontos sobre a fronteira de decisão
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = knn.predict(grid_points)
    Z = Z.reshape(xx.shape)

    fig_new = plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor="k", cmap=plt.cm.RdYlBu)
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
    st.subheader("Fronteira de Decisão com Novos Pontos")
    st.pyplot(fig_new)


if __name__ == "__main__":
    main()
