import typing as t

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sklearn.manifold
import sklearn.decomposition
import sklearn.preprocessing

import utils


def plot_y():
    fig, axes = plt.subplots(3, 4, sharex=True)
    fig_2, axes_2 = plt.subplots(3, figsize=(9, 10))

    for i, num_train_episodes in enumerate([500, 1000, 3000]):
        _, y = utils.get_metadata(
            num_train_episodes=num_train_episodes, artificial=False
        )

        for j in np.arange(y.shape[1]):
            ax = axes[i][j]
            ax.set_title(num_train_episodes)
            y_cur = y.iloc[:, j]
            sns.histplot(y_cur, ax=ax, stat="density", bins=8, palette="deep")
            ax.set_ylabel("Densidade")

        aux = y.values.T.flatten()
        aux = pd.DataFrame.from_dict(
            {"Algoritmo": np.repeat(y.columns, y.shape[0]), "Converged": aux > 0.0}
        )

        sns.countplot(
            x="Algoritmo", hue="Converged", data=aux, ax=axes_2[i], palette="deep"
        )

        if i != 2:
            axes_2[i].set_xlabel(None)

        ax = axes_2[i]
        ax.set_title(f"Episodios = {num_train_episodes}")
        ax.set_ylabel("Frequencia")
        legend_labels, _ = ax.get_legend_handles_labels()
        ax.legend(legend_labels, ["Não", "Sim"], title="Convergiu?")

    plt.tight_layout()
    plt.show()


def plot_tsne(n_tries: int = 10, artificial: bool = False):
    assert n_tries > 0

    fig, axes = plt.subplots(3, 4)
    fig.suptitle(f"{'Artificial' if artificial else 'Elaboradas'}")

    for j, num_train_episodes in enumerate([500, 1000, 3000]):
        X, y = utils.get_metadata(
            num_train_episodes=num_train_episodes, artificial=artificial
        )

        pca = sklearn.decomposition.PCA(n_components=0.90)

        X = X.iloc[:, np.isnan(X.values).mean(axis=0) <= 0.7]
        X.interpolate(axis=0, inplace=True)
        X = X.iloc[:, X.values.std(axis=0, ddof=0) > 0.0]
        X = sklearn.preprocessing.StandardScaler().fit_transform(X)

        m = X.shape[1]
        X = pca.fit_transform(X)

        print(f"PCA dimension reduction: {X.shape[1] / m:.3f}")

        kl_divs = np.full(n_tries, fill_value=np.inf)

        for i in np.arange(n_tries):
            print(f"{j + 1}th dataset, {i + 1}th try ...", end=" ")
            tsne = sklearn.manifold.TSNE(method="barnes_hut", random_state=i)
            tsne.fit(X)
            kl_divs[i] = tsne.kl_divergence_
            print("Done.")

        argmin_ind = kl_divs.argmin()
        X_embedded = sklearn.manifold.TSNE(
            method="barnes_hut", random_state=argmin_ind
        ).fit_transform(X)

        colors = ["red", "blue"]

        for i in np.arange(y.shape[1]):
            ax = axes[j][i]
            ax.scatter(
                *X_embedded.T,
                c=y.iloc[:, i].values > 0.0,
                cmap=matplotlib.colors.ListedColormap(colors),
            )
            ax.set_title(num_train_episodes)

    plt.show()


if __name__ == "__main__":
    plot_y()
    # plot_tsne(artificial=True)
    # plot_tsne(artificial=False)
