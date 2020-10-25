import typing as t

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

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
        ax.legend(legend_labels, ["Sim", "Não"], title="Convergiu?")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_y()
