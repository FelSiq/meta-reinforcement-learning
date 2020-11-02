"""Plot the environment varying hyper-parameters."""
import matplotlib.pyplot as plt
import gym
import numpy as np

import test_envs.gridworld


def plot():
    height = 10
    width = 10

    probs_a = np.linspace(0, 0.50, 1)
    probs_b = np.linspace(0, 0.25, 1)

    if probs_a.size * probs_b.size == 1:
        env = get_env(
            path_noise_prob=probs_a[0],
            blind_switch_prob=probs_b[0],
            height=height,
            width=width,
        )
        render = env.render("rgb_array")
        plt.imshow(render, aspect="auto")
        plt.scatter(env.start[1], env.start[0], color="red")

        traps_y, traps_x = zip(*list(env.traps))
        goals_y, goals_x = zip(*list(env.goals))

        plt.scatter(traps_x, traps_y, c=[(1.0, 0.4, 0.0)], marker="X", s=128)
        plt.scatter(goals_x, goals_y, c=[(0.0, 0.4, 1.0)], marker="*", s=128)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        return

    fig, axes = plt.subplots(probs_a.size, probs_b.size, sharex=True, sharey=True)

    fontsize = 24
    fig.text(
        0.5,
        0.01,
        "Probabilidade de movimento ruídoso [0.0, 0.5]",
        ha="center",
        fontsize=fontsize,
    )
    fig.text(
        0.01,
        0.5,
        "Probabilidade de desativar busca guiada [0.0, 0.25]",
        va="center",
        rotation="vertical",
        fontsize=fontsize,
    )

    for i, pa in enumerate(probs_a):
        for j, pb in enumerate(probs_b):
            subplot = 1 + i * probs_b.size + j
            ax = axes[probs_a.size - i - 1][j]
            print(
                f"Rendering subplot {subplot} / {probs_a.size * probs_b.size} "
                f"(Pa = {pa:.2f}, Pb = {pb:.2f})...",
                end=" ",
            )
            env = get_env(
                path_noise_prob=pa, blind_switch_prob=pb, height=height, width=width
            )
            render = env.render("rgb_array")
            ax.imshow(render, aspect="auto")
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)
            plt.plot()
            print("Done")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.show()


def get_env(
    path_noise_prob: float,
    blind_switch_prob: float,
    width: int = 32,
    height: int = 32,
    random_state: int = 16,
):
    assert 0 <= path_noise_prob <= 1
    assert 0 <= blind_switch_prob <= 1
    assert width > 0
    assert height > 0

    env = gym.make(
        "Gridworld-v0",
        height=height,
        width=width,
        path_noise_prob=path_noise_prob,
        blind_switch_prob=blind_switch_prob,
    )

    env.seed(random_state)

    env.reset()

    return env


if __name__ == "__main__":
    plot()
