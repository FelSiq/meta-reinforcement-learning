import typing as t
import argparse
import os

import pandas as pd
import numpy as np
import gym
import test_envs

import algs.sarsa
import algs.mc_control
import algs.q_learning
import algs.double_q_learning


def save_data(data: pd.DataFrame, full_path: str) -> None:
    data.to_csv(full_path)


def save_seeds(full_path: str, *args):
    with open(full_path, "w") as f_aux:
        f_aux.write(",".join(map(str, args)))


def load_seeds(full_path: str):
    print(full_path)
    with open(full_path, "r") as f_aux:
        return map(int, f_aux.read().strip().split(","))


def extract_metafeatures(env, artificial: bool) -> t.List[t.Union[int, float]]:
    """Extract meta-features."""
    if artificial:
        metafeatures = [
            env.height,
            env.width,
            env.num_traps,
            env.num_goals,
            env.path_noise_prob,
            env.blind_switch_prob,
        ]

    else:
        raise NotImplementedError

    return metafeatures


def run_base_models(
    env,
    base_models: t.List[algs.base.BaseModelDiscrete],
    num_episodes: int,
) -> t.List[float]:
    res = len(base_models) * [0.0]

    for i, model in enumerate(base_models):
        print(f"  Running {i + 1} / {len(base_models)} base algorithm...", end=" ")
        model.optimize(num_episodes=num_episodes)
        res[i] = model.run()
        print("Done")

    return res


def init_env(
    env_seed: int, hyperparam_seed: int, env_max_steps: int, reward_per_action: float
):
    if env_seed is None or hyperparam_seed is None:
        raise ValueError("Both random seeds must be given!")

    np.random.seed(hyperparam_seed)

    env_args = dict(
        height=np.random.randint(12, 48),
        width=np.random.randint(12, 48),
        num_traps=np.random.randint(8, 32),
        num_goals=np.random.randint(1, 8),
        path_noise_prob=0.5 * np.random.random(),
        blind_switch_prob=0.25 * np.random.random(),
    )

    env = gym.make(
        id="Gridworld-v0",
        max_timesteps=env_max_steps,
        reward_per_action=reward_per_action,
        **env_args,
    )
    env.seed(env_seed)

    return env


def binsearch(y):
    start = 0
    ind = y.shape[0]
    end = y.shape[0] - 1
    while start <= end:
        middle = start + (end - start) // 2
        if pd.isna(y.iloc[middle, :]).any():
            ind = middle
            end = middle - 1
        else:
            start = middle + 1
    return ind


def build_metadataset(
    size: int,
    env_seed: int,
    hyperparam_seed: int,
    alg_seed: int,
    artificial: bool,
    checkpoint_path: str,
    checkpoint_iter: int,
    num_episodes: int,
    env_max_steps: int,
    reward_per_action: float,
):
    assert num_episodes > 0

    base_alg_names = ["SARSA", "MCCONTROL", "QLEARNING", "DQLEARNING"]

    if artificial:
        metafeat_names = [
            "HEIGHT",
            "WIDTH",
            "NUM_TRAPS",
            "NUM_GOALS",
            "PATH_NOISE_PROB",
            "BLIND_SWITCH_PROB",
        ]

    else:
        raise NotImplementedError

    path_suffix = f"_{num_episodes}{'_artificial' if artificial else ''}.csv"

    X_checkpoint_path = os.path.join(checkpoint_path, "X_checkpoint" + path_suffix)
    y_checkpoint_path = os.path.join(checkpoint_path, "y_checkpoint" + path_suffix)
    seeds_checkpoint_path = os.path.join(
        checkpoint_path, "seeds_checkpoint" + path_suffix
    )

    try:
        X = pd.read_csv(X_checkpoint_path, index_col=0)
        y = pd.read_csv(y_checkpoint_path, index_col=0)
        start_ind = binsearch(y)
        env_seed, hyperparam_seed, alg_seed = load_seeds(
            full_path=seeds_checkpoint_path
        )
        print(
            f"Loaded checkpoint files, restoring from iteration {start_ind + 1} / {max(size, y.shape[0])}."
        )

    except FileNotFoundError:
        print("Checkpoint files not found.")
        X = pd.DataFrame(index=np.arange(size), columns=metafeat_names)
        y = pd.DataFrame(index=np.arange(size), columns=base_alg_names)
        start_ind = 0

    if X.shape[0] < size:
        X = X.append(pd.DataFrame(index=np.arange(X.shape[0], size), columns=X.columns))
        y = y.append(pd.DataFrame(index=np.arange(y.shape[0], size), columns=X.columns))

    for i in np.arange(start_ind, X.shape[0]):
        print(f"Began iteration: {i + 1} / {size}...")
        env = init_env(
            env_seed=env_seed,
            hyperparam_seed=hyperparam_seed,
            env_max_steps=env_max_steps,
            reward_per_action=reward_per_action,
        )

        base_models = [
            algs.sarsa.SARSA(env=env, random_state=alg_seed),
            algs.mc_control.MCControl(env=env, random_state=alg_seed),
            algs.q_learning.QLearning(env=env, random_state=alg_seed),
            algs.double_q_learning.DQLearning(env=env, random_state=alg_seed),
        ]

        cur_X = extract_metafeatures(env=env, artificial=artificial)
        cur_y = run_base_models(
            env=env,
            base_models=base_models,
            num_episodes=num_episodes,
        )

        assert len(cur_X) == X.shape[1]
        assert len(cur_y) == y.shape[1]

        X.iloc[i, :] = cur_X
        y.iloc[i, :] = cur_y

        env_seed += 1
        hyperparam_seed += 2
        alg_seed += 3
        print(f"Done iteration {i}.")

        if i % checkpoint_iter == 0:
            save_data(data=X, full_path=X_checkpoint_path)
            save_data(data=y, full_path=y_checkpoint_path)
            save_seeds(seeds_checkpoint_path, env_seed, hyperparam_seed, alg_seed)
            print("Created metadata and random seeds checkpoints.")

    save_data(data=X, full_path=X_checkpoint_path)
    save_data(data=y, full_path=y_checkpoint_path)
    save_seeds(seeds_checkpoint_path, env_seed, hyperparam_seed, alg_seed)
    print("Finished and created metadata checkpoint.")

    return pd.concat((X, y), axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect meta-data (metafeatures and algorithms estimated total reward)"
    )

    parser.add_argument(
        "size",
        metavar="N",
        type=int,
        help="size (number of samples) of metadata",
    )
    parser.add_argument(
        "--num-episodes",
        default=3000,
        metavar="M",
        type=int,
        help="number of episodes to optimize each base model",
    )
    parser.add_argument(
        "--reward-per-action",
        default=-0.001,
        metavar="R",
        type=float,
        help="reward for each agent action",
    )
    parser.add_argument(
        "--env-max-steps",
        default=1000,
        metavar="K",
        type=int,
        help="maximum number of timesteps allowed in the environment (a.k.a. budget)",
    )
    parser.add_argument(
        "--artificial",
        action="store_true",
        help="if given, extract artificial metafeatures",
    )
    parser.add_argument(
        "--env-seed",
        default=16,
        metavar="R",
        type=int,
        help="initial random seed for the environment",
    )
    parser.add_argument(
        "--hyperparam-seed",
        default=48,
        metavar="S",
        type=int,
        help="initial random seed for the environment",
    )
    parser.add_argument(
        "--alg-seed",
        default=32,
        metavar="T",
        type=int,
        help="initial random seed for each base algorithm",
    )
    parser.add_argument(
        "--checkpoint-iter",
        default=10,
        metavar="I",
        type=int,
        help="number of iterations while extracting meta-features to save the current results to file",
    )

    args = parser.parse_args()
    out_dir_path = "metadata"

    try:
        os.mkdir(out_dir_path)

    except OSError:
        pass

    filename = f"metafeatures{'_artificial' if args.artificial else ''}.csv"
    full_path = os.path.join(out_dir_path, filename)

    res = build_metadataset(checkpoint_path=out_dir_path, **vars(args))

    print("Finished")

    res.to_csv(full_path)
    print(f"Saved dataframe to {full_path}.")
