import argparse

import gym

import algs.sarsa
import test_envs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run SARSA algorithm in Gridworld environment."
    )
    parser.add_argument("--height", type=int, help="height of the grid", default=10)
    parser.add_argument("--width", type=int, help="width of the grid", default=10)
    parser.add_argument(
        "--num-goals",
        type=int,
        help="number of goals/good states which ends the episode with +1 reward",
        default=1,
    )
    parser.add_argument(
        "--num-traps",
        type=int,
        help="number of traps/bad states which ends the episode with -1 reward",
        default=8,
    )
    parser.add_argument(
        "--display-delay",
        type=float,
        help="delay in seconds between each render frame",
        default=0.01,
    )
    parser.add_argument(
        "--path-noise-prob",
        type=float,
        help="probability of noise movements while generating gridworld",
        default=0.25,
    )
    parser.add_argument(
        "--reward-per-action",
        type=float,
        help="reward given to the agent per action/timestep",
        default=-0.001,
    )
    parser.add_argument(
        "--env-seed",
        type=int,
        help="environment seed for pseudo-random numbers",
        default=16,
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        help="model seed for pseudo-random numbers",
        default=None,
    )
    parser.add_argument(
        "--num-episodes", type=int, help="number of episodes to optimize", default=1024
    )
    parser.add_argument(
        "--epsilon",
        metavar=("START", "END"),
        type=float,
        nargs=2,
        help="epsilon (e-greedy argument) in the form [start, end]",
        default=(0.90, 0.05),
    )
    parser.add_argument(
        "--epsilon-decay-steps",
        type=int,
        help="Number of steps (episodes) to decay epsilon (e-greedy argument)",
        default=512,
    )
    parser.add_argument(
        "--episodes-to-print",
        type=int,
        help="optimization delay (in episodes) between each log",
        default=32,
    )
    parser.add_argument(
        "--episodes-to-render",
        type=int,
        help="optimization delay (in episodes) between each render",
        default=128,
    )
    parser.add_argument(
        "--no-load",
        help="if given, do not load any saved modes (WARNING: this operation will overwrite any previously saved models if --no-save is not given!)",
        action="store_true",
    )
    parser.add_argument(
        "--no-save",
        help="do not save the optimized model",
        action="store_true",
    )
    parser.add_argument(
        "--reset-epsilon",
        help="if given, reset the model epsilon (e-greedy) to the new given schedule ('start', 'end' and 'epsilon-decay-steps')",
        action="store_true",
    )

    args = parser.parse_args()

    try:
        if args.no_load:
            raise FileNotFoundError

        model = algs.sarsa.SARSA.load("models/sarsa")

    except FileNotFoundError:
        env = gym.make(
            id="Gridworld-v0",
            height=args.height,
            width=args.width,
            num_traps=args.num_traps,
            num_goals=args.num_goals,
            display_delay=args.display_delay,
            reward_per_action=args.reward_per_action,
            path_noise_prob=args.path_noise_prob,
        )
        env.seed(args.env_seed)

        model = algs.sarsa.SARSA(
            env,
            epsilon=args.epsilon,
            epsilon_decay_steps=args.epsilon_decay_steps,
            random_state=args.model_seed,
        )

    model.connect_values_to_env()

    if args.reset_epsilon:
        model.reset_epsilon(
            new_epsilon=args.epsilon, epsilon_decay_steps=args.epsilon_decay_steps
        )

    try:
        model.optimize(
            num_episodes=args.num_episodes,
            episodes_to_print=args.episodes_to_print,
            episodes_to_render=args.episodes_to_render,
        )

    except KeyboardInterrupt:
        pass

    if not args.no_save:
        model.save("models/sarsa")
