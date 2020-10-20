import argparse

import gym

import algs.mc_control
import test_envs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run Monte Carlo Policy Control algorithm in Gridworld environment."
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
        "--discount-factor",
        type=float,
        help="Discount factor of the return",
        default=0.99,
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
        "--blind-switch-prob",
        type=float,
        help="probability disabling/enabling informed search while generating the environment",
        default=0.10,
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
    parser.add_argument(
        "--first-visit-only",
        help="if true, update Q-value estimates only with the first appearance of each (s, a) pair in each episode.",
        action="store_true",
    )

    args = parser.parse_args()

    model_filepath = (
        f"models/mc_control_{'first_visit' if args.first_visit_only else 'every_visit'}"
    )

    try:
        if args.no_load:
            raise FileNotFoundError

        model = algs.mc_control.MCControl.load(model_filepath)

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
            blind_switch_prob=args.blind_switch_prob,
        )
        env.seed(args.env_seed)

        model = algs.mc_control.MCControl(
            env,
            every_visit=not args.first_visit_only,
            epsilon=args.epsilon,
            epsilon_decay_steps=args.epsilon_decay_steps,
            random_state=args.model_seed,
            discount_factor=args.discount_factor,
        )

    model.connect_values_to_env()

    if args.reset_epsilon:
        model.reset_epsilon(
            new_epsilon=args.epsilon, new_epsilon_decay_steps=args.epsilon_decay_steps
        )

    try:
        model.optimize(
            num_episodes=args.num_episodes,
            episodes_to_print=args.episodes_to_print,
            episodes_to_render=args.episodes_to_render,
        )

    except KeyboardInterrupt:
        pass

    model.run(render=True)

    if not args.no_save:
        model.save(model_filepath)
