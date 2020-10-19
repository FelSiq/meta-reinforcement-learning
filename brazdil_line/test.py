import gym

import algs.sarsa
import test_envs


if __name__ == "__main__":
    try:
        model = algs.sarsa.SARSA.load("models/sarsa")

    except FileNotFoundError:
        env = gym.make(
            id="Gridworld-v0",
            height=32,
            width=32,
            num_traps=32,
            num_goals=2,
            display_delay=0.01,
            reward_per_action=-0.001,
            path_noise_prob=0.10,
        )
        env.seed(16)

        model = algs.sarsa.SARSA(env, epsilon_decay_steps=1500, random_state=32)

        model.connect_values_to_env()

    model.optimize(
        num_episodes=3500,
        episodes_to_print=100,
        episodes_to_render=500,
    )

    model.save("models/sarsa")
