import gym

import algs.sarsa
import test_envs


if __name__ == "__main__":
    env = gym.make(
        id="Gridworld-v0",
        height=20,
        width=20,
        num_traps=32,
        num_goals=3,
        display_delay=0.01,
        reward_per_action=-0.001,
        path_noise_prob=0.15,
    )
    env.seed(16)
    model = algs.sarsa.SARSA(env, epsilon_decay_steps=1500, random_state=32)
    model.connect_values_to_env()
    model.optimize(
        num_episodes=3000,
        episodes_to_print=150,
        episodes_to_render=150,
    )
