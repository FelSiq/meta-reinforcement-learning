import gym

import algs.sarsa
import test_envs.gridworld


if __name__ == "__main__":
    env = test_envs.gridworld.Gridworld(
        height=128,
        width=128,
        num_traps=32,
        num_goals=4,
        display_delay=0.01,
        reward_per_action=-0.001,
        path_noise_prob=0.25,
    )
    env.seed(11)
    model = algs.sarsa.SARSA(env, epsilon_decay_steps=1500, random_state=32)
    model.optimize(
        num_episodes=3000,
        episodes_to_print=50,
        episodes_to_render=50,
    )
