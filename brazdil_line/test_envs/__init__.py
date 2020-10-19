import gym

gym.envs.registration.register(
    id="Gridworld-v0",
    entry_point="test_envs.gridworld:Gridworld",
)
