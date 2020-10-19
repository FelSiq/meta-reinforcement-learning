import typing as t
import pickle

import numpy as np

from . import base


class QLearning(base.BaseModelDiscrete):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_values = dict()

    def take_greedy_action(self, state: t.Any) -> int:
        if state not in self.q_values:
            return np.random.randint(self.num_actions)

        return self.q_values[state].argmax()

    def step(self, pack: t.Dict[str, t.Any]) -> float:
        state = pack["state"]
        action = pack["action"]
        reward = pack["reward"]
        next_state = pack.get("next_state")

        td_error = reward

        self.q_values.setdefault(state, np.zeros(self.num_actions))

        if next_state is not None:
            greedy_a = self.take_greedy_action(next_state)

            self.q_values.setdefault(next_state, np.zeros(self.num_actions))

            td_error += (
                self.discount_factor * self.q_values[next_state][greedy_a]
                - self.q_values[state][action]
            )

        self.q_values[state][action] += self.learning_rate * td_error

        return td_error

    def optimize(
        self,
        num_episodes: int,
        episodes_to_print: int = -1,
        episodes_to_render: int = -1,
    ) -> float:
        num_episodes = int(num_episodes)

        assert num_episodes > 0

        if self.random_state is not None:
            np.random.seed(self.random_state)

        error = 0.0

        for epi_ind in np.arange(num_episodes):
            state = self.env.reset()

            done = False

            while not done:
                if episodes_to_render > 0 and epi_ind % episodes_to_render == 0:
                    self.env.render(mode="human")

                action = self.pick_action(state)
                next_state, reward, done, _ = self.env.step(action)

                error += self.step(
                    dict(
                        state=state,
                        action=action,
                        reward=reward,
                        next_state=next_state,
                    )
                )

                state = next_state

                if done:
                    self.episode_end()
                    self.env.close()

            if episodes_to_print > 0 and epi_ind % episodes_to_print == 0:
                error /= episodes_to_print
                print(
                    f"Episode: {epi_ind:<{6}} - avg. temporal-difference (TD) error: {error:.4f} - epsilon: {self.epsilon:.4f}"
                )
                error = 0.0

        return self

    def connect_values_to_env(self, *args, **kwargs) -> None:
        super().connect_values_to_env(*args, **kwargs)
        self.env.state_values_material = {
            "state_values": self.q_values,
            "epsilon": self.epsilon,
        }

    @classmethod
    def load(cls, filepath: str) -> "QLearning":
        filepath = cls.check_pickle_filename(filepath)

        model = None

        with open(filepath, "rb") as f:
            model = pickle.load(f)

        return model

    def save(self, filepath: str) -> None:
        filepath = self.check_pickle_filename(filepath)

        with open(filepath, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
