import typing as t
import pickle

import numpy as np

from . import base


class MCControl(base.BaseModelDiscrete):
    def __init__(self, env, every_visit: bool = True, *args, **kwargs):
        super().__init__(env=env, *args, **kwargs)
        self.q_values = dict()
        self.every_visit = every_visit

    @staticmethod
    def _preprocess_value_val_func(item):
        return item[1, :]

    def take_greedy_action(self, state: t.Any, train: bool = True) -> int:
        if state not in self.q_values:
            return np.random.randint(self.num_actions)

        return self.q_values[state][1, :].argmax()

    def step(self, pack: t.Dict[str, t.Any]) -> float:
        states = pack["states"]
        actions = pack["actions"]
        rewards = pack["rewards"]

        avg_mc_error = 0.0

        n = len(rewards)

        returns = (1 + n) * [0.0]

        for i in np.arange(n - 1, -1, -1):
            returns[i] = rewards[i] + self.discount_factor * returns[i + 1]

        visited = set()

        for i in np.arange(n):
            state = states[i]
            action = actions[i]

            pair = (state, action)

            if self.every_visit or pair not in visited:
                if not self.every_visit:
                    visited.add(pair)

                self.q_values.setdefault(
                    state, np.zeros((2, self.num_actions), dtype=float)
                )
                self.q_values[state][0, action] += 1

                freq, cur_q_val = self.q_values[state][:, action]

                mc_error = returns[i] - cur_q_val
                avg_mc_error += mc_error
                self.q_values[state][1, action] += mc_error / freq

        return avg_mc_error / n

    def optimize(
        self,
        num_episodes: int,
        episodes_to_print: int = -1,
        episodes_to_render: int = -1,
    ) -> float:
        num_episodes = int(num_episodes)

        assert num_episodes > 0

        cur_random_seed = self.random_state

        error = 0.0

        for epi_ind in np.arange(num_episodes):
            state = self.env.reset()

            if cur_random_seed is not None:
                np.random.seed(cur_random_seed)
                cur_random_seed += 1

            done = False

            rewards = []
            actions = []
            states = [state]

            while not done:
                if episodes_to_render > 0 and epi_ind % episodes_to_render == 0:
                    self.env.render(mode="human")

                action = self.pick_action(state)
                next_state, reward, done, _ = self.env.step(action)

                rewards.append(reward)
                actions.append(action)
                states.append(next_state)

                state = next_state

                if done:
                    self.episode_end()
                    self.env.close()

            error += self.step(dict(rewards=rewards, actions=actions, states=states))

            if episodes_to_print > 0 and epi_ind % episodes_to_print == 0:
                error /= episodes_to_print
                print(
                    f"Episode: {epi_ind:<{6}} - avg. monte carlo (MC) {'every visit' if self.every_visit else 'first visit'} estimation error: {error:.4f} - epsilon: {self.epsilon:.4f}"
                )
                error = 0.0

        return self

    def connect_values_to_env(self, *args, **kwargs) -> None:
        super().connect_values_to_env(*args, **kwargs)
        self.env.state_values_material = {
            "state_values": self.q_values,
            "preprocess_value_val_func": self._preprocess_value_val_func,
            "epsilon": self.epsilon,
        }

    @classmethod
    def load(cls, filepath: str) -> "MCControl":
        filepath = cls.check_pickle_filename(filepath)

        model = None

        with open(filepath, "rb") as f:
            model = pickle.load(f)

        return model

    def save(self, filepath: str) -> None:
        filepath = self.check_pickle_filename(filepath)

        with open(filepath, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
