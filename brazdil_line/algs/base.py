import typing as t

import numpy as np


class BaseModelDiscrete:
    def __init__(
        self,
        env: t.Any,
        learning_rate: float = 0.01,
        discount_factor: float = 1.0,
        epsilon: t.Union[float, t.Tuple[float, float]] = (0.90, 0.05),
        epsilon_decay_steps: int = 1000,
        update_epsilon_schedule: str = "episode",
        random_state: t.Optional[int] = None,
    ):
        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0]

        assert update_epsilon_schedule in {"episode", "action"}
        assert num_actions > 0
        assert num_states > 0 or num_states is None
        assert epsilon_decay_steps > 0
        assert learning_rate > 0.0
        assert 0.0 <= discount_factor <= 1.0

        self.env = env

        self.epsilon_start, self.epsilon_end = self._unpack_epsilon(epsilon)
        self.epsilon = self.epsilon_start

        self.num_actions = num_actions
        self.num_states = num_states

        self.num_episodes_done = 0
        self.num_actions_done = 0
        self.num_epsilon_steps_done = 0

        self.update_epsilon_in_action = update_epsilon_schedule == "action"
        self.update_epsilon_in_episode = update_epsilon_schedule == "episode"
        self.epsilon_decay_steps = epsilon_decay_steps

        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        self.random_state = random_state

    @staticmethod
    def _unpack_epsilon(
        epsilon: t.Union[float, t.Tuple[float, float]]
    ) -> t.Tuple[float, float]:
        if hasattr(epsilon, "__len__"):
            assert len(epsilon) == 2

        else:
            epsilon = (epsilon, epsilon)

        epsilon = tuple(map(float, epsilon))

        _start, _end = epsilon

        assert 0 <= _start <= 1
        assert 0 <= _end <= 1
        assert _start >= _end

        return tuple(epsilon)

    def reset_epsilon(
        self,
        new_epsilon: t.Optional[t.Union[float, t.Tuple[float, float]]] = None,
        new_epsilon_decay_steps: t.Optional[int] = None,
    ) -> None:
        if new_epsilon is not None:
            self.epsilon_start, self.epsilon_end = self._unpack_epsilon(new_epsilon)

        if new_epsilon_decay_steps is not None:
            self.epsilon_decay_steps = new_epsilon_decay_steps

        self.epsilon = self.epsilon_start
        self.num_epsilon_steps_done = 0

    def _update_epsilon(self, timestep: int) -> float:
        if timestep >= self.epsilon_decay_steps:
            self.epsilon = self.epsilon_end
            return self.epsilon

        self.epsilon = self.epsilon_end + (
            1.0 - timestep / self.epsilon_decay_steps
        ) * (self.epsilon_start - self.epsilon_end)

        return self.epsilon

    def pick_action(self, state: t.Any) -> int:
        if np.random.random() > self.epsilon:
            action = self.take_greedy_action(state)

        else:
            action = np.random.randint(self.num_actions)

        self.num_actions_done += 1

        if self.update_epsilon_in_action:
            self.num_epsilon_steps_done += 1
            self._update_epsilon(timestep=self.num_epsilon_steps_done)

        return int(action)

    def episode_end(self) -> None:
        self.num_episodes_done += 1

        if self.update_epsilon_in_episode:
            self.num_epsilon_steps_done += 1
            self._update_epsilon(timestep=self.num_epsilon_steps_done)

    def take_greedy_action(self, state: t.Any) -> int:
        raise NotImplementedError

    def step(self, pack: t.Dict[str, t.Any]) -> float:
        raise NotImplementedError

    def optimize(
        self, num_episodes: int, episodes_to_print: int = -1, render: bool = False
    ) -> float:
        raise NotImplementedError

    def connect_values_to_env(self, *args, **kwargs) -> None:
        if not hasattr(self.env, "state_values_material"):
            raise RuntimeError("Environment does not support this operation.")

    def disconnect_values_to_env(self) -> None:
        if not hasattr(self.env, "state_values_material"):
            raise RuntimeError("Environment does not support this operation.")

        self.env.state_values_material = None

    @staticmethod
    def check_pickle_filename(filepath: str) -> str:
        if not filepath.endswith(".pickle"):
            filepath += ".pickle"

        return filepath

    def load(self, filepath: str) -> "BaseModelDiscrete":
        raise NotImplementedError

    def save(self, filepath: str) -> None:
        raise NotImplementedError
