import typing as t
import collections
import enum

import gym
import gym.spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("GTK3Agg")


Coord = collections.namedtuple("Coord", ("y", "x"))


class CellColorRGB:
    TRAP = (127, 0, 0)
    CLEAN = (255, 255, 255)
    WALL = (0, 0, 0)
    GOAL = (0, 255, 64)


class CellCode(enum.IntEnum):
    TRAP = -1
    CLEAN = 0
    WALL = 1
    GOAL = 3


class Gridworld(gym.Env):
    metadata = {"render.modes": ["human"]}
    action_space = gym.spaces.Discrete(4)
    observation_space = gym.spaces.Box(
        low=np.array((0, 0)), high=np.array((np.inf, np.inf)), dtype=float
    )
    reward_range = (-1.0, 1.0)

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        num_goals: int = 2,
        num_traps: int = 8,
        path_noise_prob: float = 0.25,
        traps_ends_episode: bool = True,
        reward_per_action: float = 0.0,
        display_delay: float = 0.05,
    ):
        super().__init__()

        assert 0.0 <= path_noise_prob <= 1.0
        assert width > 0
        assert height > 0
        assert num_traps >= 0
        assert num_goals > 0
        assert height * width >= 1 + num_goals + num_traps
        assert display_delay > 0.0

        self.path_noise_prob = path_noise_prob
        self.traps_ends_episode = traps_ends_episode
        self.width = width
        self.height = height

        self.reward_per_action = reward_per_action

        self.num_traps = num_traps
        self.num_goals = num_goals

        self.start = Coord(-1, -1)
        self.current_pos = self.start
        self.goals = set()  # type: t.Set[Coord]
        self.traps = set()  # type: t.Set[Coord]
        self.done = False

        self.random_state = None

        self.map = np.empty((self.width, self.height))

        self.plot_ax = None
        self.plot_fig = None
        self.plot_draw_bg = False
        self.plot_display_delay = display_delay

    def _gen_map(self):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        def move(ty, tx):
            if not 0 <= ty < self.height or not 0 <= tx < self.width:
                queue.insert(0, visited.pop())
                return

            queue.insert(0, (ty, tx))

        self.map = np.full(
            (self.height, self.width), fill_value=CellCode.WALL, dtype=np.int8
        )
        selected_inds = np.random.choice(
            self.height * self.width,
            replace=False,
            size=1 + self.num_goals + self.num_traps,
        )

        self.start = Coord(*divmod(selected_inds[0], self.width))
        self.goals = {
            Coord(*divmod(ind, self.width))
            for ind in selected_inds[1 : 1 + self.num_goals]
        }
        self.traps = {
            Coord(*divmod(ind, self.width))
            for ind in selected_inds[1 + self.num_goals :]
        }

        for y, x in self.goals:
            self.map[y, x] = CellCode.GOAL

        for y, x in self.traps:
            self.map[y, x] = CellCode.TRAP

        # Note: for each goal and trap cell, drill a path through start cell

        for start in self.goals.union(self.traps):
            visited = set()
            queue = [start]

            while queue:
                y, x = queue.pop()

                if self.map[y, x] == CellCode.WALL:
                    self.map[y, x] = CellCode.CLEAN

                if (y, x) == self.start:
                    break

                visited.add((y, x))
                dx = dy = 0

                if np.random.random() > 0.5:
                    dy = +1 if y < self.start.y else -1

                else:
                    dx = +1 if x < self.start.x else -1

                # Note: insert noise
                if np.random.random() <= self.path_noise_prob:
                    dx *= -1
                    dy *= -1

                move(y + dy, x + dx)

        return self.map

    def step(self, action):
        new_y, new_x = self.current_pos

        if action == 0:
            new_x = min(self.current_pos.x + 1, self.width - 1)

        elif action == 1:
            new_x = max(self.current_pos.x - 1, 0)

        elif action == 2:
            new_y = min(self.current_pos.y + 1, self.height - 1)

        elif action == 3:
            new_y = max(self.current_pos.y - 1, 0)

        new_pos = Coord(new_y, new_x)

        cell_val = self.map[self.current_pos]
        new_cell_val = self.map[new_pos]

        if new_cell_val != CellCode.WALL:
            self.current_pos = new_pos
            cell_val = new_cell_val

        observation = self.current_pos

        reward = self.reward_per_action

        if cell_val == CellCode.GOAL:
            reward = 1

        if cell_val == CellCode.TRAP:
            reward = -1

        if cell_val == CellCode.GOAL or (
            self.traps_ends_episode and cell_val == CellCode.TRAP
        ):
            self.done = True

        info = dict()

        return observation, reward, self.done, info

    def reset(self):
        self.done = False
        self._gen_map()
        self.current_pos = self.start
        self.plot_draw_bg = False
        return self.current_pos

    def render(self, mode: str = "rgb_array"):
        if mode == "human":
            if self.plot_fig is None:
                self.plot_fig = plt.figure()
                self.plot_draw_bg = False

            if self.plot_ax is None:
                self.plot_ax = self.plot_fig.add_subplot(111)

            if not self.plot_draw_bg:
                self.plot_ax.imshow(self.map)
                self.plot_draw_bg = True

            points = self.plot_ax.plot(
                self.current_pos.x, self.current_pos.y, "o", color="red"
            )
            plt.pause(self.plot_display_delay)
            points.pop().remove()

        elif mode == "rgb_array":
            render_img = np.dstack(
                (
                    np.full_like(self.map, fill_value=CellColorRGB.WALL[0]),
                    np.full_like(self.map, fill_value=CellColorRGB.WALL[1]),
                    np.full_like(self.map, fill_value=CellColorRGB.WALL[2]),
                )
            ).astype(np.uint8)

            render_img[self.map == CellCode.CLEAN, :] = CellColorRGB.CLEAN

            for y, x in self.goals:
                render_img[y, x, :] = CellColorRGB.GOAL

            for y, x in self.traps:
                render_img[y, x, :] = CellColorRGB.TRAP

            return render_img

        else:
            super().render(mode=mode)

    def close(self):
        if self.plot_fig is not None:
            plt.close(self.plot_fig)

        self.plot_fig = self.plot_ax = None

    def seed(self, seed=None):
        if seed is not None:
            self.random_state = seed

        return [self.random_state]


def _test_mode_rgb_array():
    fig, ax = plt.subplots(1, 1)

    env = Gridworld(traps_ends_episode=True)
    env.seed(8)

    state = env.reset()
    map_img = env.render(mode="rgb_array")
    ax.imshow(map_img)

    for epi in np.arange(1, 10 + 1):
        ax.set_title(f"Episode: {epi} of 10")
        state = env.reset()
        done = False

        while not done:
            state, _, done, _ = env.step(env.action_space.sample())
            x = state.x
            y = state.y
            points = ax.plot(x, y, "o", color="red")
            plt.pause(0.1)
            points.pop().remove()

    plt.show()


def _test_mode_human():
    env = Gridworld(traps_ends_episode=True)
    env.seed(8)

    for epi in np.arange(1, 3 + 1):
        state = env.reset()
        done = False

        while not done:
            env.render(mode="human")
            state, _, done, _ = env.step(env.action_space.sample())


if __name__ == "__main__":
    # _test_mode_rgb_array()
    _test_mode_human()
