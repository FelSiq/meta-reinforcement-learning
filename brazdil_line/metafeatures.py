"""Metafeatures build for the Gridworld environment."""
import typing as t

import numpy as np
import scipy.stats


def ft_goal_dist_euclid(env):
    start = env.start
    goals = env.goals

    dists = np.linalg.norm(np.array(goals, dtype=float) - start, ord=2, axis=0)

    return dists


def ft_goal_dist_manh(env):
    start = env.start
    goals = env.goals

    dists = np.linalg.norm(np.array(goals, dtype=float) - start, ord=1, axis=0)

    return dists


def ft_trap_dist_euclid(env):
    start = env.start
    traps = env.traps

    dists = np.linalg.norm(np.array(traps, dtype=float) - start, ord=2, axis=0)

    return dists


def ft_trap_dist_manh(env):
    start = env.start
    traps = env.traps

    dists = np.linalg.norm(np.array(traps, dtype=float) - start, ord=1, axis=0)

    return dists


def _in_radius_vals(env, values: np.ndarray, radius_prop: float):
    assert 0 < radius_prop <= 1.0

    width = env.width
    height = env.width
    start = env.start
    traps = env.traps

    radius = radius_prop * min(width, height)

    return values[values <= radius]


def ft_goal_radial_dist(env, radius_prop: float):
    euclid_dists = ft_goal_dist_euclid(env)
    return _in_radius_vals(env, values=euclid_dists, radius_prop=radius_prop)


def ft_trap_radial_dist(env, radius_prop: float):
    euclid_dists = ft_trap_dist_euclid(env)
    return _in_radius_vals(env, values=euclid_dists, radius_prop=radius_prop)


summary_functions = {
    "mean": np.mean,
    "std": np.std,
    "max": np.max,
    "min": np.min,
    "median": np.median,
    "kurtosis": scipy.stats.kurtosis,
    "skewness": scipy.stats.skew,
    "sum": np.sum,
    "len": len,
}


def summarize(
    feature: str, values: t.Union[float, np.ndarray]
) -> t.Tuple[t.List[str], t.List[float]]:
    if np.isscalar(values):
        return [feature], [values]

    mtf_names = []  # type: t.List[str]
    mtf_vals = []  # type: t.List[float]

    for summ_name, summ_func in summary_functions.items():
        mtf_vals.append(summ_func(values))
        mtf_names.append(f"{feature}.{summ_name}")

    return mtf_names, mtf_vals


def _test():
    ft = ["ab", "cd", "ef"]
    vals = [[1.0, 2.0], [-7, 7], 0]

    for ft_name, ft_vals in zip(ft, vals):
        print(summarize(ft_name, ft_vals))


if __name__ == "__main__":
    _test()
