"""Utilities available to other modules."""
import typing as t

import pandas as pd


def get_metadata(
    num_train_episodes: int, artificial: bool, num_base_models: int = 4
) -> t.Tuple[pd.DataFrame, pd.DataFrame]:

    metadata_path = (
        f"metadata/metafeatures_{num_train_episodes}"
        f"{'_artificial' if artificial else ''}.csv"
    )

    data = pd.read_csv(metadata_path, index_col=0)
    X = data.iloc[:, :-num_base_models]
    y = data.iloc[:, -num_base_models:]

    return X, y


def binsearch(y):
    start = 0
    ind = y.shape[0]
    end = y.shape[0] - 1
    while start <= end:
        middle = start + (end - start) // 2
        if pd.isna(y.iloc[middle, :]).all():
            ind = middle
            end = middle - 1
        else:
            start = middle + 1
    return ind
