import typing as t

import pandas as pd


def get_metadata(
    num_train_episodes: int, artificial: bool, num_base_models: int = 4
) -> t.Tuple[pd.DataFrame, pd.DataFrame]:

    metadata_path = (
        f"metadata/metafeatures_{num_train_episodes}"
        f"{'_artificial' if artificial else ''}.csv"
    )

    data = pd.read_csv(metadata_path)
    X = data.iloc[:, :-num_base_models]
    y = data.iloc[:, -num_base_models:]

    return X, y