import typing as t
import argparse

import matplotlib.pyplot as plt
import seaborn
import xgboost
import pandas as pd
import sklearn.metrics
import numpy as np

import utils

num_base_models = 4

parser = argparse.ArgumentParser(description="Train a meta-regressor model")
parser.add_argument(
    "num_train_episodes",
    help="specify which data to be used from the training step",
    default=500,
    type=int,
)
parser.add_argument(
    "--artificial",
    action="store_true",
    help="if true, load metafeatures from the 'artificial' set",
)
parser.add_argument(
    "--nfold",
    help="number of folds in the K-fold cross validation",
    default=10,
    type=int,
)
parser.add_argument(
    "--num-boost-round",
    help="number of repetitions of the K-fold cross validation",
    default=30,
    type=int,
)
args = parser.parse_args()


X, y = utils.get_metadata(
    num_train_episodes=args.num_train_episodes,
    artificial=args.artificial,
    num_base_models=num_base_models,
)

assert y.shape[1] == num_base_models

res_reg = []
res_cls = []

params_reg = {"objective": "reg:squarederror", "max_depth": 5, "gpu_id": 0}
params_cls = {"objective": "binary:logistic", "max_depth": 5, "gpu_id": 0}

np.random.seed(16)
random_seeds = np.random.randint(1000, size=y.shape[1])
KFold = sklearn.model_selection.KFold(n_splits=10)


for i in np.arange(num_base_models):
    print(f"Fitting {i + 1}th model...", end=" ")

    data_reg = xgboost.DMatrix(X, label=y.iloc[:, i])
    data_cls = xgboost.DMatrix(X, label=y.iloc[:, i] > 0.0)

    cur_res_reg = xgboost.cv(
        dtrain=data_reg,
        params=params_reg,
        metrics="rmse",
        num_boost_round=args.num_boost_round,
        nfold=args.nfold,
        seed=random_seeds[i],
    )

    cur_res_cls = xgboost.cv(
        dtrain=data_cls,
        params=params_cls,
        metrics="error",
        num_boost_round=args.num_boost_round,
        nfold=args.nfold,
        seed=random_seeds[i],
    )

    res_reg.append(cur_res_reg)
    res_cls.append(cur_res_cls)

    print("Done.")


def combine_stats(
    means: np.ndarray, stds: np.ndarray, sample_size_per_group: int
) -> t.Tuple[float, float]:
    num_groups = means.size

    combined_mean = np.mean(means)

    total_inst = num_groups * sample_size_per_group

    total_var = (
        sample_size_per_group * np.sum(np.square(means - combined_mean))
        + (sample_size_per_group - 1) * np.sum(np.square(stds))
    ) / (total_inst - 1)

    return combined_mean, np.sqrt(total_var)


for i, alg in enumerate(y.columns):
    print("Algorithm:", alg)

    train_mean_cls, train_std_cls, test_mean_cls, test_std_cls = res_cls[i].values.T

    combined_train_mean_cls, combined_train_std_cls = combine_stats(
        means=train_mean_cls, stds=train_std_cls, sample_size_per_group=y.shape[0]
    )
    combined_test_mean_cls, combined_test_std_cls = combine_stats(
        means=test_mean_cls, stds=test_std_cls, sample_size_per_group=y.shape[0]
    )

    print(
        "  classification - Accuracy by random chance :",
        np.mean(y.iloc[:, i].values > 0.0),
    )
    print(
        "  classification - Train accuracy / std      :",
        1.0 - combined_train_mean_cls,
        combined_train_std_cls,
    )
    print(
        "  classification - Test accuracy / std       :",
        1.0 - combined_test_mean_cls,
        combined_test_std_cls,
    )

    train_mean_reg, train_std_reg, test_mean_reg, test_std_reg = res_reg[i].values.T

    combined_train_mean_reg, combined_train_std_reg = combine_stats(
        means=train_mean_reg, stds=train_std_reg, sample_size_per_group=y.shape[0]
    )
    combined_test_mean_reg, combined_test_std_reg = combine_stats(
        means=test_mean_reg, stds=test_std_reg, sample_size_per_group=y.shape[0]
    )

    print(
        "  regression     - target standard dev       :",
        np.std(y.iloc[:, i].values, ddof=1),
    )
    print(
        "  regression     - Train rmse / std          :",
        combined_train_mean_reg,
        combined_train_std_reg,
    )
    print(
        "  regression     - Test rmse / std           :",
        combined_test_mean_reg,
        combined_test_std_reg,
    )
