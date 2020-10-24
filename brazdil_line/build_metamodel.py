import argparse

import matplotlib.pyplot as plt
import seaborn
import xgboost
import pandas as pd
import sklearn.metrics
import numpy as np

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


if args.artificial:
    data = pd.read_csv(f"metadata/metafeatures_{args.num_train_episodes}_artificial.csv")
    X = data.iloc[:, :-num_base_models]
    y = data.iloc[:, -num_base_models:]

    assert y.shape[1] == num_base_models

else:
    data = pd.read_csv(f"metadata/metafeatures_{args.num_train_episodes}.csv")
    X = data.iloc[:, :-num_base_models]
    y = data.iloc[:, -num_base_models:]

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

print("Labels standard deviation:", y.std(axis=0))
print("Labels converged class distribution:", (y > 0.0).mean(axis=0))
print(res_reg)
print(res_cls)
