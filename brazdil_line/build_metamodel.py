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
    "--artificial",
    action="store_true",
    help="if true, load metafeatures from the 'artificial' set",
)
args = parser.parse_args()


if args.artificial:
    data = pd.read_csv("metadata/metafeatures_artificial.csv")
    X = data.iloc[:, :-num_base_models]
    y = data.iloc[:, -num_base_models:]

    assert y.shape[1] == num_base_models

else:
    raise NotImplementedError

res = []
params = {"objective": "reg:squarederror", "max_depth": 5, "gpu_id": 0}

np.random.seed(16)
random_seeds = np.random.randint(1000, size=y.shape[1])
KFold = sklearn.model_selection.KFold(n_splits=10)

for i in np.arange(num_base_models):
    print(f"Fitting {i + 1}th model...", end=" ")
    data = xgboost.DMatrix(X, label=y.iloc[:, i])
    cur_res = xgboost.cv(
        dtrain=data,
        params=params,
        metrics="rmse",
        num_boost_round=30,
        nfold=10,
        seed=random_seeds[i],
    )
    res.append(cur_res)
    print("Done.")

print("Labels standard deviation:", y.std(axis=0))
print(res)
