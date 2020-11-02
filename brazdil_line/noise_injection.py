import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost
import sklearn.model_selection

import utils


num_it = 30
artificial = False
num_folds = 10
n_estimators = 8000
n_repeats = 30

params = {
    "learning_rate": 0.08,
    "max_depth": 5,
    "min_child_weight": 30,
    "gamma": 0.5,
    "alpha": 0.5,
    "lambda": 400,
    "subsample": 0.7,
    "colsample_bytree": 0.3,
    "objective": "binary:logistic",
    "scale_pos_weight": 0.9,
    "seed": 16,
    "gpu_id": 0,
    "tree_method": "gpu_hist",
}

X, y = utils.get_metadata(num_train_episodes=500, artificial=artificial)

y = y > 0.0


np.random.seed(16)

noise_std_props = np.linspace(0, 10, num_it)


baseline_value = np.array(
    [
        0.647,
        0.680,
        0.679,
        0.751,
    ]
)

baseline_std = np.array(
    [
        0.033,
        0.029,
        0.030,
        0.015,
    ]
)

data_splitter = sklearn.model_selection.RepeatedStratifiedKFold(
    n_splits=num_folds, n_repeats=n_repeats
)


try:
    os.mkdir("./backup")

except OSError:
    pass


for j in np.arange(y.shape[1]):
    print(f"Algorithm: {y.columns[j]}")
    y_cur = y.iloc[:, j]

    try:
        performance = np.load(f"./backup/{y.columns[j]}_perf.npy")
        print(f"Got backup files for class {y.columns[j]}.")

    except FileNotFoundError:
        performance = np.full((num_it, 2), fill_value=np.nan)

    for i, noise_std_prop in enumerate(noise_std_props):
        print(f"{i + 1:<{3}} / {noise_std_props.size}...", end="\r")
        X_noise = X + noise_std_prop * X.std(axis=0).values * np.random.randn(*X.shape)
        dtrain = xgboost.DMatrix(X_noise, label=y_cur)
        res = xgboost.cv(
            metrics="auc",
            dtrain=dtrain,
            params=params,
            num_boost_round=n_estimators,
            early_stopping_rounds=500,
            stratified=True,
            folds=data_splitter,
            seed=32,
        )

        performance[i, :] = res.iloc[-1, :][["test-auc-mean", "test-auc-std"]]

        np.save(f"./backup/{y.columns[j]}_perf.npy", performance)

    plt.errorbar(
        x=noise_std_props,
        y=performance[:, 0],
        yerr=performance[:, 1],
        label=y.columns[j],
    )

plt.xlim((0, noise_std_props[-1]))
plt.ylim((0.45, 0.80))
plt.xlabel("Prop. de ruído")
plt.ylabel("AUC")
plt.hlines(
    y=0.5,
    xmin=0,
    xmax=noise_std_props[-1],
    linestyle="dashed",
    color="red",
    label="Referência",
)
"""
plt.hlines(
    y=[baseline_value + baseline_std, baseline_value - baseline_std],
    xmin=0,
    xmax=1,
    linestyle="dotted",
    color="red",
)
"""

plt.legend()
plt.show()
