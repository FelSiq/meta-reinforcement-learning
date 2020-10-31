import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import xgboost

import utils


params = {
    "learning_rate": 0.08,
    "max_depth": 5,
    "min_child_weight": 30,
    "n_estimators": 3000,
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

for artificial in [False, True]:
    fig, axes = plt.subplots(1, 4, figsize=(20, 15), sharex=True)

    feat_imp = dict()
    X, y = utils.get_metadata(500, artificial=artificial)

    y = y > 0

    for i in np.arange(4):
        model = xgboost.XGBClassifier(**params).fit(X, y.iloc[:, i])
        imp = pd.Series(model.get_booster().get_fscore()).sort_values(ascending=True)
        imp.plot(kind="barh", ax=axes[i])

    fig.tight_layout()
    plt.show()
