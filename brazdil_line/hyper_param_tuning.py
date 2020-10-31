import xgboost
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
import sklearn.model_selection
import scipy.stats.distributions

import utils


def fit(
    X,
    y,
    params,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50,
    plot: bool = False,
    show: bool = True,
):
    dtrain = xgboost.DMatrix(X.values, label=y.values)

    add_n_estimators = "n_estimators" in params

    if add_n_estimators:
        n_estimators = params.pop("n_estimators")

    metric = "auc" if classification else "rmse"

    results = xgboost.cv(
        dtrain=dtrain,
        params=params,
        num_boost_round=num_boost_round,
        nfold=10,
        stratified=classification,
        early_stopping_rounds=early_stopping_rounds,
        metrics=metric,
    )

    if add_n_estimators:
        params["n_estimators"] = n_estimators

    print(results)

    err_mean_train, err_std_train, err_mean_test, err_std_test = results.iloc[
        -1, :
    ].values

    print(
        f"Margin ({metric}): {abs(err_mean_train - err_mean_test):.6f} pm {err_std_test + err_std_train:.6f}"
    )

    if plot:
        plt.errorbar(
            x=np.arange(results.shape[0]),
            y=results[f"train-{metric}-mean"],
            yerr=results[f"train-{metric}-std"],
            label="train",
        )
        plt.errorbar(
            x=np.arange(results.shape[0]),
            y=results[f"test-{metric}-mean"],
            yerr=results[f"train-{metric}-std"],
            label="test",
        )
        plt.legend()

        if show:
            plt.show()


def random_search(model_params, param_test, X, y, n_iter):
    random_search = sklearn.model_selection.RandomizedSearchCV(
        estimator=xgboost.sklearn.XGBClassifier(**model_params),
        param_distributions=param_test,
        scoring="roc_auc",
        n_jobs=-1,
        cv=5,
        random_state=48,
        n_iter=n_iter,
    )

    random_search.fit(X, y)

    print(
        "(Random) Best configuration:",
        random_search.best_params_,
        "score:",
        random_search.best_score_,
    )

    return random_search.best_params_


def grid_search(model_params, param_test, X, y):
    grid_search = sklearn.model_selection.GridSearchCV(
        estimator=xgboost.sklearn.XGBClassifier(**model_params),
        param_grid=param_test,
        scoring="roc_auc",
        n_jobs=-1,
        cv=5,
    )

    grid_search.fit(X, y)

    print(
        "(Grid)   Best configuration:",
        grid_search.best_params_,
        "score:",
        grid_search.best_score_,
    )

    for param, v in grid_search.best_params_.items():
        low, high = param_test[param][0], param_test[param][-1]
        assert (
            low < v < high
        ), f"Best '{param}' value is in limit [{low}, {high}], consider expanding it"

    return grid_search.best_params_


cls_id = 0
subplot_row = 1
subplot_col = 4
cv_fit = True
plots = [True, False, False, True]
cached = False
classification = True

# Preparation
X, y = utils.get_metadata(num_train_episodes=3000, artificial=False)

if classification:
    y = (y <= 0.0) if y.size == 3000 else y > 0.0

y = y.iloc[:, cls_id]

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X,
    y,
    test_size=0.1,
    shuffle=True,
    random_state=33,
    stratify=y if classification else None,
)

"""
import sklearn.ensemble

means = X_train.mean(axis=0, skipna=True)
not_nan = ~means.isna()
X_train = X_train.loc[:, not_nan].fillna(means[not_nan])
X_test = X_test.loc[:, not_nan].fillna(means[not_nan])

model = sklearn.ensemble.IsolationForest(n_estimators=8000, n_jobs=-1, random_state=16, max_features=0.6)
model.fit(X_train, y_train)
y_preds = model.predict(X_test)
print("Test ROC AUC score:", sklearn.metrics.roc_auc_score(y_test, y_preds))
print("Test accuracy score:", sklearn.metrics.accuracy_score(y_test, y_preds))
maj_prop = np.mean(y_test)
print("Maj class prop:", max(maj_prop, 1 - maj_prop))

exit(2)
"""

assert X_train.shape[1] == X_test.shape[1]

params = {
    "learning_rate": 0.08,
    "max_depth": 5,
    "min_child_weight": 30,
    "gamma": 0.5,
    "alpha": 0.5,
    "lambda": 400,
    "subsample": 0.7,
    "colsample_bytree": 0.3,
    "objective": "binary:logistic" if classification else "reg:squarederror",
    "scale_pos_weight": 0.9,
    "seed": 16,
    "gpu_id": 0,
    "tree_method": "gpu_hist",
}

print(f"(Class id = {cls_id}) Initial parameters:")

for p, v in params.items():
    print(f"{p}: {v}")

if cv_fit and plots[0]:
    plt.subplot(subplot_row, subplot_col, 1)
    fit(
        X_train,
        y_train,
        params,
        num_boost_round=params["n_estimators"],
        early_stopping_rounds=500,
        plot=False,
        show=False,
    )

if classification:
    y_preds = xgboost.XGBClassifier(**params).fit(X_train, y_train).predict(X_test)
    print("Test ROC AUC score:", sklearn.metrics.roc_auc_score(y_test, y_preds))
    print("Test accuracy score:", sklearn.metrics.accuracy_score(y_test, y_preds))
    maj_prop = np.mean(y_test)
    print("Maj class prop:", max(maj_prop, 1 - maj_prop))

else:
    y_preds = xgboost.XGBRegressor(**params).fit(X_train, y_train).predict(X_test)
    print(
        "Test RMSE", sklearn.metrics.mean_squared_error(y_test, y_preds, squared=False)
    )
    print("Target std:", np.std(y_test))


exit(0)


# Hyper-parameter tunning step 01
# Hyper-parameters: max_depth, min_child_weight
best_param1 = None

if cached:
    if cls_id == 0:
        best_param1 = {"min_child_weight": 11, "max_depth": 9}

if best_param1 is None:
    param_test1 = {
        "max_depth": np.arange(7, 10, 1),
        "min_child_weight": np.arange(6, 32, 1),
    }

    best_param1 = grid_search(params, param_test1, X_train, y_train)

params.update(best_param1)

# Hyper-parameter tunning step 02
# Hyper-parameters: gamma
best_param2 = None

if cached:
    if cls_id == 0:
        best_param2 = {"gamma": 0.84}

if best_param2 is None:
    param_test2 = {
        "gamma": np.linspace(0, 1, 32),
    }

    best_param2 = grid_search(params, param_test2, X_train, y_train)

params.update(best_param2)

# Checking if any meaningful improvement
if cv_fit and plots[1]:
    plt.subplot(subplot_row, subplot_col, 2)
    fit(X_train, y_train, params, plot=True, show=False)

# Hyper-parameter tunning step 03
# Hyper-parameters: subsample, colsample_bytree
best_param3 = None

if cached:
    if cls_id == 0:
        best_param3 = {"subsample": 0.866, "colsample_bytree": 0.8}

if best_param3 is None:
    param_test3 = {
        "subsample": scipy.stats.distributions.uniform(loc=0.73, scale=0.14),
        "colsample_bytree": scipy.stats.distributions.uniform(loc=0.73, scale=0.14),
    }

    best_param3 = random_search(params, param_test3, X_train, y_train, n_iter=128)

    """
    param_test3 = {
        "subsample": np.linspace(0.1, 1, 10),
        "colsample_bytree": np.linspace(0.1, 1, 10),
    }

    best_param3 = grid_search(params, param_test3, X_train, y_train)
    """

params.update(best_param3)

# Checking if any meaningful improvement
if cv_fit and plots[2]:
    plt.subplot(subplot_row, subplot_col, 3)
    fit(X_train, y_train, params, plot=True, show=False)

# Hyper-parameter tunning step 04
# Hyper-parameters: alpha
best_param4 = None

if cached:
    if cls_id == 0:
        best_param4 = {"reg_alpha": 3.474}

if best_param4 is None:
    """
    param_test4 = {
        "reg_alpha": np.logspace(-4, 2, 20),
    }
    """

    param_test4 = {
        "reg_alpha": np.linspace(3, 4, 20),
    }

    best_param4 = grid_search(params, param_test4, X_train, y_train)

params.update(best_param4)


# Hyper-parameter tunning step 05
# Less lr, more trees
params["learning_rate"] = 0.01
params["n_estimators"] = 2300

# Checking if any meaningful improvement
if cv_fit and plots[3]:
    plt.subplot(subplot_row, subplot_col, 4)
    fit(X_train, y_train, params, plot=True, show=True)
