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
    help="number of trees (boosting rounds) while boosting",
    default=8000,
    type=int,
)
parser.add_argument(
    "--n-repetitions",
    help="number of repetitions of the K-fold cross validation",
    default=30,
    type=int,
)
parser.add_argument(
    "--latex",
    action="store_true",
    help="if true, generate the output as a latex tabular",
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

if args.num_train_episodes == 500:
    params_reg = {
        "learning_rate": 0.08,
        "max_depth": 3,
        "min_child_weight": 40,
        "gamma": 1.20,
        "alpha": 0.5,
        "lambda": 410.0,
        "subsample": 0.6,
        "colsample_bytree": 0.3,
        "objective": "reg:squarederror",
        "scale_pos_weight": 0.9,
        "seed": 16,
        "gpu_id": 0,
        "tree_method": "gpu_hist",
    }

    params_cls = {
        "learning_rate": 0.08,
        "max_depth": 3,
        "min_child_weight": 40,
        "gamma": 1.20,
        "alpha": 0.5,
        "lambda": 410.0,
        "subsample": 0.6,
        "colsample_bytree": 0.3,
        "objective": "binary:logistic",
        "scale_pos_weight": 0.9,
        "seed": 16,
        "gpu_id": 0,
        "tree_method": "gpu_hist",
    }

elif args.num_train_episodes == 1000:
    params_reg = {
        "learning_rate": 0.06,
        "max_depth": 5,
        "min_child_weight": 20,
        "gamma": 0.5,
        "alpha": 0.5,
        "lambda": 200,
        "subsample": 0.7,
        "colsample_bytree": 0.4,
        "objective": "reg:squarederror",
        "scale_pos_weight": 0.85,
        "seed": 16,
        "gpu_id": 0,
        "tree_method": "gpu_hist",
    }

    params_cls = {
        "learning_rate": 0.06,
        "max_depth": 5,
        "min_child_weight": 20,
        "gamma": 0.5,
        "alpha": 0.5,
        "lambda": 200,
        "subsample": 0.7,
        "colsample_bytree": 0.4,
        "objective": "binary:logistic",
        "scale_pos_weight": 0.85,
        "seed": 16,
        "gpu_id": 0,
        "tree_method": "gpu_hist",
    }

elif args.num_train_episodes == 3000:
    params_cls = {
        "learning_rate": 0.1,
        "max_depth": 5,
        "min_child_weight": 20,
        "gamma": 1.0,
        "alpha": 0.0,
        "lambda": 3.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "scale_pos_weight": 2,
        "seed": 16,
        "gpu_id": 0,
        "tree_method": "gpu_hist",
    }

    params_reg = {
        "learning_rate": 0.1,
        "max_depth": 5,
        "min_child_weight": 20,
        "gamma": 1.0,
        "alpha": 0.0,
        "lambda": 3.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "scale_pos_weight": 2,
        "seed": 16,
        "gpu_id": 0,
        "tree_method": "gpu_hist",
    }


np.random.seed(16)
random_seeds = np.random.randint(1000, size=y.shape[1])

fold_splitter_cls = sklearn.model_selection.RepeatedStratifiedKFold(
    n_splits=10, n_repeats=args.n_repetitions
)
fold_splitter_reg = sklearn.model_selection.RepeatedKFold(
    n_splits=10, n_repeats=args.n_repetitions
)


for i in np.arange(num_base_models):
    print(f"Fitting {i + 1}th model...", end=" ")

    data_reg = xgboost.DMatrix(X, label=y.iloc[:, i])
    data_cls = xgboost.DMatrix(X, label=y.iloc[:, i] > 0.0)

    cur_res_reg = xgboost.cv(
        dtrain=data_reg,
        params=params_reg,
        metrics="rmse",
        num_boost_round=args.num_boost_round,
        folds=fold_splitter_reg,
        seed=random_seeds[i],
        early_stopping_rounds=500,
    )

    cur_res_cls = xgboost.cv(
        dtrain=data_cls,
        params=params_cls,
        metrics=["error", "auc"],
        num_boost_round=args.num_boost_round,
        folds=fold_splitter_cls,
        seed=random_seeds[i],
        stratified=True,
        early_stopping_rounds=500,
    )

    res_reg.append(cur_res_reg.iloc[-1, :].values)
    res_cls.append(cur_res_cls.iloc[-1, :].values)

    print("Done.")


if args.latex:
    latex_cls_out = [
        [r"    & \textbf{Algoritmo}"],
        [r"    & \textbf{$\text{ACC}_{\text{maioria}}$}"],
        [r"    & \textbf{$\text{ACC}_{\text{treino}} \pm \sigma$}"],
        [r"    & \textbf{$\text{ACC}_{\text{teste}} \pm \sigma$}"],
        [r"    & \textbf{$\text{AUC}_{\text{treino}} \pm \sigma$}"],
        [r"    & \textbf{$\text{AUC}_{\text{teste}} \pm \sigma$}"],
    ]

    latex_reg_out = [
        [r"    & \textbf{Algoritmo}"],
        [r"    & \textbf{$\sigma_{y}$}"],
        [r"    & \textbf{$\text{RMSE}_{\text{treino}} \pm \sigma$}"],
        [r"    & \textbf{$\text{RMSE}_{\text{teste}} \pm \sigma$}"],
    ]

    latex_names = {
        "sarsa": "SARSA",
        "mccontrol": "MC Control",
        "qlearning": "Q-Learning",
        "dqlearning": "D. Q-Learning",
    }


for i, alg in enumerate(y.columns):
    (
        combined_train_mean_cls,
        combined_train_std_cls,
        combined_train_auc_mean_cls,
        combined_train_auc_std_cls,
        combined_test_mean_cls,
        combined_test_std_cls,
        combined_test_auc_mean_cls,
        combined_test_auc_std_cls,
    ) = res_cls[i]
    (
        combined_train_mean_reg,
        combined_train_std_reg,
        combined_test_mean_reg,
        combined_test_std_reg,
    ) = res_reg[i]

    pos_cls_prop = np.mean(y.iloc[:, i].values > 0.0)
    maj_cls_prop = max(pos_cls_prop, 1 - pos_cls_prop)
    label_std = np.std(y.iloc[:, i].values, ddof=1)

    if not args.latex:
        print("Algorithm:", alg)

        print(
            "  classification - Accuracy by random chance :",
            maj_cls_prop,
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

        print(
            "  regression     - target standard dev       :",
            label_std,
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

    else:
        latex_cls_out[0].append(f"{latex_names[alg.lower()]}")
        latex_cls_out[1].append(f"{maj_cls_prop:.3f}")
        latex_cls_out[2].append(
            f"{1.0 - combined_train_mean_cls:.3f} $\pm$ {combined_train_std_cls:.3f}"
        )
        latex_cls_out[3].append(
            fr"{1.0 - combined_test_mean_cls:.3f} $\pm$ {combined_test_std_cls:.3f}"
        )
        latex_cls_out[4].append(
            fr"{combined_train_auc_mean_cls:.3f} $\pm$ {combined_train_auc_std_cls:.3f}"
        )
        latex_cls_out[5].append(
            fr"{combined_test_auc_mean_cls:.3f} $\pm$ {combined_test_auc_std_cls:.3f}"
        )

        latex_reg_out[0].append(f"{latex_names[alg.lower()]}")
        latex_reg_out[1].append(f"{label_std:.3f}")
        latex_reg_out[2].append(
            f"{combined_train_mean_reg:.3f} $\pm$ {combined_train_std_reg:.3f}"
        )
        latex_reg_out[3].append(
            fr"{combined_test_mean_reg:.3f} $\pm$ {combined_test_std_reg:.3f}"
        )


if args.latex:
    print(r"    \hline")

    if args.artificial:
        print(
            r"    \multicolumn{6}{|c|}{\textbf{Meta-características \aspas{Artificial}}} \\"
        )

    else:
        print(
            r"    \multicolumn{6}{|c|}{\textbf{Meta-características \aspas{Elaboradas}}} \\"
        )

    print(r"    \hline")
    print(fr"    \multirow{{{2 * y.shape[1] + 4}}}{{*}}{{{args.num_train_episodes}}}")
    print(r"    & \multicolumn{5}{c|}{\textbf{Meta-classificação}} \\ \cline{2-6}")
    print("\n".join(map(lambda st: " & ".join(st) + r" \\ \cline{2-6}", latex_cls_out)))

    print()

    print(r"    & \multicolumn{5}{c|}{\textbf{Meta-regressão}} \\ \cline{2-6}")
    print("\n".join(map(lambda st: " & ".join(st) + r" \\ \cline{2-6}", latex_reg_out)))

    print(f"    \hline")
