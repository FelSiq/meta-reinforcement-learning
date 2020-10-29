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


latex_cls_out = [
    "    "
    r"\cline{2-5} & "
    r"\textbf{Algoritmo} & "
    r"\textbf{$\text{ACC}_{\text{aleatório}}$} & "
    r"\textbf{$\text{ACC}_{\text{treino}} \pm \sigma_{\text{treino}}$} & "
    r"\textbf{$\text{ACC}_{\text{teste}} \pm \sigma_{\text{teste}}$} \\ \cline{2-5}"
]

latex_reg_out = [
    "    "
    r"\cline{2-5} & "
    r"\textbf{Algoritmo} & "
    r"\textbf{$\sigma_{y}$} & "
    r"\textbf{$\text{RMSE}_{\text{treino}} \pm \sigma_{\text{treino}}$} & "
    r"\textbf{$\text{RMSE}_{\text{teste}} \pm \sigma_{\text{teste}}$} \\ \cline{2-5}"
]


latex_names = {
    "sarsa": "SARSA",
    "mccontrol": "MC Control",
    "qlearning": "Q-Learning",
    "dqlearning": "D. Q-Learning",
}


for i, alg in enumerate(y.columns):
    train_mean_cls, train_std_cls, test_mean_cls, test_std_cls = res_cls[i].values.T

    combined_train_mean_cls, combined_train_std_cls = combine_stats(
        means=train_mean_cls, stds=train_std_cls, sample_size_per_group=y.shape[0]
    )
    combined_test_mean_cls, combined_test_std_cls = combine_stats(
        means=test_mean_cls, stds=test_std_cls, sample_size_per_group=y.shape[0]
    )

    train_mean_reg, train_std_reg, test_mean_reg, test_std_reg = res_reg[i].values.T

    combined_train_mean_reg, combined_train_std_reg = combine_stats(
        means=train_mean_reg, stds=train_std_reg, sample_size_per_group=y.shape[0]
    )
    combined_test_mean_reg, combined_test_std_reg = combine_stats(
        means=test_mean_reg, stds=test_std_reg, sample_size_per_group=y.shape[0]
    )

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
        latex_cls_out.append(
            f"{latex_names[alg.lower()]:<{15}} & {maj_cls_prop:.4f} & "
            f"{1.0 - combined_train_mean_cls:.4f} $\pm$ {combined_train_std_cls:.4f} & "
            fr"{1.0 - combined_test_mean_cls:.4f} $\pm$ {combined_test_std_cls:.4f} \\ \cline{{2-5}}"
        )
        latex_reg_out.append(
            f"{latex_names[alg.lower()]:<{15}} & {label_std:.4f} & "
            f"{combined_train_mean_reg:.4f} $\pm$ {combined_train_std_reg:.4f} & "
            fr"{combined_test_mean_reg:.4f} $\pm$ {combined_test_std_reg:.4f} \\ \cline{{2-5}}"
        )


if args.latex:
    print(r"    \hline")

    if args.artificial:
        print(
            r"    \multicolumn{5}{|c|}{\textbf{Meta-características \aspas{Artificial}}} \\"
        )

    else:
        print(
            r"    \multicolumn{5}{|c|}{\textbf{Meta-características \aspas{Elaboradas}}} \\"
        )

    print(r"    \hline")
    print(fr"    \multirow{{{2 * y.shape[1] + 4}}}{{*}}{{{args.num_train_episodes}}}")
    print(r"    & \multicolumn{4}{c|}{\textbf{Meta-classificação}} \\")
    print("\n    & ".join(latex_cls_out))

    print()

    print(r"    & \multicolumn{4}{c|}{\textbf{Meta-regressão}} \\")
    print("\n    & ".join(latex_reg_out))
    print(f"    \hline")
