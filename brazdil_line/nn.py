import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import sklearn.pipeline
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.metrics
import matplotlib.pyplot as plt


class Model(nn.Module):
    def __init__(self, n_inputs: int):
        super(Model, self).__init__()

        self.weights = nn.Sequential(
            self.seq(n_inputs, 64),
            self.seq(64, 64),
            self.seq(64, 48),
            self.seq(48, 4, final_layer=True),
        )

    def seq(self, n_inputs: int, n_outputs: int, final_layer: bool = False):
        if final_layer:
            w = nn.Sequential(
                nn.Linear(n_inputs, n_outputs),
            )

        else:
            w = nn.Sequential(
                nn.Linear(n_inputs, n_outputs, bias=False),
                nn.BatchNorm1d(n_outputs),
                nn.ReLU(inplace=True),
            )

        return w

    def forward(self, X):
        return self.weights(X)


device = "cuda"

data = pd.read_csv("metadata/metafeatures_500.csv", index_col=0)

X = data.iloc[:, :-4]
y = (data.iloc[:, -4:] > 0.0).astype(int)

X[np.isnan(X)] = -1.0

test_size = 0.1

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=test_size
)

assert X_test.shape[1] == X_train.shape[1]
assert int(y_test.size) == int(np.ceil(y.size * test_size))
assert int(y_train.size) == int(y.size * (1 - test_size))

pipeline = sklearn.pipeline.Pipeline(
    (
        ("scaler", sklearn.preprocessing.StandardScaler()),
        ("pca", sklearn.decomposition.PCA(0.999)),
    )
).fit(X_train)

X_train = pipeline.transform(X_train)
X_test = pipeline.transform(X_test)

print(X_train.shape)

X_train = torch.Tensor(X_train).to(device)
y_train = torch.Tensor(y_train.values.reshape(-1, 4)).to(device)

model = Model(X_train.shape[1]).to(device)

criterion = nn.BCEWithLogitsLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0155)

assert y_train.shape[1] == y_test.shape[1] == 4

n_iter = 8000
it_to_print = 1000

losses = np.zeros(n_iter)

for i in np.arange(n_iter):
    optim.zero_grad()
    preds = model(X_train)
    loss = criterion(preds, y_train)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    optim.step()
    losses[i] = loss.item()

    if (i + 1) % it_to_print == 0:
        print(i + 1, ":", loss.item())

with torch.no_grad():
    X_test = torch.Tensor(X_test).to(device)

    preds_test = (model(X_test).squeeze().detach().cpu().numpy() >= 0).astype(
        int, copy=False
    )
    acc_test = np.mean(preds_test == y_test, axis=0).values

    preds_train = (model(X_train).squeeze().detach().cpu().numpy() >= 0).astype(
        int, copy=False
    )
    acc_train = np.mean(preds_train == y_train.detach().cpu().numpy(), axis=0)

    mean_pos = np.mean(y_test.values, axis=0)
    maj = np.maximum(mean_pos, 1.0 - mean_pos)

    print("acc train:", acc_train)
    print("acc test:", acc_test)
    print("maj:", maj)

plt.plot(losses)
plt.show()
