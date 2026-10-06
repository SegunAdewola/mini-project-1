
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # MP1: Comparing Classifiers on the Dry Bean Dataset
# CS 5330, Oluwasegun Adewola
#
# Trained models are cached in `cache/`.

# %%
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("ucimlrepo") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ucimlrepo"], check=True)

# %%
import copy
import random
import time
from itertools import product
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
FORCE_RETRAIN = False
CACHE_DIR = Path("cache")
FIGURES_DIR = Path("report/figures")
CACHE_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cached(name, train_fn):
    """Load `cache/<name>.joblib` if it exists, otherwise run `train_fn` and save its result."""
    path = CACHE_DIR / f"{name}.joblib"
    if path.exists() and not FORCE_RETRAIN:
        print(f"loaded cached {name}")
        return joblib.load(path)
    result = train_fn()
    joblib.dump(result, path, compress=3)
    return result


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")


# %% [markdown]
# ## Step 1: Load and split the data
# Exact duplicate rows are dropped first so the same bean can't land in both train and test.
# Stratified 70/15/15 split; the scaler is fit on the training set only.

# %%
raw_path = CACHE_DIR / "dry_bean.csv"
if raw_path.exists():
    df = pd.read_csv(raw_path)
else:
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=602)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    df.to_csv(raw_path, index=False)

n_raw = len(df)
df = df.drop_duplicates().reset_index(drop=True)
feature_names = [c for c in df.columns if c != "Class"]
encoder = LabelEncoder()
y = encoder.fit_transform(df["Class"])
X = df[feature_names].to_numpy(dtype=np.float64)
class_names = list(encoder.classes_)

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15 / 0.85, stratify=y_trainval, random_state=SEED
)
scaler = StandardScaler().fit(X_train)
X_train, X_val, X_test = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

print(f"raw {n_raw}, duplicates dropped {n_raw - len(df)}, kept {len(df)}")
print(f"train {len(y_train)} / val {len(y_val)} / test {len(y_test)}")
pd.DataFrame(
    {name: np.bincount(labels) for name, labels in [("train", y_train), ("val", y_val), ("test", y_test)]},
    index=class_names,
)

# %% [markdown]
# ## Step 2: Screen the traditional classifiers
# Every configuration is fit on train and scored on validation. Highest validation macro-F1 wins.

# %%
classical_grids = {
    "Logistic regression": (lambda **p: LogisticRegression(max_iter=5000, random_state=SEED, **p),
                            {"C": [0.01, 0.1, 1, 10, 100]}),
    "k-NN": (lambda **p: KNeighborsClassifier(**p),
             {"n_neighbors": [1, 3, 5, 9, 15, 25], "weights": ["uniform", "distance"]}),
    "Decision tree": (lambda **p: DecisionTreeClassifier(random_state=SEED, **p),
                      {"max_depth": [5, 10, 15, None], "min_samples_leaf": [1, 5, 10]}),
    "Random forest": (lambda **p: RandomForestClassifier(random_state=SEED, n_jobs=-1, **p),
                      {"max_depth": [None, 15], "max_features": ["sqrt", 0.5], "n_estimators": [100, 300]}),
    "SVM": (lambda **p: SVC(random_state=SEED, **p),
            {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1], "kernel": ["linear", "rbf", "poly"]}),
}


def expand(grid):
    combos = [dict(zip(grid, values)) for values in product(*grid.values())]
    unique = []
    for params in combos:
        if params.get("kernel") == "linear":  # gamma does nothing for a linear kernel
            params = {k: v for k, v in params.items() if k != "gamma"}
        if params not in unique:
            unique.append(params)
    return unique


def screen_classical():
    rows, best_models = [], {}
    for name, (make_model, grid) in classical_grids.items():
        best_f1 = -1
        for params in expand(grid):
            model = make_model(**params)
            start = time.perf_counter()
            model.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - start
            val_pred = model.predict(X_val)
            rows.append({"model": name, "params": params, "val_accuracy": accuracy_score(y_val, val_pred),
                         "val_macro_f1": macro_f1(y_val, val_pred), "fit_seconds": fit_seconds})
            if rows[-1]["val_macro_f1"] > best_f1:
                best_f1, best_models[name] = rows[-1]["val_macro_f1"], model
    return pd.DataFrame(rows), best_models


classical_grid, classical_models = cached("classical", screen_classical)

screening = (
    classical_grid.loc[classical_grid.groupby("model")["val_macro_f1"].idxmax()]
    .sort_values("val_macro_f1", ascending=False)
    .reset_index(drop=True)
)
screening["tuning_seconds"] = screening["model"].map(classical_grid.groupby("model")["fit_seconds"].sum())
classical_name = screening.iloc[0]["model"]
classical_model = classical_models[classical_name]
print("chosen:", classical_name, screening.iloc[0]["params"])
screening

# %%
svm_grid = classical_grid.query("model == 'SVM'").copy()
svm_grid["kernel"] = svm_grid["params"].map(lambda p: p["kernel"])
print("best validation macro-F1 per SVM kernel:")
print(svm_grid.groupby("kernel")["val_macro_f1"].max().round(4))
print("worst SVM config:", round(svm_grid["val_macro_f1"].min(), 4))

# %% [markdown]
# ## Step 3: MLP, two variants
# `SimpleMLP` from class. Shallow: one hidden layer of 64. Deep: 128-64-32 with dropout 0.2.
# Adam (lr 0.001), batch 64, up to 300 epochs, early stopping on validation loss (patience 25).

# %%
class SimpleMLP(nn.Module):
    def __init__(self, in_features, hidden_sizes, n_classes, dropout=0.0):
        super().__init__()
        layers, prev_size = [], in_features
        for size in hidden_sizes:
            layers += [nn.Linear(prev_size, size), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = size
        layers.append(nn.Linear(prev_size, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_loader(X, y, batch_size, shuffle):
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=torch.Generator().manual_seed(SEED))


@torch.no_grad()
def predict(model, X):
    model.eval()
    return model(torch.tensor(X, dtype=torch.float32, device=DEVICE)).argmax(dim=1).cpu().numpy()


def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, n_correct, n_seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y_batch)
            n_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            n_seen += len(y_batch)
    return total_loss / n_seen, n_correct / n_seen


def train_mlp(hidden_sizes, dropout, max_epochs=300, patience=25):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    model = SimpleMLP(len(feature_names), hidden_sizes, len(class_names), dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = make_loader(X_train, y_train, 64, shuffle=True)
    val_loader = make_loader(X_val, y_val, 1024, shuffle=False)

    history, best_val_loss, best_epoch, best_state = [], float("inf"), 0, None
    start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "train_accuracy": train_acc, "val_accuracy": val_acc})
        if val_loss < best_val_loss:
            best_val_loss, best_epoch, best_state = val_loss, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    val_pred = predict(model, X_val)
    return {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "history": pd.DataFrame(history),
        "best_epoch": best_epoch,
        "train_seconds": time.perf_counter() - start,
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "val_accuracy": accuracy_score(y_val, val_pred),
        "val_macro_f1": macro_f1(y_val, val_pred),
    }


mlp_variants = {"shallow": ([64], 0.0), "deep": ([128, 64, 32], 0.2)}
mlp_runs = {name: cached(f"mlp_{name}", lambda h=h, d=d: train_mlp(h, d)) for name, (h, d) in mlp_variants.items()}

variant_table = pd.DataFrame([
    {"variant": name, "hidden": h, "dropout": d, "params": run["n_parameters"],
     "val_accuracy": run["val_accuracy"], "val_macro_f1": run["val_macro_f1"],
     "best_epoch": run["best_epoch"], "epochs_run": len(run["history"]), "seconds": run["train_seconds"]}
    for name, (h, d), run in zip(mlp_runs, mlp_variants.values(), mlp_runs.values())
]).sort_values("val_macro_f1", ascending=False)
mlp_name = variant_table.iloc[0]["variant"]
hidden, dropout = mlp_variants[mlp_name]
mlp_model = SimpleMLP(len(feature_names), hidden, len(class_names), dropout).to(DEVICE)
mlp_model.load_state_dict(mlp_runs[mlp_name]["state_dict"])
print("chosen:", mlp_name)
variant_table

# %% [markdown]
# ## Figures
# Same palette as the main repo: blue/orange for the two series, a blue ramp for the confusion matrices.

# %%
BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#b9b8b3", "#0b0b0b", "#52514e"
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False, "legend.fontsize": 7,
    "savefig.bbox": "tight", "savefig.dpi": 300,
})


def save(fig, name):
    fig.savefig(FIGURES_DIR / f"{name}.pdf")
    plt.show()


# %%
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.3), sharey=True)
for ax, (name, run) in zip(axes, mlp_runs.items()):
    h = run["history"]
    ax.plot(h["epoch"], h["train_loss"], color=BLUE, linewidth=1.5, label="Train")
    ax.plot(h["epoch"], h["val_loss"], color=ORANGE, linewidth=1.5, linestyle="--", label="Validation")
    ax.axvline(run["best_epoch"], color=MUTED, linewidth=0.8, linestyle=":")
    ax.text(run["best_epoch"], ax.get_ylim()[1] * 0.95, f" best epoch {run['best_epoch']}", color=MUTED, fontsize=7, va="top")
    ax.set_title(f"{name.capitalize()} MLP")
    ax.set_xlabel("Epoch")
axes[0].set_ylabel("Cross-entropy loss")
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
fig.tight_layout()
save(fig, "mlp_loss_curves")

# %% [markdown]
# ## Step 4: Test set evaluation
# The only place the test set is used.

# %%
test_predictions = {classical_name: classical_model.predict(X_test), f"MLP ({mlp_name})": predict(mlp_model, X_test)}

results = pd.DataFrame({
    label: {"accuracy": accuracy_score(y_test, pred), "macro_f1": macro_f1(y_test, pred),
            "errors": int((pred != y_test).sum())}
    for label, pred in test_predictions.items()
})
per_class_f1 = pd.DataFrame(
    {label: f1_score(y_test, pred, average=None) for label, pred in test_predictions.items()}, index=class_names
)
display(results.round(4))
display(per_class_f1.round(4))

# %%
for label, pred in test_predictions.items():
    cm = confusion_matrix(y_test, pred)
    recall = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.imshow(recall, cmap=BLUE_RAMP, vmin=0, vmax=1)
    ax.grid(False)
    for i, j in product(range(len(class_names)), repeat=2):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6.5, color="white" if recall[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"{label}, test set")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    save(fig, "confusion_classical" if label == classical_name else "confusion_mlp")

# %% [markdown]
# ## Step 5: Evidence for the discussion

# %%
print(f"{classical_name}: {classical_model.n_support_.sum()} support vectors")
print(f"MLP: {mlp_runs[mlp_name]['n_parameters']} parameters, {len(y_train) / mlp_runs[mlp_name]['n_parameters']:.1f} training rows per parameter")
print(f"SVM tuning {svm_grid['fit_seconds'].sum():.1f}s, MLP training (both variants) {variant_table['seconds'].sum():.1f}s")

# %%
class TorchWrapper:
    """Lets sklearn's permutation_importance call the MLP."""
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        return self

    def predict(self, X):
        return predict(self.model, X)


importances = {}
for label, model in [(classical_name, classical_model), (f"MLP ({mlp_name})", TorchWrapper(mlp_model))]:
    result = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=SEED,
                                    scoring=lambda est, X, y: macro_f1(y, est.predict(X)))
    importances[label] = pd.Series(result.importances_mean, index=feature_names)
importances = pd.DataFrame(importances)
top = importances.loc[importances.mean(axis=1).sort_values(ascending=False).index[:10]][::-1]

fig, ax = plt.subplots(figsize=(3.5, 3.0))
y_pos = np.arange(len(top))
for offset, label, color in zip([0.19, -0.19], top.columns, [BLUE, ORANGE]):
    ax.barh(y_pos + offset, top[label], height=0.34, color=color, label=label)
ax.set_yticks(y_pos, top.index)
ax.set_xlabel("Macro-F1 drop when feature is shuffled")
ax.grid(axis="y", visible=False)
ax.legend(loc="lower right")
fig.tight_layout()
save(fig, "feature_importance")
importances.sort_values(classical_name, ascending=False).round(4)
