# Explanations: SHAP and permutation importance for every task, seed, model and fold
# (honest split only), then feature rankings.
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.exceptions import ConvergenceWarning

from data_loading import FEATURES
from harness import get_task, make_pipe, get_cv, check_no_leak
from models import MODEL_NAMES, make_model

TASKS = ["D1", "D3", "D4"]
SEEDS = range(10)
IMP_FILE = "results/importances.csv"
RANK_FILE = "results/rankings.csv"


def shap_values(name, model, X_train_s, X_test_s, seed):
    # SHAP values on the scaled data, for class 1 (PD)
    if name == "LogReg":
        explainer = shap.LinearExplainer(model, shap.maskers.Independent(X_train_s))
        sv = explainer.shap_values(X_test_s)
    elif name in ["RF", "XGB"]:
        sv = shap.TreeExplainer(model).shap_values(X_test_s)
        if isinstance(sv, list):  # older shap: one array per class
            sv = sv[1]
        elif sv.ndim == 3:        # newer shap: (rows, features, classes)
            sv = sv[:, :, 1]
    else:  # SVM and MLP
        np.random.seed(seed)
        if name == "SVM":
            f = model.decision_function
        else:
            f = lambda x: model.predict_proba(x)[:, 1]
        background = shap.kmeans(X_train_s, 20)
        explainer = shap.KernelExplainer(f, background)
        sv = explainer.shap_values(X_test_s, nsamples=200, silent=True)
    return np.asarray(sv)


def direction(x, sv):
    # correlation between feature value and its SHAP value:
    # > 0 means a higher value pushes towards PD, < 0 towards healthy
    if np.std(x) == 0 or np.std(sv) == 0:
        return np.nan
    return np.corrcoef(x, sv)[0, 1]


def run_task(task, seeds=SEEDS):
    X, y, groups = get_task(task)
    rows = []
    mlp_warnings = 0
    for seed in seeds:
        t0 = time.time()
        cv = get_cv(seed, honest=True)
        check_no_leak(cv, X, y, groups)
        for fold, (train, test) in enumerate(cv.split(X, y, groups)):
            X_tr, X_te = X.iloc[train], X.iloc[test]
            for name in MODEL_NAMES:
                pipe = make_pipe(make_model(name, seed))
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    pipe.fit(X_tr, y[train])
                if name == "MLP":
                    mlp_warnings += sum(issubclass(w.category, ConvergenceWarning) for w in caught)

                # the data exactly as the model sees it (imputed + scaled)
                X_tr_s = pipe[:-1].transform(X_tr)
                X_te_s = pipe[:-1].transform(X_te)
                sv = shap_values(name, pipe.named_steps["model"], X_tr_s, X_te_s, seed)

                perm = permutation_importance(pipe, X_te, y[test], scoring="balanced_accuracy",
                                              n_repeats=10, random_state=seed)

                for j, feat in enumerate(FEATURES):
                    rows.append({"task": task, "seed": seed, "fold": fold, "model": name,
                                 "feature": feat,
                                 "shap": np.mean(np.abs(sv[:, j])),
                                 "perm": perm.importances_mean[j],
                                 "shap_direction": direction(X_te_s[:, j], sv[:, j])})
        print("  %s seed %d done (%.0f s)" % (task, seed, time.time() - t0), flush=True)
    return pd.DataFrame(rows), mlp_warnings


def make_rankings(imp):
    # average over the 5 folds of a seed, then rank (1 = most important)
    mean_imp = imp.groupby(["task", "model", "seed", "feature"])[["shap", "perm"]].mean().reset_index()
    rng = np.random.default_rng(0)  # fixed seed for random tie-breaking
    rows = []
    for (task, model, seed), g in mean_imp.groupby(["task", "model", "seed"]):
        for method in ["shap", "perm"]:
            values = g[method].values
            tie_break = rng.random(len(values))
            order = np.lexsort((tie_break, -values))  # sort by value (high first), ties at random
            ranks = np.empty(len(values), dtype=int)
            ranks[order] = np.arange(1, len(values) + 1)
            for feat, v, r in zip(g["feature"], values, ranks):
                rows.append({"task": task, "model": model, "seed": seed, "method": method,
                             "feature": feat, "importance": v, "rank": r})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    all_imp = []
    for task in TASKS:
        print("Task", task, flush=True)
        imp, mlp_warn = run_task(task)
        all_imp.append(imp)
        pd.concat(all_imp).to_csv(IMP_FILE, index=False)  # save after each task
        print("  saved %s (MLP ConvergenceWarnings in %s: %d)" % (IMP_FILE, task, mlp_warn), flush=True)

    imp = pd.concat(all_imp)
    ranks = make_rankings(imp)
    ranks.to_csv(RANK_FILE, index=False)
    print("saved", RANK_FILE)

    # top 5 SHAP features per task: how often in the top 5 across models and seeds
    shap_ranks = ranks[ranks["method"] == "shap"]
    for task in TASKS:
        t = shap_ranks[shap_ranks["task"] == task]
        n_runs = t.groupby(["model", "seed"]).ngroups
        top5 = (t[t["rank"] <= 5].groupby("feature").size()
                .reindex(FEATURES, fill_value=0).sort_values(ascending=False).head(5))
        print("\n%s: top 5 SHAP features (times in top 5, out of %d model-seed runs)" % (task, n_runs))
        print(top5.to_string())

    # direction check for D4, LogReg
    d4 = imp[(imp["task"] == "D4") & (imp["model"] == "LogReg")]
    direc = d4.groupby("feature")["shap_direction"].agg(
        mean_corr="mean", share_towards_PD=lambda s: (s > 0).mean()).reindex(FEATURES)
    print("\nD4 LogReg direction (mean_corr > 0: higher value -> towards PD)")
    print(direc.round(3).to_string())
