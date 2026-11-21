# Shared tools for all experiments: load a task, build a pipeline, make the split, compute metrics.
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score, recall_score,
                             f1_score, matthews_corrcoef)

from data_loading import FEATURES, load_oxford, load_istanbul


def load_italian_vowel(path="data/processed/italian_vowel_features.csv"):
    df = pd.read_csv(path)
    df = df[df["group"].isin(["PD", "HEC"])].copy()
    df["label"] = (df["group"] == "PD").astype(int)  # 1 = PD, 0 = elderly healthy
    return df[FEATURES + ["person_id", "label"]]


def get_task(name):
    if name == "D1":
        df = load_oxford()
    elif name == "D3":
        df = load_istanbul()
    elif name == "D4":
        df = load_italian_vowel()
    else:
        raise ValueError("unknown task: " + name)
    X = df[FEATURES]
    y = df["label"].values
    groups = df["person_id"].values
    return X, y, groups


def make_pipe(model):
    # imputer and scaler are inside the pipeline, so they only learn from training data
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("model", model)])


def get_cv(seed, honest=True):
    if honest:
        return StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    # leaky record-wise split, ONLY for comparison
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)


def check_no_leak(cv, X, y, groups):
    for i, (train, test) in enumerate(cv.split(X, y, groups)):
        shared = set(groups[train]) & set(groups[test])
        if shared:
            raise RuntimeError("fold %d: %d people in both train and test, e.g. %s"
                               % (i, len(shared), sorted(shared)[:3]))


def get_metrics(y_true, y_pred, y_score):
    return {"bal_acc": balanced_accuracy_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_score),
            "sensitivity": recall_score(y_true, y_pred, pos_label=1),  # PD found
            "specificity": recall_score(y_true, y_pred, pos_label=0),  # healthy found
            "f1": f1_score(y_true, y_pred),
            "mcc": matthews_corrcoef(y_true, y_pred)}
