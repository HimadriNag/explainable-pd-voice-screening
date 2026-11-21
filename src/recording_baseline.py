# "Recording-only" baseline: can we tell the groups apart using ONLY
# recording conditions (background noise, loudness, duration), no voice features?
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score

REC_FEATURES = ["noise_db", "loudness_db", "duration_s"]

# (kind, positive group, negative group)
TASKS = [("vowel", "PD", "HEC"),
         ("reading", "PD", "YHC"),
         ("reading", "PD", "HEC"),
         ("reading", "HEC", "YHC")]

SEEDS = range(10)


def run_task(check, kind, pos, neg):
    df = check[(check["kind"] == kind) & (check["group"].isin([pos, neg]))]
    X = df[REC_FEATURES].values
    y = (df["group"] == pos).astype(int).values
    groups = df["person_id"].values

    scores = []
    for seed in SEEDS:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        pred = np.zeros(len(y), dtype=int)
        for train, test in cv.split(X, y, groups):
            model = Pipeline([("scale", StandardScaler()),
                              ("lr", LogisticRegression(class_weight="balanced", max_iter=1000))])
            model.fit(X[train], y[train])
            pred[test] = model.predict(X[test])
        # one score per seed, from the test predictions of all 5 folds together
        scores.append(balanced_accuracy_score(y, pred))

    return {"task": kind + ": " + pos + " vs " + neg,
            "people_" + "pos": df[df["group"] == pos]["person_id"].nunique(),
            "people_" + "neg": df[df["group"] == neg]["person_id"].nunique(),
            "files": len(df),
            "bal_acc_mean": np.mean(scores),
            "bal_acc_sd": np.std(scores)}


if __name__ == "__main__":
    check = pd.read_csv("results/recording_check.csv")
    table = pd.DataFrame([run_task(check, *t) for t in TASKS])
    print(table.round(3).to_string(index=False))

    os.makedirs("results", exist_ok=True)
    table.to_csv("results/table_recording_baseline.csv", index=False)
    print("\nSaved to results/table_recording_baseline.csv")
