# Sanity test: with random labels (shuffled between people), the model should score
# about 0.5. If it scores much higher, something is leaking.
# Run from the project folder: python tests/test_sanity.py
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from harness import get_task, make_pipe, get_cv, check_no_leak

SEEDS = range(10)


def shuffle_labels_by_person(y, groups, seed):
    # each person keeps ONE label, but labels are swapped randomly between people
    people = np.unique(groups)
    person_label = {p: y[groups == p][0] for p in people}
    rng = np.random.default_rng(seed)
    new_labels = rng.permutation([person_label[p] for p in people])
    new_map = dict(zip(people, new_labels))
    return np.array([new_map[g] for g in groups])


def cv_score(X, y, groups, seed):
    # one balanced accuracy from the test predictions of all 5 folds
    cv = get_cv(seed, honest=True)
    check_no_leak(cv, X, y, groups)
    pred = np.zeros(len(y), dtype=int)
    for train, test in cv.split(X, y, groups):
        pipe = make_pipe(LogisticRegression(class_weight="balanced", max_iter=1000))
        pipe.fit(X.iloc[train], y[train])
        pred[test] = pipe.predict(X.iloc[test])
    return balanced_accuracy_score(y, pred)


def run():
    X, y, groups = get_task("D1")
    real = [cv_score(X, y, groups, s) for s in SEEDS]
    shuffled = [cv_score(X, shuffle_labels_by_person(y, groups, s), groups, s) for s in SEEDS]
    return real, shuffled


def test_shuffled_labels_score_near_chance():
    _, shuffled = run()
    assert abs(np.mean(shuffled) - 0.5) < 0.1


if __name__ == "__main__":
    real, shuffled = run()
    print("Oxford, Logistic Regression, honest split, 10 seeds")
    print("real labels:     %.3f +- %.3f" % (np.mean(real), np.std(real)))
    print("shuffled labels: %.3f +- %.3f" % (np.mean(shuffled), np.std(shuffled)))
    print("per seed (shuffled):", np.round(shuffled, 3).tolist())
    assert abs(np.mean(shuffled) - 0.5) < 0.1
    print("\nPASSED: shuffled labels score close to 0.5 (chance).")
