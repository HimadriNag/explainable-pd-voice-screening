# Test: the honest split never puts the same person in train and test.
# Run from the project folder: python tests/test_leakage.py
import os
import sys
import warnings
# the leaky split ignores groups on purpose, so hide that expected warning
warnings.filterwarnings("ignore", message="The groups parameter is ignored")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from harness import get_task, get_cv, check_no_leak

TASKS = ["D1", "D3", "D4"]
SEEDS = range(10)


def shared_people(cv, X, y, groups):
    # total number of people found in both train and test, over all folds
    total = 0
    for train, test in cv.split(X, y, groups):
        total += len(set(groups[train]) & set(groups[test]))
    return total


def test_honest_split_has_no_shared_people():
    for task in TASKS:
        X, y, groups = get_task(task)
        for seed in SEEDS:
            cv = get_cv(seed, honest=True)
            assert shared_people(cv, X, y, groups) == 0
            check_no_leak(cv, X, y, groups)  # must not raise


def test_leaky_split_does_share_people():
    # proves the test can fail: the record-wise split leaks people
    for task in TASKS:
        X, y, groups = get_task(task)
        for seed in SEEDS:
            cv = get_cv(seed, honest=False)
            assert shared_people(cv, X, y, groups) > 0
            try:
                check_no_leak(cv, X, y, groups)
                raise AssertionError("check_no_leak did not catch the leak")
            except RuntimeError:
                pass


if __name__ == "__main__":
    for task in TASKS:
        X, y, groups = get_task(task)
        honest = [shared_people(get_cv(s, True), X, y, groups) for s in SEEDS]
        leaky = [shared_people(get_cv(s, False), X, y, groups) for s in SEEDS]
        print("%s (%d people): shared people per seed  honest=%s  leaky=%s"
              % (task, len(set(groups)), honest, leaky))

    test_honest_split_has_no_shared_people()
    test_leaky_split_does_share_people()
    print("\nPASSED: honest split shares 0 people; leaky split shares people and check_no_leak catches it.")
