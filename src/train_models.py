# Train all 5 models on D1, D3, D4 with honest (by person) and leaky (by record) splits.
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from harness import get_task, make_pipe, get_cv, check_no_leak, get_metrics
from models import MODEL_NAMES, make_model, get_score

# the leaky split ignores groups on purpose, so hide that expected warning
warnings.filterwarnings("ignore", message="The groups parameter is ignored")

TASKS = ["D1", "D3", "D4"]
SEEDS = range(10)
METRICS = ["bal_acc", "roc_auc", "sensitivity", "specificity", "f1", "mcc"]


def run_all():
    rows = []
    for task in TASKS:
        X, y, groups = get_task(task)
        for split in ["honest", "leaky"]:
            for seed in SEEDS:
                cv = get_cv(seed, honest=(split == "honest"))
                if split == "honest":
                    check_no_leak(cv, X, y, groups)
                for fold, (train, test) in enumerate(cv.split(X, y, groups)):
                    for name in MODEL_NAMES:
                        pipe = make_pipe(make_model(name, seed))  # a NEW pipeline every time
                        pipe.fit(X.iloc[train], y[train])
                        pred = pipe.predict(X.iloc[test])
                        score = get_score(pipe, X.iloc[test])
                        m = get_metrics(y[test], pred, score)
                        m.update({"task": task, "split": split, "seed": seed,
                                  "fold": fold, "model": name})
                        rows.append(m)
            print("done:", task, split)
    return pd.DataFrame(rows)


def per_seed(scores):
    # average the 5 folds inside each seed first
    return scores.groupby(["task", "split", "model", "seed"])[METRICS].mean().reset_index()


def make_table2(seed_scores):
    honest = seed_scores[seed_scores["split"] == "honest"]
    rows = []
    for (task, model), g in honest.groupby(["task", "model"], sort=False):
        row = {"task": task, "model": model}
        for m in METRICS:
            row[m + "_mean"] = g[m].mean()
            row[m + "_sd"] = g[m].std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)


def make_table3(seed_scores):
    rows = []
    for (task, model), g in seed_scores.groupby(["task", "model"], sort=False):
        h = g[g["split"] == "honest"]["bal_acc"]
        l = g[g["split"] == "leaky"]["bal_acc"]
        rows.append({"task": task, "model": model,
                     "honest_mean": h.mean(), "honest_sd": h.std(ddof=0),
                     "leaky_mean": l.mean(), "leaky_sd": l.std(ddof=0),
                     "gap": l.mean() - h.mean()})
    return pd.DataFrame(rows)


def plot_gap(table3, out_base):
    # one panel per dataset; dark solid bar = honest, light hatched bar = leaky (works in black and white)
    fig, axes = plt.subplots(1, len(TASKS), figsize=(11, 3.8), sharey=True)
    x = np.arange(len(MODEL_NAMES))
    w = 0.38
    for ax, task in zip(axes, TASKS):
        t = table3[table3["task"] == task].set_index("model").loc[MODEL_NAMES]
        ax.bar(x - w / 2 - 0.01, t["honest_mean"], w, yerr=t["honest_sd"], capsize=2,
               color="0.25", edgecolor="black", linewidth=0.6, label="Honest (by person)")
        ax.bar(x + w / 2 + 0.01, t["leaky_mean"], w, yerr=t["leaky_sd"], capsize=2,
               color="0.85", edgecolor="black", linewidth=0.6, hatch="///", label="Leaky (by record)")
        ax.axhline(0.5, color="0.4", linestyle=":", linewidth=1, label="Chance (0.5)")
        ax.set_title(task)
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_NAMES)
        ax.set_ylim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.9", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Balanced accuracy (mean ± SD, 10 seeds)")
    # one legend above the panels, so it never covers a bar
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_base + ".png", dpi=300)
    fig.savefig(out_base + ".pdf")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    scores = run_all()
    scores.to_csv("results/scores.csv", index=False)
    print("saved", len(scores), "fold results to results/scores.csv")

    seed_scores = per_seed(scores)
    table2 = make_table2(seed_scores)
    table3 = make_table3(seed_scores)
    table2.to_csv("results/table2_performance.csv", index=False)
    table3.to_csv("results/table3_protocol_gap.csv", index=False)
    plot_gap(table3, "results/fig_protocol_gap")

    pd.set_option("display.width", 200)
    print("\nTABLE 2 (honest split, mean over 10 seeds)")
    print(table2[["task", "model"] + [m + "_mean" for m in METRICS] + ["bal_acc_sd"]].round(3).to_string(index=False))
    print("\nTABLE 3 (balanced accuracy, honest vs leaky)")
    print(table3.round(3).to_string(index=False))
