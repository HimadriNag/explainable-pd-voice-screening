# How stable are the feature rankings? Compare rankings across seeds, models,
# datasets and methods, and test against random rankings (chance).
import os
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loading import FEATURES

RBO_P = 0.9
N_BOOT = 2000
N_NULL_PAIRS = 20000
N_NULL_AVG = 10000
METRICS = ["jaccard3", "jaccard5", "kendall", "rbo"]
LEVELS = ["seed", "model", "dataset", "method"]

# textbook: a higher value means more PD-like (+1) or more healthy (-1)
TEXTBOOK = {"jitter_local": 1, "jitter_abs": 1, "jitter_rap": 1, "jitter_ppq5": 1,
            "shimmer_local": 1, "shimmer_db": 1, "shimmer_apq3": 1, "shimmer_apq5": 1,
            "shimmer_apq11": 1, "HNR": -1, "NHR": 1}


# ---------- agreement between two rankings ----------
# a ranking = array of 11 ranks, in FEATURES order (1 = most important)

def top_k(ranks, k):
    return set(np.where(ranks <= k)[0])


def jaccard(r1, r2, k):
    a, b = top_k(r1, k), top_k(r2, k)
    return len(a & b) / len(a | b)


def rbo(r1, r2, p=RBO_P):
    # overlap at each depth d, weighted by p^(d-1); divided by the total weight
    # so that two identical lists give exactly 1
    order1, order2 = np.argsort(r1), np.argsort(r2)
    n = len(r1)
    score, weight = 0.0, 0.0
    for d in range(1, n + 1):
        overlap = len(set(order1[:d]) & set(order2[:d])) / d
        score += p ** (d - 1) * overlap
        weight += p ** (d - 1)
    return score / weight


def agreement(r1, r2):
    return {"jaccard3": jaccard(r1, r2, 3),
            "jaccard5": jaccard(r1, r2, 5),
            "kendall": kendalltau(r1, r2)[0],
            "rbo": rbo(r1, r2)}


# ---------- rankings ----------

def rank_with_random_ties(values, rng):
    tie_break = rng.random(len(values))
    order = np.lexsort((tie_break, -values))
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def per_seed_rankings(r, method):
    # {(task, model, seed): ranks array}
    out = {}
    sub = r[r["method"] == method]
    for key, g in sub.groupby(["task", "model", "seed"]):
        out[key] = g.set_index("feature").loc[FEATURES, "rank"].values
    return out


def seed_averaged_rankings(r, method):
    # average importance over the 10 seeds, then rank again: {(task, model): ranks}
    rng = np.random.default_rng(0)
    out = {}
    sub = r[r["method"] == method]
    mean_imp = sub.groupby(["task", "model", "feature"])["importance"].mean()
    for (task, model), g in mean_imp.groupby(level=[0, 1]):
        values = g.droplevel([0, 1]).loc[FEATURES].values
        out[(task, model)] = rank_with_random_ties(values, rng)
    return out


def make_pairs(r, method):
    # list of (r1, r2) ranking pairs for each level
    seeds = per_seed_rankings(r, method)
    avg = seed_averaged_rankings(r, method)
    tasks = sorted({k[0] for k in avg})
    models = sorted({k[1] for k in avg})
    pairs = {}

    # 1) seed: same task + model, different seeds
    pairs["seed"] = []
    for t in tasks:
        for m in models:
            keys = sorted(k for k in seeds if k[0] == t and k[1] == m)
            pairs["seed"] += [(seeds[a], seeds[b]) for a, b in combinations(keys, 2)]

    # 2) model: same task, different models (seed-averaged)
    pairs["model"] = [(avg[(t, a)], avg[(t, b)]) for t in tasks for a, b in combinations(models, 2)]

    # 3) dataset: same model, different tasks (seed-averaged)
    pairs["dataset"] = [(avg[(a, m)], avg[(b, m)]) for m in models for a, b in combinations(tasks, 2)]
    return pairs


def method_pairs(r):
    # 4) method: SHAP vs permutation, same task + model + seed
    s = per_seed_rankings(r, "shap")
    p = per_seed_rankings(r, "perm")
    return [(s[k], p[k]) for k in sorted(s)]


# ---------- statistics ----------

def null_distribution(rng):
    # agreement values for many pairs of random rankings
    n = len(FEATURES)
    rows = []
    for _ in range(N_NULL_PAIRS):
        r1 = rng.permutation(n) + 1
        r2 = rng.permutation(n) + 1
        rows.append(agreement(r1, r2))
    return pd.DataFrame(rows)


def summarise(values, null_values, rng):
    values = np.asarray(values)
    real_mean = values.mean()
    # 95% bootstrap CI: resample the pairs
    boot = [rng.choice(values, len(values), replace=True).mean() for _ in range(N_BOOT)]
    # chance: average of the same number of random pairs, many times
    null_avgs = np.array([rng.choice(null_values, len(values), replace=True).mean()
                          for _ in range(N_NULL_AVG)])
    return {"n_pairs": len(values), "mean": real_mean,
            "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5),
            "chance_mean": null_values.mean(),
            "p_value": (null_avgs >= real_mean).mean()}


def direction_table(path="results/importances.csv"):
    imp = pd.read_csv(path)
    d4 = imp[(imp["task"] == "D4") & (imp["model"] == "LogReg")]
    t = d4.groupby("feature")["shap_direction"].agg(
        mean_corr="mean", share_towards_PD=lambda s: (s > 0).mean()).reindex(FEATURES)
    t["model_direction"] = np.where(t["mean_corr"] > 0, "towards PD", "towards healthy")
    t["textbook_direction"] = ["towards PD" if TEXTBOOK[f] > 0 else "towards healthy" for f in FEATURES]
    t["matches_textbook"] = t["model_direction"] == t["textbook_direction"]
    return t.reset_index()


def stable_core(r):
    rows = []
    for method in ["shap", "perm"]:
        sub = r[r["method"] == method]
        n_runs = sub.groupby(["task", "model", "seed"]).ngroups
        share = (sub[sub["rank"] <= 5].groupby("feature").size()
                 .reindex(FEATURES, fill_value=0) / n_runs)
        for feat, s in share.items():
            rows.append({"method": method, "feature": feat, "share_top5": s,
                         "runs": n_runs, "stable": s >= 0.8})
    return pd.DataFrame(rows).sort_values(["method", "share_top5"], ascending=[False, False])


def plot_rbo(table, chance, out_base):
    t = table[(table["metric"] == "rbo") & (table["method"].isin(["shap", "shap vs perm"]))]
    t = t.set_index("level").loc[LEVELS]
    x = np.arange(len(LEVELS))
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.errorbar(x, t["mean"], yerr=[t["mean"] - t["ci_low"], t["ci_high"] - t["mean"]],
                fmt="o", color="black", markersize=7, capsize=4, linewidth=1.2,
                label="Mean RBO (95% bootstrap CI)")
    ax.axhline(chance, color="0.4", linestyle="--", linewidth=1, label="Chance (random rankings)")
    for xi, m in zip(x, t["mean"]):
        ax.text(xi + 0.12, m, "%.2f" % m, va="center", fontsize=9)
    labels = ["Seeds", "Models", "Datasets", "SHAP vs\npermutation"]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, len(LEVELS) - 0.3)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Ranking agreement (RBO, p = 0.9)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_base + ".png", dpi=300)
    fig.savefig(out_base + ".pdf")
    plt.close(fig)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    r = pd.read_csv("results/rankings.csv")

    direction_table().to_csv("results/table_direction_D4.csv", index=False)
    print("saved results/table_direction_D4.csv")

    print("making random-ranking null distribution...", flush=True)
    null = null_distribution(rng)

    # collect all pair lists: (level, method label, pairs)
    jobs = []
    for method in ["shap", "perm"]:
        for level, pairs in make_pairs(r, method).items():
            jobs.append((level, method, pairs))
    jobs.append(("method", "shap vs perm", method_pairs(r)))

    rows = []
    for level, method, pairs in jobs:
        print("  level %-8s %-13s (%d pairs)" % (level, method, len(pairs)), flush=True)
        agree = pd.DataFrame([agreement(a, b) for a, b in pairs])
        for metric in METRICS:
            row = {"level": level, "method": method, "metric": metric}
            row.update(summarise(agree[metric].values, null[metric].values, rng))
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv("results/table4_stability.csv", index=False)

    core = stable_core(r)
    core.to_csv("results/table5_stable_core.csv", index=False)

    plot_rbo(table, null["rbo"].mean(), "results/fig_stability")

    pd.set_option("display.width", 200)
    print("\nTABLE 4 (stability)")
    print(table.round(3).to_string(index=False))
    print("\nTABLE 5 (stable core: share of runs in top 5)")
    print(core.round(3).to_string(index=False))
    print("\nsaved results/table4_stability.csv, results/table5_stable_core.csv, results/fig_stability.png/.pdf")
