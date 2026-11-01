"""Aggregate per-trial results into tables and figures.

Usage: uv run python scripts/analyze.py [--exp 003_demo_diversity] [--tag eval]
Reads exp/<exp>/results/trials_<nnn>_<model>_seed<k>_<tag>.csv (+ replay)
Writes summary_<tag>_*.csv and fig_<tag>_*.png in the same folder.
"""

import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crane_reach.sim import EXP_ROOT

RES = None
CAT_ORDER = ["base", "inside", "outside"]
CAT_JA = {"base": "基準付近", "inside": "範囲内(±7.5°)", "outside": "範囲外(±15°)", "all": "全体"}
COLORS = {"A": "#8a8f98", "B": "#2a6fdb", "C": "#e07a1f", "replay": "#555555"}


def load(tag):
    frames = []
    for f in sorted(RES.glob("trials_*.csv")):
        m = re.fullmatch(r"trials_\d{3}_(A|B|C|replay\w+?)(?:_(seed\d+))?_(eval\w*)", f.stem)
        if not m or m.group(3) != tag:
            continue
        df = pd.read_csv(f)
        df["model"] = m.group(1) if not m.group(1).startswith("replay") else "replay"
        df["seed"] = m.group(2) or "-"
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summarize(df: pd.DataFrame, keys) -> pd.DataFrame:
    g = df.groupby(keys, sort=False, observed=True)
    out = pd.DataFrame({
        "trials": g.size(),
        "success": g.success.sum(),
        "collision": g.collision.sum(),
        "timeout": g.outcome.apply(lambda s: (s == "timeout").sum()),
        "table_contact": g.table_contact.sum(),
        "self_penetration": g.self_penetration.sum(),
        "final_err_cm_mean": g.final_err_cm.mean().round(2),
        "final_err_cm_success": df.final_err_cm.where(df.success).groupby([df[k] for k in keys], sort=False, observed=True).mean().round(2).values,
        "final_err_cm_fail": df.final_err_cm.where(~df.success).groupby([df[k] for k in keys], sort=False, observed=True).mean().round(2).values,
        "time_to_success_s": df.time_to_success.where(df.success).groupby([df[k] for k in keys], sort=False, observed=True).mean().round(2).values,
    }).reset_index()
    out["success_rate"] = (out.success / out.trials).round(3)
    out["success_str"] = out.success.astype(str) + "/" + out.trials.astype(str)
    out["collision_str"] = out.collision.astype(str) + "/" + out.trials.astype(str)
    return out


def main():
    global RES
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="003_demo_diversity")
    ap.add_argument("--tag", default="eval")
    args = ap.parse_args()
    RES = EXP_ROOT / "exp" / args.exp / "results"
    tag = args.tag
    df = load(tag)
    cats = [c for c in CAT_ORDER if c in set(df.category)] or sorted(df.category.unique())
    if set(df.category) - set(CAT_ORDER):
        cats = sorted(df.category.unique())
    df["category"] = pd.Categorical(df.category, cats, ordered=True)
    models = [m for m in ["A", "B", "C", "replay"] if m in set(df.model)]

    by_seed_cat = summarize(df, ["model", "seed", "category"])
    by_seed_all = summarize(df, ["model", "seed"])
    by_cat = summarize(df, ["model", "category"])
    by_all = summarize(df, ["model"])
    by_group = summarize(df, ["model", "group_deg"])
    for name, t in [("by_seed_category", by_seed_cat), ("by_seed", by_seed_all), ("by_category", by_cat),
                    ("overall", by_all), ("by_group", by_group)]:
        t.to_csv(RES / f"summary_{tag}_{name}.csv", index=False)
    # outcome-wise final error (supplementary)
    summarize(df, ["model", "outcome"]).to_csv(RES / f"summary_{tag}_by_outcome.csv", index=False)
    # known (training start posture) vs unused start postures
    df["known"] = np.where(df.model == "C", df.known_C, df.known_B)
    summarize(df, ["model", "known"]).to_csv(RES / f"summary_{tag}_known_vs_unused.csv", index=False)

    print(by_cat[["model", "category", "success_str", "success_rate", "collision_str", "timeout",
                  "final_err_cm_mean", "self_penetration", "table_contact"]].to_string(index=False))
    print(by_all[["model", "success_str", "collision_str", "timeout", "final_err_cm_mean"]].to_string(index=False))
    print(by_seed_all[["model", "seed", "success_str", "collision_str", "timeout"]].to_string(index=False))

    # figure 1: success rate per start-angle group (mean over seeds, dots = individual seeds)
    groups = sorted(df.group_deg.unique())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        sub = df[df.model == m]
        per_seed = sub.groupby(["seed", "group_deg"]).success.mean().unstack()
        mean = per_seed.mean(0).reindex(groups)
        x = np.arange(len(groups)) + (i - (len(models) - 1) / 2) * width
        ax[0].bar(x, mean.values, width, color=COLORS[m], label=f"{m} (n_seeds={len(per_seed)})")
        for s in per_seed.index:
            ax[0].scatter(x, per_seed.loc[s].reindex(groups).values, s=10, color="k", zorder=3)
    ax[0].set_xticks(np.arange(len(groups)), [f"{g:+.1f}°" for g in groups])
    ax[0].set(xlabel="start joint1 offset group", ylabel="success rate", ylim=(0, 1.05),
              title="Success rate by start-angle group (5 states / group / seed)")
    inside = [i for i, g in enumerate(groups) if -10 <= g <= 10]
    if inside:
        ax[0].axvspan(min(inside) - 0.5, max(inside) + 0.5, color="#f3e9d8", zorder=0)
        ax[0].text(np.mean(inside), 1.08, "C's demo range (-10..+10°)", ha="center", fontsize=8)
    ax[0].set_ylim(0, 1.15)
    ax[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4)
    for m in models:
        sub = df[df.model == m]
        ax[1].scatter(sub.d_joint1_deg + (0.25 if m == "C" else -0.25 if m == "B" else 0), sub.final_err_cm,
                      s=14, alpha=0.7, color=COLORS[m], label=m,
                      marker="o")
        bad = sub[sub.collision]
        ax[1].scatter(bad.d_joint1_deg + (0.25 if m == "C" else -0.25 if m == "B" else 0), bad.final_err_cm,
                      s=40, facecolors="none", edgecolors="red", lw=1)
    ax[1].axhline(2.0, ls="--", c="gray", lw=0.8)
    ax[1].set(xlabel="start joint1 offset [deg]", ylabel="final tip-target distance [cm]", yscale="log",
              title="Final error per trial (red ring = obstacle contact)")
    ax[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4)
    fig.tight_layout()
    fig.savefig(RES / f"fig_{tag}_success_and_error.png", dpi=150)
    plt.close(fig)

    # figure 2: outcome counts per model x category
    fig, ax = plt.subplots(figsize=(8, 3.5))
    labels, succ, coll, tout = [], [], [], []
    for m in models:
        for c in cats:
            s = df[(df.model == m) & (df.category == c)]
            if len(s) == 0:
                continue
            labels.append(f"{m}\n{c}")
            succ.append(s.success.mean())
            coll.append(s.collision.mean())
            tout.append((s.outcome == "timeout").mean())
    x = np.arange(len(labels))
    ax.bar(x, succ, color="#3cbe5a", label="success")
    ax.bar(x, coll, bottom=succ, color="#e6463c", label="collision")
    ax.bar(x, tout, bottom=np.array(succ) + np.array(coll), color="#e6a028", label="timeout")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set(ylabel="fraction of trials", title="Outcome by model and evaluation category (all seeds)")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout()
    fig.savefig(RES / f"fig_{tag}_outcomes.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
