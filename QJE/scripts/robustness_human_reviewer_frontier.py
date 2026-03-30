"""
Robustness check: Collaboration frontier using human reviewers only.

Re-estimates the collaboration frontier (Equation 6 in the main text) using
the mean of non-AI-flagged reviewer scores as the dependent variable, to verify
that the monotonic quality decline is not an artifact of AI reviewer leniency.

Produces the 15 values for Table F.5:
  Column 1 "All Reviewers":  beta0, Delta_10-25, Delta_25-50, Delta_50-75, Delta_75-100, N
  Column 2 "Human Only":     beta0, beta1, beta2, Delta_1-10, Delta_10-25, Delta_25-50,
                              Delta_50-75, Delta_75-100, N

Supports two input formats:
  (A) Generic:     paper_id, reviewer_id, overall_score, ai_reviewer, ai_content_pct
  (B) Repo-native: submission_number, rating, ai_classification, ai_percentage

Usage:
    python robustness_human_reviewer_frontier.py --input reviews.csv --output results/
    python robustness_human_reviewer_frontier.py --submissions subs.csv --reviews revs.csv --output results/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


# ── Column mapping ──────────────────────────────────────────────────────────

GENERIC_COLS = {"paper_id", "reviewer_id", "overall_score", "ai_reviewer", "ai_content_pct"}
REPO_COLS = {"submission_number", "rating", "ai_classification", "ai_percentage"}

HUMAN_CLASSIFICATIONS = {"Fully human-written", "Lightly AI-edited"}


def _detect_and_normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Map repo-native column names to the generic schema expected below."""
    cols = set(df.columns)
    if GENERIC_COLS.issubset(cols):
        return df  # already in generic format

    if not {"submission_number", "rating"}.issubset(cols):
        raise ValueError(
            "Input CSV must contain either the generic columns "
            f"{sorted(GENERIC_COLS)} or the repo-native columns "
            f"{sorted(REPO_COLS)}."
        )

    out = df.rename(columns={
        "submission_number": "paper_id",
        "rating": "overall_score",
    })

    # ai_reviewer: binary flag (0 = human, 1 = AI)
    if "ai_classification" in cols:
        out["ai_reviewer"] = (~out["ai_classification"].isin(HUMAN_CLASSIFICATIONS)).astype(int)
    elif "ai_reviewer" not in cols:
        raise ValueError("Need either 'ai_classification' or 'ai_reviewer' column.")

    # ai_content_pct: paper-level AI content (0-100)
    if "ai_percentage" in cols and "ai_content_pct" not in cols:
        pct = out["ai_percentage"]
        if pct.dtype == "object":
            pct = pct.str.replace("%", "", regex=False)
        out["ai_content_pct"] = pd.to_numeric(pct, errors="coerce")

    # reviewer_id: create if missing
    if "reviewer_id" not in out.columns:
        out["reviewer_id"] = range(len(out))

    return out


# ── Core analysis ───────────────────────────────────────────────────────────

def compute_paper_means(df: pd.DataFrame) -> pd.DataFrame:
    """Compute paper-level means for all reviewers and human-only reviewers."""
    all_means = (
        df.groupby("paper_id")
        .agg(
            rating_all=("overall_score", "mean"),
            ai_content=("ai_content_pct", "first"),
            n_reviews=("overall_score", "count"),
        )
        .reset_index()
    )

    human_df = df[df["ai_reviewer"] == 0]
    human_means = (
        human_df.groupby("paper_id")
        .agg(rating_human=("overall_score", "mean"), n_human=("overall_score", "count"))
        .reset_index()
    )

    merged = all_means.merge(human_means, on="paper_id", how="inner")
    merged = merged[merged["n_human"] >= 1]
    return merged


def estimate_frontier(y: np.ndarray, ai: np.ndarray) -> dict:
    """Estimate quadratic collaboration frontier: y = b0 + b1*AI + b2*AI^2."""
    X = sm.add_constant(np.column_stack([ai, ai**2]))
    model = sm.OLS(y, X).fit(cov_type="HC1")
    return {
        "b0": round(float(model.params[0]), 4),
        "b1": round(float(model.params[1]), 6),
        "b2": round(float(model.params[2]), 8),
        "b0_se": round(float(model.bse[0]), 4),
        "b1_se": round(float(model.bse[1]), 6),
        "b2_se": round(float(model.bse[2]), 8),
        "b1_pval": float(model.pvalues[1]),
        "b2_pval": float(model.pvalues[2]),
        "nobs": int(model.nobs),
        "r2": round(float(model.rsquared), 4),
    }


def _stars(pval: float) -> str:
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return ""


def compute_bin_differences(df: pd.DataFrame, score_col: str) -> dict:
    """Compute mean rating differences vs. pure human papers (0% AI) by bin."""
    bins = [(0, 0), (1, 10), (10, 25), (25, 50), (50, 75), (75, 100)]
    baseline = df.loc[df["ai_content"] == 0, score_col]
    if len(baseline) == 0:
        baseline = df.loc[df["ai_content"] <= 1, score_col]
    baseline_mean = baseline.mean()

    diffs = {"baseline_mean": round(float(baseline_mean), 4), "baseline_n": int(len(baseline))}
    for lo, hi in bins:
        if lo == 0 and hi == 0:
            continue
        mask = (df["ai_content"] > lo) & (df["ai_content"] <= hi)
        subset = df[mask]
        if len(subset) > 0:
            delta = subset[score_col].mean() - baseline_mean
            diffs[f"delta_{lo}_{hi}"] = round(float(delta), 4)
            diffs[f"n_{lo}_{hi}"] = int(len(subset))
        else:
            diffs[f"delta_{lo}_{hi}"] = None
            diffs[f"n_{lo}_{hi}"] = 0
    return diffs


# ── Pretty-print for Table F.5 ─────────────────────────────────────────────

def print_table_f5(results: dict) -> None:
    """Print the 15 values in a format ready for copy-paste into LaTeX."""
    a = results["all_reviewers"]
    h = results["human_only"]

    print("\n" + "=" * 72)
    print("TABLE F.5 — Collaboration Frontier Robustness (Human Reviewers Only)")
    print("=" * 72)
    header = f"{'':30s} {'All Reviewers':>18s} {'Human Only':>18s}"
    print(header)
    print("-" * 72)

    def _fmt(val, se=None, pval=None):
        if val is None:
            return "—"
        s = f"{val:.4f}"
        if pval is not None:
            s += _stars(pval)
        if se is not None:
            s += f"  ({se:.4f})"
        return s

    print(f"{'β₀ (Intercept)':30s} {_fmt(a['b0'], a['b0_se']):>18s} {_fmt(h['b0'], h['b0_se']):>18s}")
    print(f"{'β₁ (AI Content)':30s} {'':>18s} {_fmt(h['b1'], h['b1_se'], h['b1_pval']):>18s}")
    print(f"{'β₂ (AI Content²)':30s} {'':>18s} {_fmt(h['b2'], h['b2_se'], h['b2_pval']):>18s}")
    print("-" * 72)

    ab = a["bin_diffs"]
    hb = h["bin_diffs"]

    for lo, hi in [(1, 10), (10, 25), (25, 50), (50, 75), (75, 100)]:
        key = f"delta_{lo}_{hi}"
        label = f"Δ {lo}–{hi}%"
        a_val = _fmt(ab.get(key)) if key in ab else "—"
        h_val = _fmt(hb.get(key)) if key in hb else "—"
        # All Reviewers column only has 10-25 through 75-100
        if lo == 1:
            a_val = ""
        print(f"{label:30s} {a_val:>18s} {h_val:>18s}")

    print("-" * 72)
    print(f"{'N':30s} {a['nobs']:>18d} {h['nobs']:>18d}")
    print(f"{'R²':30s} {a['r2']:>18.4f} {h['r2']:>18.4f}")
    print("=" * 72)


# ── Figure ──────────────────────────────────────────────────────────────────

def make_figure(df: pd.DataFrame, outdir: Path) -> None:
    """Side-by-side collaboration frontier: all reviewers vs. human only."""
    bins = [0, 1, 10, 25, 50, 75, 100]
    labels = ["0%", "1-10%", "10-25%", "25-50%", "50-75%", "75-100%"]
    df = df.copy()
    df["bin"] = pd.cut(df["ai_content"], bins=bins, labels=labels, include_lowest=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, col, title in zip(
        axes,
        ["rating_all", "rating_human"],
        ["All Reviewers", "Human Reviewers Only"],
    ):
        grouped = df.groupby("bin", observed=True)[col].agg(["mean", "sem", "count"])
        x = range(len(grouped))
        ax.bar(x, grouped["mean"], yerr=1.96 * grouped["sem"], capsize=4,
               color="steelblue", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("AI Content Bin")
        if col == "rating_all":
            ax.set_ylabel("Mean Rating")

    fig.suptitle("Collaboration Frontier Robustness", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(outdir / "fig_human_frontier_robustness.png", dpi=300, bbox_inches="tight")
    plt.close()


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Human-reviewer-only collaboration frontier (Table F.5)"
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--input", help="Single review-level CSV (generic or repo-native format)")
    grp.add_argument("--submissions", help="Submissions CSV (used with --reviews)")

    parser.add_argument("--reviews", help="Reviews CSV (used with --submissions)")
    parser.add_argument("--output", default="results", help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    if args.input:
        df = pd.read_csv(args.input)
        df = _detect_and_normalise(df)
    else:
        if not args.reviews:
            parser.error("--submissions requires --reviews")
        subs = pd.read_csv(args.submissions)
        revs = pd.read_csv(args.reviews)
        # Merge paper-level ai_percentage into review rows
        revs = revs.merge(
            subs[["submission_number", "ai_percentage"]],
            on="submission_number",
            how="left",
        )
        df = _detect_and_normalise(revs)

    papers = compute_paper_means(df)

    # Estimate both frontiers
    res_all = estimate_frontier(papers["rating_all"].values, papers["ai_content"].values)
    res_human = estimate_frontier(papers["rating_human"].values, papers["ai_content"].values)

    # Bin differences
    bins_all = compute_bin_differences(papers, "rating_all")
    bins_human = compute_bin_differences(papers, "rating_human")

    results = {
        "all_reviewers": {**res_all, "bin_diffs": bins_all},
        "human_only": {**res_human, "bin_diffs": bins_human},
    }

    out_path = outdir / "human_frontier_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print_table_f5(results)
    print(f"\nJSON saved to {out_path}")

    make_figure(papers, outdir)
    print(f"Figure saved to {outdir / 'fig_human_frontier_robustness.png'}")


if __name__ == "__main__":
    main()
