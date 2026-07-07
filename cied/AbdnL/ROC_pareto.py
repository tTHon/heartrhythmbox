"""
Pareto-frontier ROC curve for a 2D threshold grid search
(probability threshold x pixel-area threshold) for abandoned lead detection.

Input: CSV with columns
    prob_threshold, pixel_threshold, TP, FP, FN, TN,
    sensitivity, PPV, specificity, F1

Output:
    cied/AbdnL/pareto_dedup.csv          - unique non-dominated operating points
    cied/AbdnL/roc_pareto_frontier.png   - ROC curve plot with selected threshold marked
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_CSV = "cied/AbdnL/oof_detection_thresholds.csv"
OUTPUT_CSV = "cied/AbdnL/pareto_dedup.csv"
OUTPUT_PNG = "cied/AbdnL/roc_pareto_frontier.png"

# Threshold combination(s) to highlight on the plot.
# Multiple entries are supported for tied optima.
SELECTED_THRESHOLDS = [
    {"prob_threshold": 0.80, "pixel_threshold": 600, "label": "prob>0.8, pixel>600px"},
    {"prob_threshold": 0.85, "pixel_threshold": 300, "label": "prob>0.85, pixel>300px (tie)"},
]


def compute_pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify non-dominated (Pareto-optimal) operating points.

    A point (sensitivity_i, specificity_i) is DOMINATED if there exists
    another point j such that:
        sensitivity_j >= sensitivity_i  AND  specificity_j >= specificity_i
        AND at least one of the two is strictly greater.

    Dominated points are discarded because some other threshold combination
    achieves equal-or-better performance on both axes simultaneously —
    it is never a rational choice.
    """
    df = df.dropna(subset=["sensitivity", "specificity"]).reset_index(drop=True)
    sens = df["sensitivity"].to_numpy()
    spec = df["specificity"].to_numpy()
    n = len(df)

    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        dominates_i = (sens >= sens[i]) & (spec >= spec[i]) & (
            (sens > sens[i]) | (spec > spec[i])
        )
        if dominates_i.any():
            is_dominated[i] = True

    return df.loc[~is_dominated].reset_index(drop=True)


def dedupe_frontier(pareto: pd.DataFrame) -> pd.DataFrame:
    """
    Many different (prob, pixel) combinations can land on the exact same
    (sensitivity, specificity) point (ties). Keep one representative row
    per unique point (highest F1 as tie-break) so the curve doesn't plot
    redundant overlapping markers.
    """
    pareto = pareto.copy()
    pareto["fpr"] = 1 - pareto["specificity"]
    dedup = (
        pareto.sort_values("F1", ascending=False)
        .drop_duplicates(subset=["sensitivity", "specificity"], keep="first")
        .sort_values("sensitivity")
        .reset_index(drop=True)
    )
    return dedup


def compute_auc(dedup: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """
    Trapezoidal-rule AUC over the deduplicated Pareto frontier, anchored
    at (FPR=0, Sens=0) and (FPR=1, Sens=1) so the curve spans the full
    unit square like a conventional ROC curve.
    """
    d = dedup.sort_values("fpr").reset_index(drop=True)
    if d["fpr"].min() > 0:
        d = pd.concat(
            [pd.DataFrame([{"fpr": 0.0, "sensitivity": 0.0}]), d], ignore_index=True
        )
    if d["fpr"].max() < 1.0:
        d = pd.concat(
            [d, pd.DataFrame([{"fpr": 1.0, "sensitivity": 1.0}])], ignore_index=True
        )
    auc = np.trapezoid(d["sensitivity"], d["fpr"])
    return auc, d


def plot_roc(d: pd.DataFrame, auc: float, selected: list[dict], out_path: str):

    plt.rcParams.update({"font.family": "inter"})

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(
        d["fpr"], d["sensitivity"], "-o",
        color="#2166ac", markersize=4, linewidth=1.8,
        label="Pareto-optimal operating points",
    )
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="Chance")

    # Label every Pareto-frontier point with its (prob, pixel) threshold,
    # skipping the synthetic anchor points (0,0) and (1,1) added for AUC
    # integration, which have no corresponding threshold combination.
    for _, row in d.iterrows():
        if pd.isna(row.get("prob_threshold")) or pd.isna(row.get("pixel_threshold")):
            continue
        label = f"p>{row['prob_threshold']:.2f}\npx>{int(row['pixel_threshold'])}"
        ax.annotate(
            label,
            xy=(row["fpr"], row["sensitivity"]),
            xytext=(6, 6), textcoords="offset points",
            fontsize=6.5, color="#333333",
        )

    for i, sel in enumerate(selected):
        if "sensitivity" in sel and "specificity" in sel:
            ax.scatter(
                [1 - sel["specificity"]], [sel["sensitivity"]],
                color="#d6604d", s=100, zorder=5,
                label=f"Selected: {sel['label']}" if i == 0 else None,
            )

    ax.set_xlabel("1 - Specificity (False Positive Rate)")
    ax.set_ylabel("Sensitivity")
    ax.set_title(f"ROC Curve — Pareto Frontier of 2D Threshold Grid\n(AUC = {auc:.3f})")
    ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(-0.03, 1.08)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    df = pd.read_csv(INPUT_CSV)
    pareto = compute_pareto_frontier(df)
    dedup = dedupe_frontier(pareto)
    dedup.to_csv(OUTPUT_CSV, index=False)

    auc, d_full = compute_auc(dedup)
    print(f"Unique Pareto-optimal points: {len(dedup)}")
    print(f"AUC (Pareto frontier, trapezoidal): {auc:.4f}")

    # Fill in exact sensitivity/specificity for the selected thresholds by
    # matching back to the original (non-deduped) sweep results.
    for sel in SELECTED_THRESHOLDS:
        match = df[
            (df["prob_threshold"] == sel["prob_threshold"])
            & (df["pixel_threshold"] == sel["pixel_threshold"])
        ]
        if not match.empty:
            sel["sensitivity"] = float(match["sensitivity"].iloc[0])
            sel["specificity"] = float(match["specificity"].iloc[0])

    plot_roc(d_full, auc, SELECTED_THRESHOLDS, OUTPUT_PNG)
    print(f"Saved: {OUTPUT_CSV}, {OUTPUT_PNG}")


if __name__ == "__main__":
    main()