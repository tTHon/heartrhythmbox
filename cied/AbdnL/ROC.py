"""
Generate ROC curves for the abandoned-lead detection model across
ALL probability thresholds, sweeping pixel-area threshold within each.

Each probability threshold (0.3, 0.35, ..., 0.95) gets its own ROC curve
(built from its pixel-threshold sweep). All curves are plotted together
so the best-performing probability/pixel combination can be identified
by whichever curve bulges furthest toward the top-left corner.

The single best operating point across the ENTIRE grid (by Youden's J)
is also identified and marked.

Input : detection_thresholds.csv
Output: roc_all_thresholds.png / .pdf
        printed summary table (AUC per prob threshold + global best point)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import auc

INPUT_CSV = "cied/AbdnL/detection_thresholds.csv"
#OUT_PNG = "cied/AbdnL/roc_all_thresholds.png"
#OUT_PDF = "cied/AbdnL/roc_all_thresholds.pdf"

df = pd.read_csv(INPUT_CSV)

# Recompute sensitivity / FPR directly from raw counts for precision
df["sensitivity_calc"] = df["TP"] / (df["TP"] + df["FN"])
df["fpr_calc"] = df["FP"] / (df["FP"] + df["TN"])
df["specificity_calc"] = df["TN"] / (df["TN"] + df["FP"])
df["youden_j"] = df["sensitivity_calc"] + df["specificity_calc"] - 1

prob_levels = sorted(df["prob_threshold"].unique())
cmap = cm.get_cmap("viridis", len(prob_levels))

plt.figure(figsize=(7.5, 7.5))

auc_summary = []

for i, prob in enumerate(prob_levels):
    sub = df[df["prob_threshold"] == prob].copy()
    sub = sub.sort_values("pixel_threshold")

    fpr = sub["fpr_calc"].values
    tpr = sub["sensitivity_calc"].values

    order = np.argsort(fpr)
    fpr_sorted = fpr[order]
    tpr_sorted = tpr[order]

    # anchor endpoints
    if fpr_sorted[0] > 0:
        fpr_sorted = np.insert(fpr_sorted, 0, 0.0)
        tpr_sorted = np.insert(tpr_sorted, 0, 0.0)
    if fpr_sorted[-1] < 1:
        fpr_sorted = np.append(fpr_sorted, 1.0)
        tpr_sorted = np.append(tpr_sorted, 1.0)

    roc_auc = auc(fpr_sorted, tpr_sorted)
    auc_summary.append((prob, roc_auc, len(sub)))

    plt.plot(fpr_sorted, tpr_sorted, color=cmap(i), lw=1.6, alpha=0.85,
              label=f"prob>{prob:.2f} (AUC={roc_auc:.3f})")

# Chance line
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1, label="Chance")

# ------------------------------------------------------------------
# Global best operating point across the ENTIRE grid (max Youden's J)
# ------------------------------------------------------------------
best_row = df.loc[df["youden_j"].idxmax()]
plt.scatter(best_row["fpr_calc"], best_row["sensitivity_calc"],
            color="red", s=140, zorder=6, marker="*",
            label=(f"Global best (prob>{best_row['prob_threshold']:.2f}, "
                    f"pixel>{int(best_row['pixel_threshold'])})\n"
                    f"J={best_row['youden_j']:.3f}"))

# Current chosen operating point: prob>0.5, pixel>2700
chosen = df[(df["prob_threshold"] == 0.5) & (df["pixel_threshold"] == 2750)]
if not chosen.empty:
    plt.scatter(chosen["fpr_calc"], chosen["sensitivity_calc"],
                color="crimson", s=110, zorder=6, marker="D",
                label="Chosen operating point (prob>0.5, pixel>2,750)")

plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
plt.title("ROC Curves Across All Probability Thresholds\n(each curve = pixel-area sweep at fixed softmax probability threshold)")
plt.xlim(-0.02, 1.02)
plt.ylim(-0.02, 1.02)
plt.legend(loc="lower right", fontsize=6.5, ncol=1, framealpha=0.9)
plt.grid(alpha=0.3)
plt.tight_layout()

#plt.savefig(OUT_PNG, dpi=300)
#plt.savefig(OUT_PDF)
#plt.close()
plt.show()

# ------------------------------------------------------------------
# Print summary
# ------------------------------------------------------------------
print("AUC per probability threshold (pixel-area sweep within each):\n")
print(f"{'prob_threshold':>15s} {'AUC':>8s} {'n_points':>10s}")
for prob, a, n in auc_summary:
    print(f"{prob:>15.2f} {a:>8.4f} {n:>10d}")

print("\nGlobal best operating point (max Youden's J) across the full grid:")
print(best_row[["prob_threshold", "pixel_threshold", "TP", "FP", "FN", "TN",
                 "sensitivity_calc", "specificity_calc", "youden_j"]])

if not chosen.empty:
    print("\nCurrently chosen operating point (prob>0.5, pixel>2,750):")
    print(chosen[["TP", "FP", "FN", "TN", "sensitivity_calc",
                   "specificity_calc", "youden_j"]].to_string(index=False))