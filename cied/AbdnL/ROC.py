"""
Generate an ROC curve for the abandoned-lead detection model,
sweeping the pixel-area threshold while holding the softmax
probability threshold fixed at 0.5.

Input : detection_thresholds.csv
    columns: prob_threshold, pixel_threshold, TP, FP, FN, TN,
             sensitivity, PPV, specificity, F1
Output: roc_pixel_threshold_prob50.png (+ .pdf)
        Also prints the AUC and a small table of operating points.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

# ------------------------------------------------------------------
# 1. Load and filter data
# ------------------------------------------------------------------
INPUT_CSV = "cied/AbdnL/detection_thresholds.csv"
OUT_PNG = "cied/AbdnL/roc_pixel_threshold_prob50.png"
OUT_PDF = "cied/AbdnL/roc_pixel_threshold_prob50.pdf"

df = pd.read_csv(INPUT_CSV)

# Fixed probability threshold = 0.5, sweep pixel_threshold
sub = df[np.isclose(df["prob_threshold"], 0.5)].copy()
sub = sub.sort_values("pixel_threshold").reset_index(drop=True)

# Recompute sensitivity / 1-specificity directly from counts for precision
sub["sensitivity_calc"] = sub["TP"] / (sub["TP"] + sub["FN"])
sub["fpr_calc"] = sub["FP"] / (sub["FP"] + sub["TN"])  # 1 - specificity

# ------------------------------------------------------------------
# 2. Build ROC points: anchor at (0,0) and (1,1) for a proper curve
# ------------------------------------------------------------------
fpr = sub["fpr_calc"].values
tpr = sub["sensitivity_calc"].values

# Sort by FPR ascending so the curve is monotonic for AUC / plotting
order = np.argsort(fpr)
fpr_sorted = fpr[order]
tpr_sorted = tpr[order]

# Anchor endpoints if not already present
if fpr_sorted[0] > 0:
    fpr_sorted = np.insert(fpr_sorted, 0, 0.0)
    tpr_sorted = np.insert(tpr_sorted, 0, 0.0)
if fpr_sorted[-1] < 1:
    fpr_sorted = np.append(fpr_sorted, 1.0)
    tpr_sorted = np.append(tpr_sorted, 1.0)

roc_auc = auc(fpr_sorted, tpr_sorted)

# ------------------------------------------------------------------
# 3. Identify the current operating point: pixel_threshold = 2500
# ------------------------------------------------------------------
op_row = sub[sub["pixel_threshold"] == 2500]

# ------------------------------------------------------------------
# 4. Plot
# ------------------------------------------------------------------
plt.figure(figsize=(6, 6))
plt.plot(fpr_sorted, tpr_sorted, color="#1f4e79", lw=2,
         marker="o", markersize=4,
         label=f"Pixel-threshold sweep (prob > 0.5)\nAUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1,
         label="Chance (AUC = 0.500)")

if not op_row.empty:
    plt.scatter(op_row["fpr_calc"], op_row["sensitivity_calc"],
                color="crimson", s=90, zorder=5,
                label="Operating point (pixel > 2,500)")

plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
plt.title("ROC Curve: Case-Level Abandoned Lead Detection\n(Pixel-Area Threshold Sweep at Softmax Probability > 0.5)")
plt.legend(loc="lower right", fontsize=9)
plt.xlim(-0.02, 1.02)
plt.ylim(-0.02, 1.02)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#plt.savefig(OUT_PNG, dpi=300)
#plt.savefig(OUT_PDF)
plt.close()

# ------------------------------------------------------------------
# 5. Print summary table
# ------------------------------------------------------------------
print(f"AUC (pixel-threshold sweep, prob>0.5): {roc_auc:.4f}\n")
print(sub[["pixel_threshold", "TP", "FP", "FN", "TN",
           "sensitivity_calc", "fpr_calc"]].to_string(index=False))