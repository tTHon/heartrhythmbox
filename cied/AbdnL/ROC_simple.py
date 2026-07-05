"""
Simplified ROC curve for the abandoned-lead detection operating threshold.
Plots a single pixel-area sweep (at the fixed, pre-specified probability
threshold) with the selected operating point and the pre-specified minimum
sensitivity requirement marked.

Input:  oof_detection_thresholds.csv
        (from oof_grid_search.py — out-of-fold, leakage-free grid search)
Output: roc_simple_prob{PROB_THRESHOLD}_oof.png / .pdf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

# ── EDIT THESE ────────────────────────────────────────────────────
INPUT_CSV        = "cied/AbdnL/oof_detection_thresholds.csv"   # path to your grid-search output
OUTPUT_DIR       = "cied/AbdnL"                              # where to save the figure
PROB_THRESHOLD   = 0.80    # fixed probability threshold shown on the curve
OP_PIXEL         = 600     # selected operating point's pixel threshold
MIN_SENSITIVITY  = 0.975   # pre-specified minimum sensitivity requirement
# ──────────────────────────────────────────────────────────────────

res = pd.read_csv(INPUT_CSV)
sub = res[res['prob_threshold'] == PROB_THRESHOLD].sort_values(
    'pixel_threshold', ascending=False  # descending: low pixel -> high sens/low spec first
).copy()

fpr = (1 - sub['specificity']).values
tpr = sub['sensitivity'].values

# NOTE: sorted by pixel_threshold (not by fpr value) so ties in FPR/specificity
# don't get shuffled by an unstable argsort — this keeps the ROC staircase
# monotonic and avoids the "zigzag" artifact.
fpr_sorted, tpr_sorted = fpr, tpr
if fpr_sorted[0] > 0:
    fpr_sorted = np.insert(fpr_sorted, 0, 0.0); tpr_sorted = np.insert(tpr_sorted, 0, 0.0)
if fpr_sorted[-1] < 1:
    fpr_sorted = np.append(fpr_sorted, 1.0); tpr_sorted = np.append(tpr_sorted, 1.0)
roc_auc = auc(fpr_sorted, tpr_sorted)

op = sub[sub['pixel_threshold'] == OP_PIXEL].iloc[0]
op_fpr = 1 - op['specificity']
op_tpr = op['sensitivity']

plt.rcParams['font.family'] = 'inter'

plt.figure(figsize=(6.5, 6.5))
plt.plot(fpr_sorted, tpr_sorted, color="#1f4e79", lw=2.2, marker="o", markersize=4, alpha=0.85,
          label=f"Pixel-area threshold sweep\n(softmax probability > {PROB_THRESHOLD:.2f})\nAUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1, label="Chance")

# Pre-specified minimum sensitivity requirement
plt.axhline(MIN_SENSITIVITY, color="darkorange", linestyle=":", lw=1.8,
            label=f"Pre-specified minimum\nsensitivity = {MIN_SENSITIVITY:.3f}")

plt.scatter([op_fpr], [op_tpr], color="crimson", s=140, zorder=5, marker="D",
             label=(f"Selected operating point\n(pixel > {OP_PIXEL})\n"
                     f"Sensitivity = {op_tpr:.3f}, Specificity = {op['specificity']:.3f}"))

plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
plt.title(f"ROC Curve: Case-Level Abandoned Lead Detection\n(Out-of-Fold Predictions, Softmax Probability Threshold = {PROB_THRESHOLD:.2f})")
plt.xlim(-0.02, 1.02); plt.ylim(-0.02, 1.02)
plt.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
plt.grid(alpha=0.3)
plt.tight_layout()

out_png = f"{OUTPUT_DIR}/roc_simple_prob{int(PROB_THRESHOLD*100):02d}_oof.png"
out_pdf = f"{OUTPUT_DIR}/roc_simple_prob{int(PROB_THRESHOLD*100):02d}_oof.pdf"
plt.savefig(out_png, dpi=300)
plt.savefig(out_pdf)
plt.close()
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
print(f"AUC = {roc_auc:.4f}")
print(f"Operating point: sensitivity={op_tpr:.4f}, specificity={op['specificity']:.4f}")