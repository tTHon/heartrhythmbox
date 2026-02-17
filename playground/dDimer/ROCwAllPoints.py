import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib as mpl

def bootstrap_auc_ci(y, scores, n_bootstraps=2000, seed=42):
    rng = np.random.default_rng(seed)
    bootstrapped_scores = []
    y = np.array(y)
    scores = np.array(scores)
    for _ in range(n_bootstraps):
        indices = rng.integers(0, len(y), len(y))
        if len(np.unique(y[indices])) < 2:
            continue
        score = roc_auc_score(y[indices], scores[indices])
        bootstrapped_scores.append(score)
    return np.percentile(bootstrapped_scores, [2.5, 97.5])

# Load data
df = pd.read_csv('playground/dDimer/dDimer.csv')
data = df[['dDimer', 'Thrombus']].dropna()
y = data['Thrombus']
scores = data['dDimer']

# Calculate ROC
fpr, tpr, thresholds = roc_curve(y, scores)
auc = roc_auc_score(y, scores)
ci_lower, ci_upper = bootstrap_auc_ci(y, scores)

# Optimal cutoff
youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
opt_fpr, opt_tpr = fpr[optimal_idx], tpr[optimal_idx]
optimal_threshold = thresholds[optimal_idx]

# --- Final "Contest-Ready" Plotting ---
BG_COLOR = "#0B0F17"
MAIN_CURVE = "#3BE8FF"
CUTOFF_COLOR = "#FF9F43"
TEXT_DIM = "#8E9AAF"

plt.rcParams.update({'font.family': 'sans-serif', 'text.color': 'white'})
fig, ax = plt.subplots(figsize=(10, 10), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)

# 1. Background Grid & Reference
ax.grid(True, linestyle=':', lw=0.5, color='#2D3436', alpha=0.4, zorder=0)
ax.plot([0, 1], [0, 1], linestyle='--', lw=1, color=TEXT_DIM, alpha=0.2, label='Chance Level')

# 2. Curve + Glow + Gradient
ax.fill_between(fpr, tpr, color=MAIN_CURVE, alpha=0.05)
for i in range(1, 6):
    ax.plot(fpr, tpr, color=MAIN_CURVE, lw=i*2, alpha=0.02, zorder=2)
ax.plot(fpr, tpr, color=MAIN_CURVE, lw=3, solid_capstyle='round', zorder=3, label='d-Dimer Performance')

# 3. Discrete Threshold Points (Granularity)
ax.scatter(fpr, tpr, s=20, color=MAIN_CURVE, alpha=0.5, edgecolors='none', zorder=4, label='Data Thresholds')

# 4. Optimal Cutoff Point & Effect
ax.scatter(opt_fpr, opt_tpr, s=400, color=CUTOFF_COLOR, alpha=0.2, zorder=5)
ax.scatter(opt_fpr, opt_tpr, s=120, color=CUTOFF_COLOR, edgecolors='white', lw=2, zorder=7, label='Optimal Cutoff (Youden Index)')

# 5. Clear Callout
ax.annotate(
    f"Optimal Threshold\n{optimal_threshold:,.0f} ng/mL\n(Sens: {opt_tpr:.0%}, Spec: {1-opt_fpr:.0%})",
    xy=(opt_fpr, opt_tpr),
    xytext=(opt_fpr + 0.12, opt_tpr - 0.12),
    fontsize=11, fontweight='bold', color='white',
    arrowprops=dict(arrowstyle="->", color='white', connectionstyle="arc3,rad=.2", alpha=0.7),
    bbox=dict(boxstyle='round,pad=0.6', facecolor='#1E2530', edgecolor=CUTOFF_COLOR, alpha=0.9, lw=1.5)
)

# 6. Legends (Self-Explanatory for Audience)
legend = ax.legend(loc='lower right', bbox_to_anchor=(0.65, 0.22), frameon=False, fontsize=10, labelcolor=TEXT_DIM)
for text in legend.get_texts():
    text.set_alpha(0.8)

# 7. Header & Titles
plt.text(0, 1.10, "Diagnostic Performance of d-Dimer", fontsize=26, fontweight='bold', ha='left')
plt.text(0, 1.06, f"Predicting Thrombus Formation | Cohort Size n={len(y)}", fontsize=14, color=TEXT_DIM, ha='left')

ax.set_xlabel("1 − SPECIFICITY (False Positive Rate)", fontsize=11, color=TEXT_DIM, labelpad=15, fontweight='bold')
ax.set_ylabel("SENSITIVITY (True Positive Rate)", fontsize=11, color=TEXT_DIM, labelpad=15, fontweight='bold')

# 8. Stats Summary Card
stats_box = (
    fr"$\mathbf{{Area\ Under\ Curve:\ {auc:.3f}}}$" + "\n"
    fr"$\mathit{{95\%\ Confidence\ Interval:\ [{ci_lower:.3f}–{ci_upper:.3f}]}}$" + "\n\n"
    f"Prevalence (Event Rate): {y.mean():.1%}"
)
ax.text(0.96, 0.04, stats_box, transform=ax.transAxes, fontsize=13, 
        ha='right', va='bottom', family='inter',
        bbox=dict(boxstyle='round,pad=1.2', facecolor='#161B22', edgecolor='#2D3436', alpha=0.85))

# Final clean up
for s in ['top', 'right']: ax.spines[s].set_visible(False)
ax.spines['left'].set_color('#2D3436')
ax.spines['bottom'].set_color('#2D3436')

plt.tight_layout()
plt.savefig("playground/dDimer/ROCEdited.png", dpi=300, facecolor=BG_COLOR)
print("Plot generated successfully.")