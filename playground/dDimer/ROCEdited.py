import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# --- BOOTSTRAP FUNCTION ---
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

# --- DATA LOADING ---
try:
    df = pd.read_csv('playground/dDimer/dDimer.csv')
    data = df[['dDimer', 'Thrombus']].dropna()
    y = data['Thrombus']
    scores = data['dDimer']
except FileNotFoundError:
    print("CSV not found. Generating synthetic data...")
    np.random.seed(42)
    y = np.random.randint(0, 2, 200)
    scores = np.random.normal(0, 1, 200) + (y * 1.5)

# --- CALCULATIONS ---
fpr, tpr, thresholds = roc_curve(y, scores)
auc = roc_auc_score(y, scores)
ci_lower, ci_upper = bootstrap_auc_ci(y, scores)

youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
opt_fpr, opt_tpr = fpr[optimal_idx], tpr[optimal_idx]
optimal_threshold = thresholds[optimal_idx]

# --- PLOTTING SETUP ---
BG_COLOR = "none"
MAIN_CURVE = "#3BE8FF"
CUTOFF_COLOR = "#FF9F43"
TEXT_DIM = "#B1BAC9"
TEXT_BRIGHT = "white"

plt.rcParams.update({
    'font.family': 'inter', 
    'text.color': TEXT_BRIGHT,
    'font.size': 18,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'axes.linewidth': 3  # Thicker spines
})

fig, ax = plt.subplots(figsize=(14, 14), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)

# 1. OFFSET AXES (Move them away from data)
ax.spines['left'].set_position(('outward', 2))
ax.spines['bottom'].set_position(('outward', 2))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# 2. BOUNDED SPINES (The Key Fix)
# This forces the axis line to draw ONLY from 0 to 1
ax.spines['left'].set_bounds(0, 1)
ax.spines['bottom'].set_bounds(0, 1)

# Color the spines
ax.spines['left'].set_color(TEXT_DIM)
ax.spines['bottom'].set_color(TEXT_DIM)

# 3. VISIBLE TICKS
ax.tick_params(
    axis='both', which='major', colors=TEXT_DIM,
    width=3, length=10, direction='out'
)

# 4. Background Grid
ax.grid(True, linestyle=':', lw=1, color='#2D3436', alpha=0.4, zorder=0)
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color=TEXT_DIM, alpha=0.3, label='Chance Level')

# 5. Curve + Glow
ax.fill_between(fpr, tpr, color=MAIN_CURVE, alpha=0.05)
for i in range(1, 6):
    ax.plot(fpr, tpr, color=MAIN_CURVE, lw=i*2, alpha=0.05, zorder=2)
ax.plot(fpr, tpr, color=MAIN_CURVE, lw=4, solid_capstyle='round', zorder=3, label='d-Dimer Performance')

# 6. Optimal Cutoff Point
ax.scatter(opt_fpr, opt_tpr, s=600, color=CUTOFF_COLOR, alpha=0.3, zorder=5)
ax.scatter(opt_fpr, opt_tpr, s=200, color=CUTOFF_COLOR, edgecolors='white', lw=3, zorder=7, label='Optimal Cutoff')

# 7. Callout
ax.annotate(
    f"Optimal Threshold\n{optimal_threshold:,.0f} ng/mL\n(Sens: {opt_tpr:.0%}, Spec: {1-opt_fpr:.0%})",
    xy=(opt_fpr, opt_tpr),
    xytext=(opt_fpr + 0.15, opt_tpr - 0.20),
    fontsize=18, fontweight='bold', color='white',
    arrowprops=dict(arrowstyle="->", color='white', connectionstyle="arc3,rad=.2", alpha=0.7, lw=2),
    bbox=dict(boxstyle='round,pad=0.8', facecolor='#1E2530', edgecolor=CUTOFF_COLOR, alpha=0.9, lw=2)
)

# 8. Legends
#legend = ax.legend(loc='lower right', bbox_to_anchor=(0.85, 0.25), frameon=False, fontsize=18, labelcolor=TEXT_DIM)
#for text in legend.get_texts():
    #text.set_alpha(0.8)

# 9. Titles & Headers
ax.text(-0.05, 1.12, "Diagnostic Performance of d-Dimer", 
        fontsize=34, fontweight='bold', ha='left', transform=ax.transAxes, color=TEXT_BRIGHT)
ax.text(-0.05, 1.07, f"Predicting Thrombus Formation | Cohort Size n={len(y)}", 
        fontsize=22, color=TEXT_DIM, ha='left', transform=ax.transAxes)

# 10. Axis Labels
ax.set_xlabel("1 − SPECIFICITY (False Positive Rate)", color=TEXT_DIM, labelpad=20, fontweight='bold')
ax.set_ylabel("SENSITIVITY (True Positive Rate)", color=TEXT_DIM, labelpad=20, fontweight='bold')

# 11. Stats Summary Card
stats_box = (
    fr"$\mathbf{{AUC:\ {auc:.3f}}}$" + "\n"
    f"95% CI: [{ci_lower:.3f}–{ci_upper:.3f}]" + "\n\n"
    f"Prevalence: {y.mean():.1%}"
)
ax.text(0.9, 0.15, stats_box, transform=ax.transAxes, fontsize=20, 
        ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=1.0', facecolor='none', edgecolor='#2D3436', alpha=0.9))

plt.tight_layout()
plt.subplots_adjust(top=0.88, bottom=0.15, left=0.15, right=0.95)

plt.savefig("playground/dDimer/ROCEdited.png", dpi=300, facecolor='none')
print("Bounded axis plot generated successfully.")