import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Data from the document
results = [
    {"label": "Systolic BP (2 mg)", "mean": -9.8, "lower": -12.6, "upper": -7.0},
    {"label": "Systolic BP (1 mg)", "mean": -8.7, "lower": -11.5, "upper": -5.8},
    {"label": "Systolic BP - Resistant Subgroup (2 mg)", "mean": -9.8, "lower": -13.1, "upper": -6.4},
    {"label": "Systolic BP - Resistant Subgroup (1 mg)", "mean": -9.1, "lower": -12.6, "upper": -5.7},
    {"label": "Diastolic BP (2 mg)", "mean": -3.9, "lower": -5.7, "upper": -2.0},
    {"label": "Diastolic BP (1 mg)", "mean": -3.3, "lower": -5.2, "upper": -1.4},
]

# Preparation
labels = [r["label"] for r in results]
means = [r["mean"] for r in results]
lowers = [r["lower"] for r in results]
uppers = [r["upper"] for r in results]
err_low = [m - l for m, l in zip(means, lowers)]
err_high = [u - m for m, u in zip(means, uppers)]

# Large font setting
FONT_SIZE = 36
# Attempt to use Inter, fallback to DejaVu Sans or sans-serif
font_name = "Inter" if "Inter" in [f.name for f in fm.fontManager.ttflist] else "sans-serif"

plt.rcParams.update({
    'font.size': FONT_SIZE,
    'font.family': font_name,
    'axes.titlesize': FONT_SIZE + 4,
    'axes.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE
})

# Scaling the figure size to fit large fonts
fig, ax = plt.subplots(figsize=(26, 12))

# Reference line
ax.axvline(0, color='pink', linestyle='--', linewidth=3, alpha=0.7)

# Plotting error bars
ax.errorbar(means, range(len(labels)), xerr=[err_low, err_high], fmt='o',
            color='purple', capsize=12, capthick=4, markersize=20, elinewidth=4)

# Adding data labels for Mean and CI
for i, r in enumerate(results):
    text = f"{r['mean']:.1f} ({r['lower']:.1f} to {r['upper']:.1f})"
    # Place text to the right of the confidence interval
    ax.text(r['upper'] + 0.5, i, text, va='center', ha='left', fontsize=FONT_SIZE, fontweight='bold')

# Formatting
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
ax.set_xlabel("Least-Square Mean Placebo-Corrected Difference (mm Hg)")
ax.set_title("BaxHTN: Blood Pressure Reductions at Week 12")

# Set x-axis limit to make room for text labels
ax.set_xlim(-16, 8)

ax.invert_yaxis()
ax.grid(axis='x', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig("playground/plots/BaxHTN.png", dpi=300, bbox_inches='tight', transparent=True)