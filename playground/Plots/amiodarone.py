import matplotlib.pyplot as plt
import numpy as np

# Data grouped and sorted by baseline value (Before Amiodarone)
erps = ['ERP RA', 'ERP RV', 'ERP AVN', 'ERP Ant AP']  # Reordered for visual clarity
before_means = [232, 229, 287, 265]  # Reordered to match new ERP order
before_stds = [41, 7, 31, 14]         # Reordered to match new ERP order
iv_means = [256, 231, 326, 305]      # Reordered to match new ERP order
iv_stds = [42, 10, 35, 21]           # Reordered to match new ERP order
oral_means = [268, 247, 323, 354]    # Reordered to match new ERP order
oral_stds = [50, 16, 37, 32]         # Reordered to match new ERP order


x = np.arange(len(erps))
width = 0.25

TEXT_COLOR = '#FbF4F4'    # Deep Navy (Matches Stitch's outlines/eyes)
BAR_COLOR2 = '#9454b4'     
BAR_COLOR3 = '#5088b8'     
BAR_COLOR1 = '#FFD700'   # Raincoat Yellow (Matches Stitch's raincoat)
error_bar_color = '#777777'  # Dark Gray for error bars (Neutral contrast)

plt.rcParams.update({
    'font.size': 26,
    'font.family': 'inter',
    'font.weight': 'regular',
    'text.color': TEXT_COLOR,
    'axes.labelcolor': TEXT_COLOR,
    'xtick.color': TEXT_COLOR,
    'ytick.color': TEXT_COLOR,
    'axes.edgecolor': TEXT_COLOR,
    'figure.facecolor': 'none', # Transparent Figure
    'axes.facecolor': 'none',   # Transparent Axis
    'savefig.facecolor': 'none' # Transparent Export
})

# Create plot canvas
fig, ax = plt.subplots(figsize=(16, 9))

# Construct grouped bar series with customized color palette and error boundaries
rects1 = ax.bar(x - width, before_means, width, yerr=before_stds, ecolor=error_bar_color, linewidth=28,
                label='Before Amiodarone', color=BAR_COLOR1, capsize=12)
rects2 = ax.bar(x, iv_means, width, yerr=iv_stds, ecolor=error_bar_color, linewidth=26,
                label='IV Amiodarone', color=BAR_COLOR2, capsize=12)
rects3 = ax.bar(x + width, oral_means, width, yerr=oral_stds, ecolor=error_bar_color, linewidth=28,
                label='Oral Amiodarone', color=BAR_COLOR3, capsize=12)

# Format titles and axis styling
ax.set_ylabel('Effective Refractory Period (ms)', fontsize=28, fontweight='bold')
#ax.set_title('Comparison of Effective Refractory Periods (ERPs) by Treatment Phase', fontsize=20, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(erps, fontsize=26, fontweight='bold')
ax.set_ylim(0, 450)
ax.legend(fontsize=26, loc='upper left')
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Helper function to auto-annotate columns with mean values
def autolabel(rects, fontsize=26):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=fontsize, fontweight='bold', color=TEXT_COLOR)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
#plt.show()
plt.savefig('playground/Plots/erp_comparison.png', dpi=300, transparent=True)