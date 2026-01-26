import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# ---------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------

# 1. REBOOT Trial (Stable, linear risk)
# Added 0.25 (3mo) and 0.5 (6mo) points based on linear incidence (~1.2% per year)
x_reboot = np.array([0, 0.25, 0.5, 1, 2, 3, 4, 5])
y_reboot = np.array([0, 0.3,  0.6, 1.2, 2.4, 3.6, 4.8, 5.6])

# 2. Norwegian Registry (LVEF >= 50% No Symptoms)
# Steep early rise
x_norway_best = np.array([0, 0.25, 0.5, 1, 2, 3, 4, 5])
y_norway_best = np.array([0, 2.8,  4.2, 6.5, 10.2, 13.0, 15.8, 18.0])

# 3. Norwegian Registry (All Patients)
# Very steep early rise
x_norway_all = np.array([0, 0.25, 0.5, 1, 2, 3, 4, 5])
y_norway_all = np.array([0, 5.0,  7.5, 11.0, 16.5, 20.5, 23.8, 26.6])


# ---------------------------------------------------------
# SMOOTHING
# ---------------------------------------------------------
def get_smooth_curve(x, y, num_points=300):
    spl = make_interp_spline(x, y, k=2) 
    x_smooth = np.linspace(x.min(), x.max(), num_points)
    y_smooth = spl(x_smooth)
    y_smooth[y_smooth < 0] = 0
    return x_smooth, y_smooth

x_reboot_smooth, y_reboot_smooth = get_smooth_curve(x_reboot, y_reboot)
x_norway_best_smooth, y_norway_best_smooth = get_smooth_curve(x_norway_best, y_norway_best)
x_norway_all_smooth, y_norway_all_smooth = get_smooth_curve(x_norway_all, y_norway_all)

# ---------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------
color_reboot      = '#0E8A73'  # Deep Teal
color_norway_best = '#B36B00'  # Dark Gold
color_norway_all  = '#C0392B'  # Deep Terra Cotta
color_text        = '#2C3E50'

fig, ax = plt.subplots(figsize=(10, 7))

# Style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'text.color': color_text,
    'axes.labelcolor': color_text,
    'xtick.color': color_text,
    'ytick.color': color_text
})

# Axes
ax.spines['left'].set_position(('outward', 10))
ax.spines['bottom'].set_position(('outward', 10))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_color(color_text)
ax.spines['bottom'].set_color(color_text)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.tick_params(axis='both', which='major', labelsize=14, length=10, width=2)

# Plot
ax.plot(x_norway_all_smooth, y_norway_all_smooth, color=color_norway_all, linewidth=3.5, linestyle=':', label='Norwegian Registry (All Patients)')
ax.plot(x_norway_best_smooth, y_norway_best_smooth, color=color_norway_best, linewidth=3.5, linestyle='--', label='Norwegian Registry (LVEF ≥50%)')
ax.plot(x_reboot_smooth, y_reboot_smooth, color=color_reboot, linewidth=3.5, label='REBOOT (Combined)')

# Labels
ax.set_title('Mortality: REBOOT Trial vs Norwegian Registry', fontsize=18, pad=20, fontweight='bold', color=color_text)
ax.set_xlabel('Time (Years)', fontsize=16, color=color_text, labelpad=20)
ax.set_ylabel('Cumulative Incidence of Death (%)', fontsize=16, color=color_text, labelpad=20)
ax.set_xlim(0, 5.0)
ax.set_ylim(0, 30)

# Annotations
ax.annotate(f"{y_norway_all[-1]:.1f}%", xy=(5.0, y_norway_all[-1]), xytext=(5.1, y_norway_all[-1]),
             fontsize=12, color=color_norway_all, fontweight='bold', va='center')
ax.annotate(f"{y_norway_best[-1]:.1f}%", xy=(5.0, y_norway_best[-1]), xytext=(5.1, y_norway_best[-1]),
             fontsize=12, color=color_norway_best, fontweight='bold', va='center')
ax.annotate(f"{y_reboot[-1]:.1f}%", xy=(5.0, y_reboot[-1]), xytext=(5.1, y_reboot[-1]),
             fontsize=12, color=color_reboot, fontweight='bold', va='center')

# Legend
legend = ax.legend(fontsize=12, loc='upper left', frameon=False)
for text in legend.get_texts():
    text.set_color(color_text)

# Grid
ax.grid(True, alpha=0.15, color=color_text, linestyle='--')

plt.tight_layout()
#plt.savefig('reboot_vs_norway_points_added.png', transparent=True, dpi=300)
plt.show()