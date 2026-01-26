import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

# ---------------------------------------------------------
# DATA PREPARATION (Corrected for Natural Shape)
# ---------------------------------------------------------

# 1. REBOOT Trial (Linear Risk)
# The original trial is a steady accumulation of events.
# 0 -> 1.2% at 1 year.
x_reboot = np.array([0, 1.0])
y_reboot = np.array([0, 1.2]) 

# 2. Norwegian Registry (LVEF >= 50%)
# "Early drop" means steep mortality rise in first 3 months.
# Data: 0 -> 2.5% (3mo) -> 4.0% (1yr)
x_norway_best = np.array([0, 0.25, 1.0])
y_norway_best = np.array([0, 2.5,  4.0])

# 3. Norwegian Registry (All Patients - High Risk)
# Very steep early mortality.
# Data: 0 -> 6.0% (3mo) -> 9.0% (1yr)
x_norway_all = np.array([0, 0.25, 1.0])
y_norway_all = np.array([0, 6.0,  9.0])


# ---------------------------------------------------------
# SMOOTHING (Monotonic)
# ---------------------------------------------------------
def get_natural_curve(x, y, num_points=300):
    # PchipInterpolator guarantees the curve is monotonic (no wiggles)
    pch = PchipInterpolator(x, y)
    x_smooth = np.linspace(x.min(), x.max(), num_points)
    y_smooth = pch(x_smooth)
    return x_smooth, y_smooth

x_reboot_smooth, y_reboot_smooth = get_natural_curve(x_reboot, y_reboot)
x_norway_best_smooth, y_norway_best_smooth = get_natural_curve(x_norway_best, y_norway_best)
x_norway_all_smooth, y_norway_all_smooth = get_natural_curve(x_norway_all, y_norway_all)

# ---------------------------------------------------------
# PLOTTING (Dark Stitch Theme)
# ---------------------------------------------------------
# Colors sampled from the provided Stitch image
bg_color = '#231F34'       # Deep Navy/Purple Background
grid_color = '#3E3B50'     # Subtle Grid
text_color = '#E0E0E0'     # Off-white for text

# REBOOT: Brighter Cyan/Light Blue (was #4FC3F7)
col_reboot = '#00E5FF'

# Norway Best: Brighter Yellow/Gold (was #F9A825)
col_norway_b = '#FFD700'

# Norway All: Brighter, more vivid Red (was #FF5252)
col_norway_a = '#FF3333'

fig, ax = plt.subplots(figsize=(15, 10))
fig.patch.set_facecolor(bg_color)
ax.set_facecolor(bg_color)

# Plotting
ax.plot(x_norway_all_smooth, y_norway_all_smooth, color=col_norway_a, linewidth=4.5, alpha=1, linestyle='dotted', label='Norwegian Registry (All Patients)')
ax.plot(x_norway_best_smooth, y_norway_best_smooth, color=col_norway_b, linewidth=4.5, alpha=1, linestyle='--', label='Norwegian Registry (LVEF ≥50% w/o HF)')
ax.plot(x_reboot_smooth, y_reboot_smooth, color=col_reboot, linewidth=4.5, linestyle='-.', label='REBOOT (Both Arms)')

# Styling
ax.set_title('Mortality: REBOOT Trial vs Norwegian Registry (1 Year)', fontsize=20, color=text_color, fontweight='bold', pad=20)
ax.set_xlabel('Time (Years)', fontsize=28, color=text_color, labelpad=10)
ax.set_ylabel('Cumulative Mortality (%)', fontsize=28, color=text_color, labelpad=10)

# Axes & Grid
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color(text_color)
ax.spines['left'].set_color(text_color)

# Modern Detached Axes
ax.spines['left'].set_position(('outward', 20))
ax.spines['bottom'].set_position(('outward', 20))
ax.spines['left'].set_linewidth(2.5)
ax.spines['bottom'].set_linewidth(2.5)



ax.tick_params(axis='both', which='major', colors=text_color, labelsize=24, length=10, width=2)
ax.grid(color=grid_color, linestyle=':', linewidth=1, alpha=0.5)

# Limits
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 12)  # Tight fit for 1 year data

# Annotations (Directly on the graph)
ax.text(1.02, y_norway_all[-1], f"{y_norway_all[-1]}%", color=col_norway_a, fontsize=36, fontweight='bold', va='center')
ax.text(1.02, y_norway_best[-1], f"{y_norway_best[-1]}%", color=col_norway_b, fontsize=36, fontweight='bold', va='center')
ax.text(1.02, y_reboot[-1], f"{y_reboot[-1]}%", color=col_reboot, fontsize=36, fontweight='bold', va='center')

# Legend
legend = ax.legend(fontsize=28, loc='upper left', frameon=False)
for text in legend.get_texts():
    text.set_color(text_color)

plt.tight_layout()
plt.savefig('playground/Plots/reboot_vs_norway.png', dpi=300, bbox_inches='tight', transparent=True)
#plt.show()