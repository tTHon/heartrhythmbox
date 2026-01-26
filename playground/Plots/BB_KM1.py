import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# DATA PREPARATION (Same as before)
# ---------------------------------------------------------

# 1. COMMIT Trial
n_total_commit = 46077
combined_events_commit = np.array([2890, 469, 158, 54])
cum_events_commit = np.cumsum(np.concatenate(([0], combined_events_commit)))
y_commit = (cum_events_commit / n_total_commit) * 100
x_commit = np.array([0, 7, 14, 21, 28]) / 365.25

# 2. REBOOT Trial
n_total_reboot = 8438
pct_year1 = ((50 + 44) / n_total_reboot) * 100
x_reboot = np.array([0, 1])
y_reboot = np.array([0, pct_year1])

# 3. Korea Registry
x_korea = np.array([0, 0.25, 0.5, 0.75, 1.0])
korea_bb = np.array([0, 0.5, 1.5, 2.0, 2.8])
korea_nobb = np.array([0, 1.5, 3.0, 4.0, 5.0])
y_korea = (korea_bb + korea_nobb) / 2


# ---------------------------------------------------------
# PLOTTING (Modern "Detached" Axes)
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(15, 10))

# ---------------------------------------------------------
# COLOR PALETTE (Stitch & Raincoat Theme)
# ---------------------------------------------------------
color_commit = '#154360'   # Stitch's Fur (Deep Navy)
color_korea  = '#B36B00'   # Raincoat (Golden Orange)
color_reboot = '#0E8A73'   # Sky/Teal (Tropical)
#color_korea  = '#F39C12'   # Raincoat (Golden Orange)
#color_reboot = '#1ABC9C'   # Sky/Teal (Tropical)
color_text   = '#2C3E50'   # Slate Grey (Softer than black)

# Update Font & Line Settings
plt.rcParams.update({
    'font.size': 24,
    'font.family': 'sans-serif',
    'text.color': color_text,
    'axes.labelcolor': color_text,
    'xtick.color': color_text,
    'ytick.color': color_text
})

# Modern Detached Axes
ax.spines['left'].set_position(('outward', 20))
ax.spines['bottom'].set_position(('outward', 20))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_color(color_text)
ax.spines['bottom'].set_color(color_text)
ax.spines['left'].set_linewidth(2.5)
ax.spines['bottom'].set_linewidth(2.5)

# Plotting with the New Colors
ax.plot(x_commit, y_commit, color=color_commit, linewidth=4.5, label='COMMIT')
ax.plot(x_korea, y_korea, color=color_korea, linewidth=4.5, label='Korea Registry')
ax.plot(x_reboot, y_reboot, color=color_reboot, linewidth=4.5, label='REBOOT')
ax.tick_params(axis='both', which='major', labelsize=24, length=10, width=2)

# Labels & Annotations
ax.set_title('Combined Mortality Rate (Both Arms) - First 1 Year', fontsize=16, pad=20, fontweight='bold')
ax.set_xlabel('Time (Years)', fontsize=28, labelpad=20)
ax.set_ylabel('Cumulative Incidence of Death (%)', fontsize=28, labelpad=20)
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 9)

# Colored Annotations to match lines
ax.annotate(f"{y_commit[-1]:.1f}%", xy=(x_commit[-1], y_commit[-1]), xytext=(0.15, 7.8),
             arrowprops=dict(facecolor=color_commit, edgecolor=color_commit, arrowstyle='->'), 
             fontsize=36, color=color_commit, fontweight='bold')

ax.annotate(f"{y_korea[-1]:.1f}%", xy=(1.0, y_korea[-1]), xytext=(0.85, y_korea[-1]+0.4), 
             fontsize=36, color=color_korea, fontweight='bold')

ax.annotate(f"{y_reboot[-1]:.1f}%", xy=(1.0, y_reboot[-1]), xytext=(0.85, y_reboot[-1]+0.4), 
             fontsize=36, color=color_reboot, fontweight='bold')

# Clean Legend
ax.legend(fontsize=28, loc='upper right', frameon=False)

# Subtle Grid
ax.grid(True, alpha=0.15, color=color_text, linestyle='--')

plt.tight_layout()
plt.savefig('playground/Plots/compareWithOldDate.png', transparent=True, dpi=300)
#plt.show()