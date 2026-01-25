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
# PLOTTING (Adjusted for White Background)
# ---------------------------------------------------------
plt.figure(figsize=(10, 7))

# Reset to default or set specific light-mode properties
plt.rcParams.update({
    'font.size': 18,
    'font.family': 'sans-serif',
    'axes.facecolor': 'none',       # Transparent
    'figure.facecolor': 'none',     # Transparent
    'text.color': 'black',          # Black text
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'legend.facecolor': 'none',
    'legend.edgecolor': 'black'
})

# Plot Lines (Darker colors for visibility on white)
# COMMIT: Black
plt.plot(x_commit, y_commit, color='black', linewidth=3, label='COMMIT (Combined)')

# Korea: Purple
plt.plot(x_korea, y_korea, color='purple', linewidth=3, linestyle='--', label='Korea Registry (Combined)')

# REBOOT: Green
plt.plot(x_reboot, y_reboot, color='green', linewidth=3, linestyle='-.', label='REBOOT (Combined)')

# Formatting
plt.title('Combined Mortality Rate (Both Arms) - First 1 Year', fontsize=16, color='black')
plt.xlabel('Time (Years)', fontsize=14, color='black')
plt.ylabel('Cumulative Incidence of Death (%)', fontsize=14, color='black')
plt.xlim(0, 1.0)
plt.ylim(0, 9)

# Annotations (Black)
plt.annotate(f"{y_commit[-1]:.1f}%", xy=(x_commit[-1], y_commit[-1]), xytext=(0.15, 7.8),
             arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->'), 
             fontsize=12, color='black')

plt.annotate(f"{y_korea[-1]:.1f}%", xy=(1.0, y_korea[-1]), xytext=(0.85, y_korea[-1]+0.5), 
             fontsize=12, color='black')

plt.annotate(f"{y_reboot[-1]:.1f}%", xy=(1.0, y_reboot[-1]), xytext=(1.02, y_reboot[-1]), 
             fontsize=12, color='black')

# Legend
legend = plt.legend(fontsize=11)
plt.setp(legend.get_texts(), color='black')

# Grid (Dark gray for white bg)
plt.grid(True, alpha=0.3, color='black', linestyle='--')
plt.tight_layout()

# Save with transparency (but elements are dark)
plt.savefig('playground/Plots/compareWithOldDate_white_bg.png', transparent=True, dpi=300)
plt.show()