import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Data
n1, x1 = 1812, 432  # Sample size and events for P1
n2, x2 = 1821, 384  # Sample size and events for P2

# Calculate proportions
p1 = x1/n1  # ≈ 0.238
p2 = x2/n2  # ≈ 0.211

# Convert proportions to percentages
p1_percentage = p1 * 100  # ≈ 23.8%
p2_percentage = p2 * 100  # ≈ 21.1%


# Calculate standard errors
#se1 = np.sqrt(p1 * (1-p1) / n1)
#se2 = np.sqrt(p2 * (1-p2) / n2)
#se_diff = np.sqrt(se1**2 + se2**2)

# Calculate standard errors
se1 = np.sqrt(p1 * (1 - p1) / n1) * 100  # Convert to percentage
se2 = np.sqrt(p2 * (1 - p2) / n2) * 100  # Convert to percentage
se_diff = np.sqrt(se1**2 + se2**2)


# Calculate difference and its 95% CI
diff = p1 - p2
diff_percentage = diff * 100  # Convert difference to percentage
ci_lower = diff - 1.96 * se_diff
ci_upper = diff + 1.96 * se_diff
ci_lower_percentage = ci_lower * 100  # Convert CI lower bound to percentage
ci_upper_percentage = ci_upper * 100  # Convert CI upper bound to percentage


# Set default font properties
plt.rcParams.update({
    'font.size': 18,
    'font.family': 'sans-serif',
        'axes.facecolor': 'none',
    'figure.facecolor': 'none',
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'legend.facecolor': 'white',
    'legend.edgecolor': 'white'
})


# Create figure with two subplots
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
fig, ax1 = plt.subplots(1, 1, figsize=(15, 10))

# Plot 1: Individual distributions
#x = np.linspace(0.15, 0.3, 1000)
x = np.linspace(15, 30, 1000)
#y1 = stats.norm.pdf(x, p1, se1)
#y2 = stats.norm.pdf(x, p2, se2)
y1 = stats.norm.pdf(x, p1_percentage, se1)
y2 = stats.norm.pdf(x, p2_percentage, se2)

ax1.plot(x, y1, 'pink', label=f'P1 = {p1:.3f}', linewidth=5)
ax1.plot(x, y2, 'cyan', label=f'P2 = {p2:.3f}',linewidth=5)
#ax1.axvline(p1, color='pink', linestyle='--', alpha=0.7, linewidth=4)
#ax1.axvline(p2, color='cyan', linestyle='--', alpha=0.7, linewidth=4)
ax1.axvline(p1_percentage, color='pink', linestyle='--', alpha=0.7, linewidth=4)
ax1.axvline(p2_percentage, color='cyan', linestyle='--', alpha=0.7, linewidth=4)
# ax1.set_title('Individual Probability Distributions')
ax1.set_xlabel('Event Rate',color='white')
ax1.set_ylabel('Density', color='white')
# ax1.legend()
# ax1.grid(True, alpha=0.3, color='white',linewidth=2)

# Set x-axis line thickness and color
ax1.spines['bottom'].set_linewidth(4)
ax1.spines['bottom'].set_color('white')
ax1.spines['left'].set_linewidth(4)
ax1.spines['left'].set_color('white')

# Calculate and plot 95% CI for p1 and p2
ci_p1_lower = p1 - 1.96 * se1
ci_p1_upper = p1 + 1.96 * se1
ci_p2_lower = p2 - 1.96 * se2
ci_p2_upper = p2 + 1.96 * se2

#ax1.axvline(ci_p1_lower, color='pink', linestyle=':', alpha=0.7, linewidth=3)
#ax1.axvline(ci_p1_upper, color='pink', linestyle=':', alpha=0.7, linewidth=3)
#ax1.axvline(ci_p2_lower, color='cyan', linestyle=':', alpha=0.7, linewidth=3)
#ax1.axvline(ci_p2_upper, color='cyan', linestyle=':', alpha=0.7, linewidth=3)

# Add noninferiority margin relative to P2
#margin_line = p2 + 0.03
#plt.axvline(margin_line, color='maroon', label='Noninferiority Margin (P2 + 3%)', linewidth=5)

# Label the 95% CI and mean
#ax1.text(p1, max(y1), f'Mean P1\n{p1:.3f}', color='cyan', ha='center', va='bottom', fontsize=12)
#ax1.text(ci_p1_lower, max(y1) / 2, f'95% CI Lower\n{ci_p1_lower:.3f}', color='cyan', ha='center', va='bottom', fontsize=12)
#ax1.text(ci_p1_upper, max(y1) / 2, f'95% CI Upper\n{ci_p1_upper:.3f}', color='cyan', ha='center', va='bottom', fontsize=12)

#ax1.text(p2, max(y2), f'Mean P2\n{p2:.3f}', color='pink', ha='center', va='bottom', fontsize=12)
#ax1.text(ci_p2_lower, max(y2) / 2, f'95% CI Lower\n{ci_p2_lower:.3f}', color='pink', ha='center', va='bottom', fontsize=12)
#ax1.text(ci_p2_upper, max(y2) / 2, f'95% CI Upper\n{ci_p2_upper:.3f}', color='pink', ha='center', va='bottom', fontsize=12)

ax1.set_xlabel('Event Rate', color='white')
ax1.set_ylabel('Density', color='white')
# sax1.legend()
#ax1.grid(True, alpha=0.3, color='white')

# Plot 2: Difference distribution with noninferiority margin
# x_diff = np.linspace(-0.1, 0.1, 1000)
# y_diff = stats.norm.pdf(x_diff, diff, se_diff)

# ax2.plot(x_diff, y_diff, 'purple', label=f'Difference (P1-P2) = {diff:.3f}')
# ax2.axvline(0, color='gray', linestyle='--', alpha=0.5, label='No Difference')
# ax2.axvline(0.03, color='red', label='Noninferiority Margin (3%)')
# ax2.axvline(ci_lower, color='black', linestyle='--', alpha=0.5, label=f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
# ax2.axvline(ci_upper, color='black', linestyle='--', alpha=0.5)

# Fill the area beyond noninferiority margin
# x_fill = x_diff[x_diff >= 0.03]
# y_fill = stats.norm.pdf(x_fill, diff, se_diff)
# ax2.fill_between(x_fill, y_fill, color='red', alpha=0.2)

# ax2.set_title('Difference Distribution with Noninferiority Margin')
# ax2.set_xlabel('Difference in Proportions (P1 - P2)')
# ax2.set_ylabel('Density')
# ax2.legend()
# ax2.grid(True, alpha=0.3)

# Add text box with statistical information
# stats_text = (
  #  f'P1 = {p1:.3f} ({x1}/{n1})\n'
  #  f'P2 = {p2:.3f} ({x2}/{n2})\n'
  #  f'Difference = {diff:.3f}\n'
  #  f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]\n'
  #  f'Noninferiority margin: 0.03\n'
  #  f'Conclusion: {"NOT noninferior" if ci_upper > 0.03 else "Noninferior"}'
#)
# plt.figtext(0.15, 0.02, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()