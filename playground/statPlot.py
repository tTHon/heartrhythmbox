import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Data
n1, x1 = 1812, 432  # Sample size and events for P1
n2, x2 = 1821, 384  # Sample size and events for P2

# Calculate proportions
p1 = x1/n1  # ≈ 0.238
p2 = x2/n2  # ≈ 0.211

# Calculate standard errors
se1 = np.sqrt(p1 * (1-p1) / n1)
se2 = np.sqrt(p2 * (1-p2) / n2)
se_diff = np.sqrt(se1**2 + se2**2)

# Calculate difference and its 95% CI
diff = p1 - p2
ci_lower = diff - 1.96 * se_diff
ci_upper = diff + 1.96 * se_diff

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Plot 1: Individual distributions
x = np.linspace(0, 0.5, 1000)
y1 = stats.norm.pdf(x, p1, se1)
y2 = stats.norm.pdf(x, p2, se2)

ax1.plot(x, y1, 'b-', label=f'P1 = {p1:.3f}')
ax1.plot(x, y2, 'g-', label=f'P2 = {p2:.3f}')
ax1.axvline(p1, color='b', linestyle='--', alpha=0.3)
ax1.axvline(p2, color='g', linestyle='--', alpha=0.3)
ax1.set_title('Individual Probability Distributions')
ax1.set_xlabel('Probability')
ax1.set_ylabel('Density')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Difference distribution with noninferiority margin
x_diff = np.linspace(-0.1, 0.1, 1000)
y_diff = stats.norm.pdf(x_diff, diff, se_diff)

ax2.plot(x_diff, y_diff, 'purple', label=f'Difference (P1-P2) = {diff:.3f}')
ax2.axvline(0, color='gray', linestyle='--', alpha=0.5, label='No Difference')
ax2.axvline(0.03, color='red', label='Noninferiority Margin (3%)')
ax2.axvline(ci_lower, color='black', linestyle='--', alpha=0.5, label=f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
ax2.axvline(ci_upper, color='black', linestyle='--', alpha=0.5)

# Fill the area beyond noninferiority margin
x_fill = x_diff[x_diff >= 0.03]
y_fill = stats.norm.pdf(x_fill, diff, se_diff)
ax2.fill_between(x_fill, y_fill, color='red', alpha=0.2)

ax2.set_title('Difference Distribution with Noninferiority Margin')
ax2.set_xlabel('Difference in Proportions (P1 - P2)')
ax2.set_ylabel('Density')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add text box with statistical information
stats_text = (
    f'P1 = {p1:.3f} ({x1}/{n1})\n'
    f'P2 = {p2:.3f} ({x2}/{n2})\n'
    f'Difference = {diff:.3f}\n'
    f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]\n'
    f'Noninferiority margin: 0.03\n'
    f'Conclusion: {"NOT noninferior" if ci_upper > 0.03 else "Noninferior"}'
)
plt.figtext(0.15, 0.02, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()