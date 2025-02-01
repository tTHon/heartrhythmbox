import numpy as np
import matplotlib.pyplot as plt

# Data
n1, x1 = 1812, 432  # Sample size and events for P1
n2, x2 = 1821, 384  # Sample size and events for P2

# Calculate proportions and standard errors
p1 = x1/n1
p2 = x2/n2
se1 = np.sqrt(p1 * (1-p1) / n1)
se2 = np.sqrt(p2 * (1-p2) / n2)
se_diff = np.sqrt(se1**2 + se2**2)

# Calculate difference and its 95% CI
diff = p1 - p2
ci_lower = diff - 1.96 * se_diff
ci_upper = diff + 1.96 * se_diff

# Create figure
plt.figure(figsize=(12, 6))

# Define y-positions
y_pos = [2, 1]  # Positions for individual proportions and difference

# Plot individual proportions and difference
plt.scatter([p1], [y_pos[0]], color='blue', s=100, zorder=5, label='P1')
plt.scatter([p2], [y_pos[1]], color='green', s=100, zorder=5, label='P2')

# Add confidence intervals
plt.hlines(y_pos[0], p1 - 1.96*se1, p1 + 1.96*se1, color='blue', zorder=4)
plt.hlines(y_pos[1], p2 - 1.96*se2, p2 + 1.96*se2, color='green', zorder=4)

# Add vertical end caps
for y in y_pos:
    if y == y_pos[0]:  # P1
        plt.vlines(p1 - 1.96*se1, y-0.1, y+0.1, color='blue')
        plt.vlines(p1 + 1.96*se1, y-0.1, y+0.1, color='blue')
    else:  # P2
        plt.vlines(p2 - 1.96*se2, y-0.1, y+0.1, color='green')
        plt.vlines(p2 + 1.96*se2, y-0.1, y+0.1, color='green')

# Add reference line at P2
plt.axvline(p2, color='gray', linestyle='--', alpha=0.5, label='P2 Reference')

# Add noninferiority margin
ni_margin = p2 + 0.03
plt.axvline(ni_margin, color='red', linestyle='--', label='Noninferiority Margin')

# Customize plot
plt.ylim(0.5, 2.5)
plt.xlim(0.15, 0.30)  # Adjust as needed to show all relevant data

# Format x-axis as percentages
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.1f}%'))

# Add labels
plt.title('Forest Plot: Noninferiority Analysis', pad=20)
plt.xlabel('Probability')

# Replace y-axis ticks with labels
plt.yticks(y_pos, [
    f'P1: {p1*100:.1f}% ({x1}/{n1})',
    f'P2: {p2*100:.1f}% ({x2}/{n2})'
])

# Add text box with statistical information
stats_text = (
    f'Difference (P1-P2) = {diff*100:.1f}%\n'
    f'95% CI: [{ci_lower*100:.1f}%, {ci_upper*100:.1f}%]\n'
    f'Noninferiority margin: 3%\n'
    f'Conclusion: {"NOT noninferior" if ci_upper > 0.03 else "Noninferior"}'
)
plt.text(0.16, 0.7, stats_text, 
         bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9))

plt.grid(True, alpha=0.3, axis='x')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()