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
se1 = np.sqrt(p1 * (1-p1) / n1)
se2 = np.sqrt(p2 * (1-p2) / n2)
se_diff = np.sqrt(se1**2 + se2**2)

# Calculate difference and its 95% CI
diff = p1 - p2
diff_percentage = diff * 100  # Convert difference to percentage
ci_lower = diff - 1.96 * se_diff
ci_upper = diff + 1.96 * se_diff
ci_lower_percentage = ci_lower * 100  # Convert CI lower bound to percentage
ci_upper_percentage = ci_upper * 100  # Convert CI upper bound to percentage

# Print the results
print(f'Event rate for P1: {p1_percentage:.2f}%')
print(f'Event rate for P2: {p2_percentage:.2f}%')
print(f'Difference in event rates: {diff_percentage:.2f}%')
print(f'95% CI for the difference: [{ci_lower_percentage:.2f}%, {ci_upper_percentage:.2f}%]')

# Create figure
plt.figure(figsize=(12, 8))

# Define focused x-range (zoom in around the relevant area)
#x_min = min(p2 - 0.05, p1 - 0.05)  # Start a bit before P2
#x_max = p2 + 0.05  # End a bit after noninferiority margin
x_min = 0.1
x_max = 0.3
x = np.linspace(x_min, x_max, 1000)

# Plot individual distributions
y1 = stats.norm.pdf(x, p1, se1)
y2 = stats.norm.pdf(x, p2, se2)

plt.plot(x, y1, 'b-', label=f'P1 = {p1:.3f}', linewidth=2)
plt.plot(x, y2, 'g-', label=f'P2 = {p2:.3f}', linewidth=2)

# Add vertical lines for point estimates
plt.axvline(p1, color='b', linestyle='--', alpha=0.3)
plt.axvline(p2, color='g', linestyle='--', alpha=0.3)

# Add noninferiority margin relative to P2
margin_line = p2 + 0.03
plt.axvline(margin_line, color='red', label='Noninferiority Margin (P2 + 3%)', linewidth=2)

# Fill the region beyond noninferiority margin
x_fill = x[x >= margin_line]
y_fill = np.zeros_like(x_fill)
plt.fill_between(x_fill, y_fill, stats.norm.pdf(x_fill, p1, se1), 
                 color='red', alpha=0.1, label='Region beyond NI margin')

# Add confidence interval for P1
ci_upper_p1 = p1 + 1.96 * se1
ci_lower_p1 = p1 - 1.96 * se1
plt.hlines(y=max(y1)/2, xmin=ci_lower_p1, xmax=ci_upper_p1, 
          colors='blue', linestyles=':', label='95% CI for P1')

# Format x-axis to show percentages
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.1f}%'))

# Add annotations
plt.title('Distribution Plot with Noninferiority Margin Analysis\n(Zoomed to Relevant Range)', pad=20)
plt.xlabel('Probability')
plt.ylabel('Density')

# Add text box with statistical information
stats_text = (
    f'P1 = {p1*100:.1f}% ({x1}/{n1})\n'
    f'P2 = {p2*100:.1f}% ({x2}/{n2})\n'
    f'Difference (P1-P2) = {(diff*100):.1f}%\n'
    f'P1 95% CI: [{ci_lower_p1*100:.1f}%, {ci_upper_p1*100:.1f}%]\n'
    f'Noninferiority margin: P2 + 3% = {margin_line*100:.1f}%\n'
    f'Conclusion: {"NOT noninferior" if ci_upper_p1 > margin_line else "Noninferior"}'
)
plt.text(x_min + 0.005, max(y1), stats_text, fontsize=10, 
         bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9))

# Add vertical lines with percentage labels
for x_val, label in [(p1, 'P1'), (p2, 'P2'), (margin_line, 'NI margin')]:
    plt.text(x_val, -max(y1)/10, f'{label}\n{x_val*100:.1f}%', 
             horizontalalignment='center', verticalalignment='top')

plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()