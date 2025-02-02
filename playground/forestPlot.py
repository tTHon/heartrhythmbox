import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Data
n1, x1 = 1812, 432  # Sample size and events for P1
n2, x2 = 1821, 384  # Sample size and events for P2

# Calculate proportions
p1 = x1 / n1  # ≈ 0.238
p2 = x2 / n2  # ≈ 0.211

# Normal approximation CI
se1 = np.sqrt(p1 * (1 - p1) / n1)
se2 = np.sqrt(p2 * (1 - p2) / n2)

ci1_lower_normal = p1 - 1.96 * se1
ci1_upper_normal = p1 + 1.96 * se1
ci2_lower_normal = p2 - 1.96 * se2
ci2_upper_normal = p2 + 1.96 * se2

# Calculate the difference and its standard error
diff = p1 - p2
se_diff = np.sqrt(se1**2 + se2**2)

# Calculate the 95% CI for the difference
ci_diff_lower = diff - 1.96 * se_diff
ci_diff_upper = diff + 1.96 * se_diff

# Plot the difference with its confidence interval
fig, ax = plt.subplots(figsize=(10, 6))

ax.errorbar(diff, 0, xerr=[[diff - ci_diff_lower], [ci_diff_upper - diff]], fmt='o', color='black', capsize=5)
ax.axvline(0, color='gray', linestyle='--', label='No Difference')

# Add non-inferiority margin line
non_inferiority_margin = 0.03
ax.axvline(non_inferiority_margin, color='red', linestyle='-', label='Non-inferiority Margin (3%)')

# Add labels and title
ax.set_xlabel('Difference in Proportions (P1 - P2)')
ax.set_yticks([])
ax.set_title('Difference in Proportions with 95% CI and Non-inferiority Margin')
ax.legend()

plt.tight_layout()
plt.show()