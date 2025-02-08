import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.ticker import ScalarFormatter

# Set font to 'Roboto'
rcParams['font.family'] = 'Roboto'

# Random Effects Model data
surgical_random_effects = {
    "OR": 0.95,  # Example value, replace with actual data
    "CI_lower": 0.92,  # Example value, replace with actual data
    "CI_upper": 0.98,  # Example value, replace with actual data
    "label": "Random Effects Model (Surgical)"
}

medical_random_effects = {
    "OR": 0.94,  # Example value, replace with actual data
    "CI_lower": 0.90,  # Example value, replace with actual data
    "CI_upper": 0.97,  # Example value, replace with actual data
    "label": "Random Effects Model (Medical + Anesthesia)"
}

combined_random_effects = {
    "OR": 0.95,  # Example value, replace with actual data
    "CI_lower": 0.93,  # Example value, replace with actual data
    "CI_upper": 0.97,  # Example value, replace with actual data
    "label": "Random Effects Model (Combined)"
}

# Create the forest plot
plt.figure(figsize=(20, 15))  # Adjust height for better visibility
y_pos = np.arange(3) 

# Plot Random Effects Models
plt.plot(surgical_random_effects["OR"], 0, 'D', color='darkorange', markersize=18)
plt.hlines(0, surgical_random_effects["CI_lower"], surgical_random_effects["CI_upper"], colors='darkorange', linewidth=10)

plt.plot(medical_random_effects["OR"], 1, 'D', color='royalblue', markersize=18)
plt.hlines(1, medical_random_effects["CI_lower"], medical_random_effects["CI_upper"], colors='royalblue', linewidth=12)

plt.plot(combined_random_effects["OR"], 2, 'D', color='forestgreen', markersize=18)
plt.hlines(2, combined_random_effects["CI_lower"], combined_random_effects["CI_upper"], colors='forestgreen', linewidth=12)

# Add a line for null effect
plt.axvline(x=1, color='red', linestyle='--')

# Add labels and title
plt.yticks(y_pos, [surgical_random_effects["label"], medical_random_effects["label"], combined_random_effects["label"]], fontsize=26)
plt.xlabel('Odds Ratio (OR)', fontsize=22)
plt.title('Forest Plot of Random Effects Models', fontsize=26)
plt.xscale('log')
plt.grid(True, axis='x', linestyle='--', alpha=0.7)

# Adjust x-axis limits for better scale
plt.xlim(0.8, 1.1)

# Set x-axis ticks and format them as numerical values
plt.xticks([0.8,0.9, 1, 1.1,], fontsize=22)
ax = plt.gca()
ax.xaxis.set_major_formatter(ScalarFormatter())

# Increase tick line thickness
ax.tick_params(axis='both', width=2)
# Increase plot border line thickness
for spine in ax.spines.values():
    spine.set_linewidth(2)


# Add labels for random effects models with larger font size
plt.text(surgical_random_effects["CI_upper"] + 0.04, 0, f'{surgical_random_effects["OR"]:.2f} [{surgical_random_effects["CI_lower"]:.2f}, {surgical_random_effects["CI_upper"]:.2f}]', va='center', ha='left', fontsize=24, fontweight='bold')
plt.text(medical_random_effects["CI_upper"] + 0.04, 1, f'{medical_random_effects["OR"]:.2f} [{medical_random_effects["CI_lower"]:.2f}, {medical_random_effects["CI_upper"]:.2f}]', va='center', ha='left', fontsize=24, fontweight='bold')
plt.text(combined_random_effects["CI_upper"] + 0.04, 2, f'{combined_random_effects["OR"]:.2f} [{combined_random_effects["CI_lower"]:.2f}, {combined_random_effects["CI_upper"]:.2f}]', va='center', ha='left', fontsize=24, fontweight='bold')

# Add OR label above the plot with more blank space
#plt.text(2.5, 2.5, 'Odds Ratio (OR)', ha='center', va='bottom', fontsize=24, fontweight='bold')

# Reverse Y-axis
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()