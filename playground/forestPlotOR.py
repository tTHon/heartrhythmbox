import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Set font to 'Roboto'
rcParams['font.family'] = 'Roboto'

# Data from the file
surgical_studies = [
    "Blohm, Sandblom, Enochsson, et al.", "Chai, Chen, Lin & Lin", "Jobback, Rognark, Bedeschi, et al.",
    "O'Neill, Lanska & Hartz", "Okoshi, Endo & Nomura", "Sharoky, Sellers, Keele, et al.",
    "Sun, Boet, Chan, et al.", "Tsugawa, Jena, Orav, et al.", "Wallis, Jerath, Satkunasivam, et al.",
    "Wallis, Ravi, Coburn, et al."
]

medical_studies = [
    "Becker, Siry–Bove, Shelton, et al.", "Berg, Hurtig & Steinsbek", "Dwyer & Kalin",
    "Haubitz–Eshchelbach, Mirsada, Sebastian, et al.", "Jerath, Satkunasivam, Kaneshwaran, et al.",
    "Meier, Yang, Liu, et al.", "Nakayama, Morita, Fujiwara & Komuro", "Sagy, Fuchs, Mizrahi, et al.",
    "Sergeant, Saha, Shin et al.", "Tsugawa, Jena, Figueroa, et al.", "Yelavarthy, Seth, Pielsticker, et al."
]

all_studies = surgical_studies + medical_studies

logOR = [
    -0.1906, -0.7765, -1.0498, -0.7550, -0.1508, 0.0929, -0.1398, -0.0305, -0.0619, -0.1278,
    0.1667, 0.1222, -0.0274, -0.0834, -0.1044, -0.3378, -0.5539, 0.1240, -0.0677, -0.0420, -0.1625
]

SE_logOR = [
    0.2217, 0.4702, 1.0080, 0.9906, 0.1059, 0.0695, 0.1595, 0.0211, 0.0191, 0.0608,
    0.2450, 0.1449, 0.2409, 0.0956, 0.0298, 0.1393, 0.4065, 0.1661, 0.0452, 0.0064, 0.1342
]

# Calculate odds ratios and confidence intervals
OR = np.exp(logOR)
CI_lower = np.exp(np.array(logOR) - 1.96 * np.array(SE_logOR))
CI_upper = np.exp(np.array(logOR) + 1.96 * np.array(SE_logOR))

# Random Effects Model data
surgical_random_effects = {
    "logOR": -0.1278,  # Example value, replace with actual data
    "SE_logOR": 0.0608,  # Example value, replace with actual data
    "label": "Random Effects Model (Surgical)"
}

medical_random_effects = {
    "logOR": -0.1625,  # Example value, replace with actual data
    "SE_logOR": 0.1342,  # Example value, replace with actual data
    "label": "Random Effects Model (Medical + Anesthesia)"
}

combined_random_effects = {
    "logOR": -0.0420,  # Example value, replace with actual data
    "SE_logOR": 0.0064,  # Example value, replace with actual data
    "label": "Random Effects Model (Combined)"
}

# Calculate OR and CI for random effects models
def calculate_random_effects(random_effects):
    OR = np.exp(random_effects["logOR"])
    CI_lower = np.exp(random_effects["logOR"] - 1.96 * random_effects["SE_logOR"])
    CI_upper = np.exp(random_effects["logOR"] + 1.96 * random_effects["SE_logOR"])
    return OR, CI_lower, CI_upper

surgical_OR, surgical_CI_lower, surgical_CI_upper = calculate_random_effects(surgical_random_effects)
medical_OR, medical_CI_lower, medical_CI_upper = calculate_random_effects(medical_random_effects)
combined_OR, combined_CI_lower, combined_CI_upper = calculate_random_effects(combined_random_effects)

# Create the forest plot
plt.figure(figsize=(10, 14))
y_pos = np.arange(len(all_studies) + 3)  # +3 for the random effects models

# Plot Surgical studies
surgical_indices = np.arange(len(surgical_studies))
plt.errorbar(OR[surgical_indices], surgical_indices, xerr=[OR[surgical_indices] - CI_lower[surgical_indices], CI_upper[surgical_indices] - OR[surgical_indices]], fmt='o', color='blue', capsize=5)

# Plot Medical + Anesthesia studies
medical_indices = np.arange(len(surgical_studies), len(all_studies))
plt.errorbar(OR[medical_indices], medical_indices, xerr=[OR[medical_indices] - CI_lower[medical_indices], CI_upper[medical_indices] - OR[medical_indices]], fmt='o', color='green', capsize=5)

# Plot Random Effects Models
plt.plot(surgical_OR, len(all_studies), 'D', color='blue', markersize=10)
plt.hlines(len(all_studies), surgical_CI_lower, surgical_CI_upper, colors='blue', linewidth=2)

plt.plot(medical_OR, len(all_studies) + 1, 'D', color='green', markersize=10)
plt.hlines(len(all_studies) + 1, medical_CI_lower, medical_CI_upper, colors='green', linewidth=2)

plt.plot(combined_OR, len(all_studies) + 2, 'D', color='purple', markersize=10)
plt.hlines(len(all_studies) + 2, combined_CI_lower, combined_CI_upper, colors='purple', linewidth=2)

# Add a line for null effect
plt.axvline(x=1, color='red', linestyle='--')

# Add labels and title
plt.yticks(y_pos, list(all_studies) + [surgical_random_effects["label"], medical_random_effects["label"], combined_random_effects["label"]])
plt.xlabel('Odds Ratio (OR)')
plt.title('Forest Plot of Surgical and Medical + Anesthesia Studies with Random Effects Models')
plt.xscale('log')
plt.grid(True, axis='x', linestyle='--', alpha=0.7)

# Add a separator between Surgical and Medical sections
plt.axhline(y=len(surgical_studies) - 0.5, color='black', linestyle='-', linewidth=1)

# Add labels for OR and 95% CI
for i in range(len(all_studies)):
    plt.text(CI_upper[i] + 0.1, i, f'{OR[i]:.2f} [{CI_lower[i]:.2f}, {CI_upper[i]:.2f}]', va='center', ha='left', fontsize=8)

# Add labels for random effects models with larger font size
plt.text(surgical_CI_upper + 0.1, len(all_studies), f'{surgical_OR:.2f} [{surgical_CI_lower:.2f}, {surgical_CI_upper:.2f}]', va='center', ha='left', fontsize=10, fontweight='bold')
plt.text(medical_CI_upper + 0.1, len(all_studies) + 1, f'{medical_OR:.2f} [{medical_CI_lower:.2f}, {medical_CI_upper:.2f}]', va='center', ha='left', fontsize=10, fontweight='bold')
plt.text(combined_CI_upper + 0.1, len(all_studies) + 2, f'{combined_OR:.2f} [{combined_CI_lower:.2f}, {combined_CI_upper:.2f}]', va='center', ha='left', fontsize=10, fontweight='bold')

# Reverse Y-axis
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()