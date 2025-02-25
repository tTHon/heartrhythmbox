import matplotlib.pyplot as plt
import numpy as np

# Data from the study
lvef_change_MICRA = -10.3
lvef_change_conv = -13.4
MICRA_tr_grades = [77.7, 30.8, 11.5, 0]  # Grade I, II, III, IV
conv_tr_grades = [13.0, 69.6,13.0, 4.3]  # Grade I, II, III, IV
tr_grades_labels = ["No change", "+1 grade", "+2 grades", "+3 grades"]
ntprobnp_change_MICRA = -56
ntprobnp_change_conv = 409

# P-values
lvef_p_value = 0.218
tricuspid_p_value = 0.009  # Overall comparison between groups
ntprobnp_p_value = 0.013

# Set default font properties
plt.rcParams.update({
    'font.size': 20,
    'font.family': 'inter',
    'axes.facecolor': 'none',
    'figure.facecolor': 'none',
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'legend.facecolor': 'none',
    'legend.edgecolor': 'black'
})

# Create the plot
fig, axes = plt.subplots(1, 3, figsize=(25, 15))
#fig.suptitle("Summary of Major Findings: Leadless vs. Conventional Pacemakers", fontsize=16)

# LVEF Change
bars_lvef = axes[0].bar(["MICRA", "CONV"], [lvef_change_MICRA, lvef_change_conv], color=["skyblue", "lightcoral"])
axes[0].set_title(f"Change in LVEF (p = {lvef_p_value})")
axes[0].set_ylabel("Change (%)")
axes[0].bar_label(bars_lvef, fontsize=16)

# Tricuspid Regurgitation Grade Distribution
width = 0.3  # Width of the bars
x = np.arange(len(tr_grades_labels))

bars_MICRA_tr = axes[1].bar(x - width/2, MICRA_tr_grades, width, label="MICRA", color="skyblue")
bars_conv_tr = axes[1].bar(x + width/2, conv_tr_grades, width, label="CONV", color="lightcoral")

axes[1].set_title(f"Change in Tricuspid Regurgitation (p = {tricuspid_p_value})")
axes[1].set_ylabel("Percentage of Patients")
axes[1].set_xticks(x)
axes[1].set_xticklabels(tr_grades_labels)
axes[1].legend()
axes[1].bar_label(bars_MICRA_tr, padding=2, fontsize=16)
axes[1].bar_label(bars_conv_tr, padding=2, fontsize=16)

# NT-proBNP Change
bars_ntprobnp = axes[2].bar(["MICRA", "CONV"], [ntprobnp_change_MICRA, ntprobnp_change_conv], color=["skyblue", "lightcoral"])
axes[2].set_title(f"Change in NT-proBNP (p = {ntprobnp_p_value})")
axes[2].set_ylabel("Change (pg/mL)")
axes[2].bar_label(bars_ntprobnp, fontsize=16)

# Add a zero line on the y-axis for the NT-proBNP Change plot
axes[2].axhline(0, color='gray', linestyle='--', linewidth=2)

# Adjust the spacing between the plots
plt.subplots_adjust(wspace=0.15)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()