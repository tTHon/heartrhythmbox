import matplotlib.pyplot as plt
import numpy as np

# Data from the study
lvef_change_MICRA = -10.3
lvef_change_conv = -13.4
gls_change_MICRA = 5.7
gls_change_conv = 5.2
MICRA_tr_grades = [59.3, 22.2, 14.8, 3.7]  # Grade I, II, III, IV
conv_tr_grades = [58.3, 37.5, 4.2, 0.0]  # Grade I, II, III, IV
tr_grades_labels = ["Grade I", "Grade II", "Grade III", "Grade IV"]
ntprobnp_change_MICRA = -56
ntprobnp_change_conv = 409

# P-values
lvef_p_value = 0.218
gls_p_value = 0.778
tricuspid_p_value = 0.009  # Overall comparison between groups
ntprobnp_p_value = 0.013

# Create the plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Summary of Major Findings: Leadless vs. Conventional Pacemakers", fontsize=16)

# LVEF Change
bars_lvef = axes[0, 0].bar(["MICRA", "CONV"], [lvef_change_MICRA, lvef_change_conv], color=["skyblue", "lightcoral"])
axes[0, 0].set_title(f"Change in LVEF (p = {lvef_p_value})")
axes[0, 0].set_ylabel("Change (%)")
axes[0, 0].bar_label(bars_lvef)

# GLS Change
bars_gls = axes[0, 1].bar(["MICRA", "CONV"], [gls_change_MICRA, gls_change_conv], color=["skyblue", "lightcoral"])
axes[0, 1].set_title(f"Change in GLS (p = {gls_p_value})")
axes[0, 1].set_ylabel("Change (units)")
axes[0, 1].bar_label(bars_gls)

# Tricuspid Regurgitation Grade Distribution
width = 0.35  # Width of the bars
x = np.arange(len(tr_grades_labels))

bars_MICRA_tr = axes[1, 0].bar(x - width/2, MICRA_tr_grades, width, label="MICRA", color="skyblue")
bars_conv_tr = axes[1, 0].bar(x + width/2, conv_tr_grades, width, label="CONV", color="lightcoral")

axes[1, 0].set_title(f"Tricuspid Regurgitation (p = {tricuspid_p_value})")
axes[1, 0].set_ylabel("Percentage of Patients")
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(tr_grades_labels)
axes[1, 0].legend()
axes[1, 0].bar_label(bars_MICRA_tr, padding=3)
axes[1, 0].bar_label(bars_conv_tr, padding=3)

# NT-proBNP Change
bars_ntprobnp = axes[1, 1].bar(["MICRA", "CONV"], [ntprobnp_change_MICRA, ntprobnp_change_conv], color=["skyblue", "lightcoral"])
axes[1, 1].set_title(f"Change in NT-proBNP (p = {ntprobnp_p_value})")
axes[1, 1].set_ylabel("Change (pg/mL)")
axes[1, 1].bar_label(bars_ntprobnp)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()