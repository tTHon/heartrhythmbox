import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Data from Figure 3 in the paper
# [Subgroup, HR, Lower CI, Upper CI, Events in Catheter Ablation, Total in Catheter Ablation, 
#  Events in Drug Therapy, Total in Drug Therapy]
data = [
    # All patients
    ["All patients", 0.75, 0.58, 0.97, 103, 203, 129, 213],
    
    # Drug-eligibility stratum
    ["Amiodarone", 0.86, 0.61, 1.22, 59, 108, 67, 109],
    ["Sotalol", 0.64, 0.43, 0.94, 44, 95, 62, 104],
]

# Convert to numpy array for easier manipulation
data_array = np.array(data, dtype=object)

# Extract data
subgroups = data_array[:, 0]
hazard_ratios = data_array[:, 1].astype(float)
lower_ci = data_array[:, 2].astype(float)
upper_ci = data_array[:, 3].astype(float)
ca_events = data_array[:, 4].astype(int)
ca_total = data_array[:, 5].astype(int)
dt_events = data_array[:, 6].astype(int)
dt_total = data_array[:, 7].astype(int)

# Calculate error bars
lower_error = hazard_ratios - lower_ci
upper_error = upper_ci - hazard_ratios

# Create figure with specific dimensions
plt.figure(figsize=(16, 10), facecolor='black')

# Set margins
plt.subplots_adjust(left=0.6, right=0.95, top=0.95, bottom=0.1)

# Create plot elements
y_positions = np.arange(len(subgroups))[::-1]  # Reverse to match forest plot convention

# Create categories and separators
category_positions = {
    "ALL": [0],
    "DRUG-ELIGIBILITY STRATUM": [1, 2],
}

# Add colored background for categories
colors = ['#efefef', '#efefef']  # Use a more visible pink for the second category
category_names = list(category_positions.keys())

current_pos = 0
for i, (cat, positions) in enumerate(category_positions.items()):
    if len(positions) > 0:
        height = len(positions)
        # For the first category, use gray only for the first row; for the second, use pink for all its rows
        if i == 0:
            rect = patches.Rectangle((0, len(subgroups) - current_pos - 1), 1, 1, 
                                   transform=plt.gca().get_yaxis_transform(), 
                                   color=colors[0], alpha=0.3, zorder=0)
            plt.gca().add_patch(rect)
        else:
            rect = patches.Rectangle((0, len(subgroups) - current_pos - height), 1, height, 
                                   transform=plt.gca().get_yaxis_transform(), 
                                   color=colors[1], alpha=0.3, zorder=0)
            plt.gca().add_patch(rect)
        current_pos += height

# Plot hazard ratios and confidence intervals
plt.errorbar(hazard_ratios, y_positions, xerr=[lower_error, upper_error], 
             fmt='o', markersize=15, color='#AFFD8E', capsize=12, elinewidth=8)

# Add vertical line at HR=1
plt.axvline(x=1, color='pink', linestyle='--', alpha=0.7)

# Add horizontal line at x-axis (y=0)
plt.axhline(y=0, color='white', linestyle='-', linewidth=1, alpha=0.7)

# Format x-axis to logarithmic scale
plt.xscale('log')
plt.xlim(0.1, 2)
plt.grid(True, axis='x')
plt.xticks([0.1, 1, 2], ['0.1', '1', '2'], color='white', fontsize=14, fontname='Inter')

# Label subgroups
plt.yticks(y_positions, [])  # No default labels

# Add events and totals as text
for i, y in enumerate(y_positions):
    plt.text(-0.01, y, subgroups[i], ha='right', va='center', fontsize=16, color='white', fontname='Inter', transform=plt.gca().get_yaxis_transform())
    hr_text = f"{hazard_ratios[i]:.2f} ({lower_ci[i]:.2f}–{upper_ci[i]:.2f})"
    plt.text(1.05, y, hr_text, ha='left', va='center', fontsize=15, color='white', fontname='Inter', transform=plt.gca().get_yaxis_transform())

# Add column header for Hazard Ratio (95% CI) closer to the plot
#plt.text(1.05, len(subgroups) + 0.5, "Hazard Ratio (95% CI)", ha='left', va='center', fontweight='bold', fontsize=16, color='white', fontname='Inter', transform=plt.gca().get_yaxis_transform())

# Add "Better" label with arrows
plt.text(0.5, -0.7, u"\u25C0---- Catheter Ablation Better", ha='center', va='center', fontsize=14, color='white', fontname='Inter')
plt.text(1.7, -0.7, u"Drug Therapy Better ----\u25B6", ha='center', va='center', fontsize=14, color='white', fontname='Inter')

# Add category names above the corresponding variables
current_pos = 0
for idx, (cat, positions) in enumerate(category_positions.items()):
    if len(positions) > 0:
        y_pos = y_positions[positions[0]] + 0.5
        # Only add the label and horizontal line if not 'ALL'
        if idx != 0:
            cat_color = 'white'
            # Draw a horizontal line only under the label for 'DRUG-ELIGIBILITY STRATUM', not at 'All patients'
            plt.hlines(y=y_pos - 0.25, xmin=-0.2, xmax=1.2, colors=cat_color, linestyles='dashed', linewidth=2, transform=plt.gca().get_yaxis_transform(), zorder=2)
            plt.text(-0.01, y_pos, cat, ha='right', va='bottom', fontweight='bold', fontsize=18, color=cat_color, fontname='Inter', transform=plt.gca().get_yaxis_transform())
        current_pos += len(positions)

# Label the axes
plt.xlabel('Hazard Ratio (95% CI)', fontsize=18, fontweight='bold', color='pink', fontname='Inter')
#plt.title('VANISH2 Trial: Primary Endpoint Hazard Ratios', fontsize=20, fontweight='bold', pad=20, color='white', fontname='Inter')

# Hide y-axis
plt.gca().yaxis.set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Set figure and axes background to black
plt.gca().set_facecolor('black')
plt.gcf().patch.set_facecolor('black')

# Change tick label colors
plt.gca().tick_params(axis='x', colors='white')

plt.tight_layout()
plt.show()