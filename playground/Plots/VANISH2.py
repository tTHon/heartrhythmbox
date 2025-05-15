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
    
    # LVEF
    ["LVEF <35%", 0.80, 0.56, 1.15, 57, 105, 62, 101],
    ["LVEF ≥35%", 0.67, 0.46, 0.98, 46, 98, 67, 112],
    
    # Index event
    ["VT storm", 0.72, 0.45, 1.18, 27, 48, 42, 58],
    ["Appropriate shock", 0.83, 0.55, 1.25, 47, 84, 48, 80],
    ["ATP", 0.34, 0.15, 0.81, 9, 20, 14, 18],
    ["Sustained VT below detection limit of ICD", 0.86, 0.48, 1.55, 20, 51, 25, 57]
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

# Increase plot size
plt.figure(figsize=(30, 10))

# Change font to Inter
plt.rcParams['font.family'] = 'Inter'

# Adjust font sizes for larger text
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

# Invert colors
plt.style.use('dark_background')

# Reduce space between variables by adjusting subplot parameters
plt.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.2)

# Set transparent background for the plot and figure
plt.gca().patch.set_alpha(0)
plt.gcf().patch.set_alpha(0)

# Change font colors to match transparency
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'

# Create plot elements
y_positions = np.arange(len(subgroups))[::-1]  # Reverse to match forest plot convention

# Create categories and separators
category_positions = {
    "All": [0],
    "Drug-eligibility stratum": [1, 2],
    "LVEF": [3, 4],
    "Index event": [5, 6, 7, 8]
}

# Remove background for categories
# Commented out the code that adds colored background for categories
# for i, (cat, positions) in enumerate(category_positions.items()):
#     if len(positions) > 0:
#         height = len(positions)
#         rect = patches.Rectangle((0, len(subgroups) - current_pos - height), 1, height, 
#                                transform=plt.gca().get_yaxis_transform(), 
#                                color=colors[i % len(colors)], alpha=0.3, zorder=0)
#         plt.gca().add_patch(rect)
#         current_pos += height

# Assign a different color for each category
category_colors = ['#AFFD8E', '#8EC6FF', '#FFB48E']

# Assign a color to each data point based on its category
point_colors = []
for i in range(len(subgroups)):
    if i in category_positions['All']:
        point_colors.append(category_colors[0])
    elif i in category_positions['Drug-eligibility stratum']:
        point_colors.append(category_colors[1])
    elif i in category_positions['LVEF']:
        point_colors.append(category_colors[2])
    else:
        point_colors.append('#FFFFFF')  # fallback color

# Plot each point with its category color
for i, y in enumerate(y_positions):
    plt.errorbar(hazard_ratios[i], y, xerr=[[lower_error[i]], [upper_error[i]]],
                 fmt='o', markersize=8, color=point_colors[i], capsize=6, elinewidth=2)

# Change other colors to complement the palette
plt.axvline(x=1, color='#FF8EA0', linestyle='--', alpha=0.7)

# Format x-axis to logarithmic scale
plt.xscale('log')
plt.xlim(0.1, 10)
plt.grid(True, axis='x', alpha=0.3)

# Label subgroups
plt.yticks(y_positions, [])  # No default labels

# Add events and totals as text
for i, y in enumerate(y_positions):
    # Subgroup names on the left
    plt.text(-0.01, y, subgroups[i], ha='right', va='center', fontsize=11, transform=plt.gca().get_yaxis_transform())
    
    # Events and totals on the right side of the plot
    event_text = f"{ca_events[i]}/{ca_total[i]}"
    plt.text(1.01, y, event_text, ha='left', va='center', fontsize=10, transform=plt.gca().get_yaxis_transform())
    
    event_text_control = f"{dt_events[i]}/{dt_total[i]}"
    plt.text(1.11, y, event_text_control, ha='left', va='center', fontsize=10, transform=plt.gca().get_yaxis_transform())
    
    # Add hazard ratio and CI as text
    hr_text = f"{hazard_ratios[i]:.2f} ({lower_ci[i]:.2f}–{upper_ci[i]:.2f})"
    plt.text(1.21, y, hr_text, ha='left', va='center', fontsize=10, transform=plt.gca().get_yaxis_transform())

# Add column headers
plt.text(-0.01, len(subgroups) + 0.5, "Subgroup", ha='right', va='center', fontweight='bold', fontsize=12, transform=plt.gca().get_yaxis_transform())
plt.text(1.01, len(subgroups) + 0.5, "Catheter\nAblation", ha='left', va='center', fontweight='bold', fontsize=11, transform=plt.gca().get_yaxis_transform())
plt.text(1.11, len(subgroups) + 0.5, "Drug\nTherapy", ha='left', va='center', fontweight='bold', fontsize=11, transform=plt.gca().get_yaxis_transform())
plt.text(1.21, len(subgroups) + 0.5, "Hazard Ratio (95% CI)", ha='left', va='center', fontweight='bold', fontsize=11, transform=plt.gca().get_yaxis_transform())

# Add "Better" labels
plt.text(0.2, -0.7, "Catheter Ablation Better", ha='center', va='center', fontsize=10)
plt.text(5, -0.7, "Drug Therapy Better", ha='center', va='center', fontsize=10)

# Add categories on the left side
current_pos = 0
for cat, positions in category_positions.items():
    if len(positions) > 0:
        y_pos = len(subgroups) - current_pos - len(positions)/2
        plt.text(-0.15, y_pos, cat, ha='right', va='center', fontweight='bold', fontsize=12, 
                transform=plt.gca().get_yaxis_transform())
        current_pos += len(positions)

# Label the axes
plt.xlabel('Hazard Ratio (95% CI)', fontsize=12, fontweight='bold')
plt.title('VANISH2 Trial: Primary Endpoint Hazard Ratios', fontsize=14, fontweight='bold', pad=20)

# Hide y-axis
plt.gca().yaxis.set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.show()