import matplotlib.pyplot as plt
import numpy as np

# Data
subgroups = ["Preserved LVEF", "Reduced LVEF"]
rr = [0.79, 0.60]
ci_lower = [0.59, 0.36]
ci_upper = [1.07, 1.00]

# Calculate error bar lengths
xerr_low = np.array(rr) - np.array(ci_lower)
xerr_high = np.array(ci_upper) - np.array(rr)
xerr = [xerr_low, xerr_high]

# Setup the dark theme style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Arial', 'DejaVu Sans', 'sans-serif']

# Colors for dark theme
bg_color = 'none'      # Dark gray background
text_color = '#f1f1f1'    # Light gray text
accent_color = '#DFC8E4'  # Light blue for points
grid_color = '#d1d1d1'    # Darker gray for grid
line_color = '#F1f1f1'    # Red for reference line

# Create figure with reduced height (2.5 inches) to decrease spacing between rows
fig, ax = plt.subplots(figsize=(12, 4))

# Set background colors for the entire figure and the plotting area
fig.patch.set_facecolor(bg_color)
ax.set_facecolor(bg_color)

# Remove spines (borders)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color(grid_color)

# Add grid
ax.grid(axis='x', linestyle='--', alpha=0.4, color=grid_color)
ax.set_axisbelow(True)

# Plot Reference Line (RR=1)
ax.axvline(1, color=line_color, linestyle='-', linewidth=3, alpha=0.8, zorder=1)
# Adjust text position slightly based on new compact layout
#ax.text(1.02, 1.4, 'No Effect (RR=1)', color=line_color, fontsize=9, va='top')

# Plot Error Bars and Points
ax.errorbar(rr, np.arange(len(subgroups)), xerr=xerr, fmt='o', 
            color=accent_color, ecolor=accent_color, elinewidth=8, capsize=8, markersize=14, 
            zorder=3)

# Configure Y-axis
ax.set_yticks(np.arange(len(subgroups)))
ax.set_yticklabels(subgroups, fontsize=20, fontweight='bold', color=text_color)
ax.invert_yaxis() 
ax.tick_params(axis='y', length=0) 

# Add annotations next to the bars
for i, (val, low, high) in enumerate(zip(rr, ci_lower, ci_upper)):
    label_text = f"RR {val:.2f} (95% CI: {low:.2f} - {high:.2f})"
    ax.text(high + 0.05, i, label_text, va='center', fontsize=18, color=text_color)

# X-axis label and styling
ax.set_xlabel("Relative Risk (RR)", fontsize=20, color=text_color, labelpad=10)
ax.tick_params(axis='x', colors=text_color, labelsize=18)

# Title
plt.title("Association between β-blockers and All-Cause Death\n(LVEF Subgroup)", 
          fontsize=18, fontweight='bold', color=text_color, pad=20, loc='left')

# Adjust layout
plt.tight_layout()

# Save with facecolor parameter to ensure the saved file background is dark
plt.savefig("playground/Plots/bb_lvef.png", bbox_inches='tight', dpi=300, facecolor=bg_color)
#plt.show()