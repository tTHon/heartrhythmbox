import matplotlib.pyplot as plt

# Data extracted from Table 2 of POTCAST.pdf
labels = [
    "Primary Composite End Point",
    "Appropriate ICD Therapy / VT",
    "Hospitalization for Arrhythmia",
    "Hospitalization for Heart Failure",
    "Death from Any Cause"
]

# Incidence Data: (n, %)
# Group 1: High-Normal Potassium (N=600)
group_intervention = [
    (136, 22.7),
    (92, 15.3),
    (40, 6.7),
    (21, 3.5),
    (34, 5.7)
]

# Group 2: Standard Care (N=600)
group_control = [
    (175, 29.2),
    (122, 20.3),
    (64, 10.7),
    (33, 5.5),
    (41, 6.8)
]

# Hazard Ratios
hrs = [0.76, 0.75, 0.63, 0.64, 0.85]

# 95% Confidence Intervals (Lower, Upper)
cis = [
    (0.61, 0.95),
    (0.57, 0.98),
    (0.42, 0.93),
    (0.37, 1.11),
    (0.54, 1.34)
]

# Calculate error bar lengths
error_lower = [hr - ci[0] for hr, ci in zip(hrs, cis)]
error_upper = [ci[1] - hr for hr, ci in zip(hrs, cis)]
errors = [error_lower, error_upper]

# Settings for large font and dark background
font_size = 40
plt.rcParams.update({'font.size': font_size, 'font.family': 'inter'})

# Colors suitable for dark blue background
bg_color = 'none' # DarkBlue
text_color = 'white'
marker_color = '#A7FC96' # DeepSkyBlue
line_color = 'white'
grid_color = 'gray'

# Create the plot
fig, ax = plt.subplots(figsize=(40, 16))
fig.patch.set_facecolor(bg_color)
ax.set_facecolor(bg_color)

# Plot the data points with error bars
y_pos = range(len(labels))
# Adjusting xlims to fit the forest plot on the right side
plot_xlims = (0.2, 2.0)

ax.errorbar(hrs, y_pos, xerr=errors, fmt='o', markersize=30, color=marker_color, ecolor=line_color, capsize=15, linewidth=10)

# Add a vertical line at HR = 1
ax.axvline(x=1, color=grid_color, linestyle='--', linewidth=3)

# Remove default y-axis labels
ax.set_yticks([]) 
ax.set_xlim(plot_xlims)

# Add column headers with white text
header_y = -1
ax.text(-1.8, header_y, "End Point", fontweight='bold', ha='left', color=text_color)
ax.text(-0.6, header_y, "High-Normal K\n(N=600)\nn (%)", fontweight='bold', ha='center', color=text_color)
ax.text(0.0, header_y, "Standard Care\n(N=600)\nn (%)", fontweight='bold', ha='center', color=text_color)
ax.text(1.6, header_y, "Hazard Ratio\n(95% CI)", fontweight='bold', ha='center', color=text_color)


# Add data rows
for i in range(len(labels)):
    y = i
    
    # Endpoint Label
    ax.text(-1.8, y, labels[i], va='center', ha='left', color=text_color)
    
    # Intervention Data
    n_int, p_int = group_intervention[i]
    ax.text(-0.6, y, f"{n_int} ({p_int}%)", va='center', ha='center', color=text_color)
    
    # Control Data
    n_ctrl, p_ctrl = group_control[i]
    ax.text(0.0, y, f"{n_ctrl} ({p_ctrl}%)", va='center', ha='center', color=text_color)
    
    # HR text
    hr, ci = hrs[i], cis[i]
    hr_text = f"{hr:.2f} ({ci[0]:.2f}-{ci[1]:.2f})"
    ax.text(1.6, y, hr_text, va='center', ha='center', color=text_color)

# Formatting
ax.invert_yaxis()  # Primary endpoint at the top
ax.set_xlabel('Hazard Ratio (log scale)', fontsize=font_size, labelpad=20, color=text_color)
#ax.set_title('Primary Endpoint and Components (POTCAST Trial)', fontsize=font_size+4, pad=40, fontweight='bold', color=text_color)

# Customize spines
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_color(line_color)
ax.spines['bottom'].set_linewidth(2)

# Customize ticks
ax.tick_params(axis='x', colors=text_color, width=4, length=10)
ax.set_xticks([0.5, 1.0, 1.5])
ax.set_xticklabels(['0.5', '1.0', '1.5'])

# Allow drawing outside the axes
ax.set_clip_on(False)

# Adjust layout manually
plt.subplots_adjust(left=0.48, right=0.88, top=0.75, bottom=0.1)

# Save command to demonstrate transparency works
plt.savefig('playground/Plots/Potcast.png', dpi=300, transparent=True)
#plt.show()