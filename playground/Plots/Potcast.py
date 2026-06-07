import matplotlib.pyplot as plt


labels = [
    "Hemorrhagic Stroke",
    "All Major Bleeding \n(Procedural & Non-Procedural)",
    "Non-procedural Related \nMajor Bleeding",
    "All major bleeding \n+ CRNMB",
    "Non-procedural Related \nMajor Bleeding + CRNMB"
]

# Incidence of ischemic stroke/se Data: (n, %)
# OAC
group_intervention = [
    (5, 0.4),
    (87, 6.4),
    (87, 6.4),
    (260, 19.0),
    (260, 19.0)
]

# Group 2: LAAO
group_control = [
    (5, 0.4),
    (83, 5.9),
    (71, 5.1),
    (181, 12.8),
    (154, 10.9)
]

# Hazard Ratios
hrs = [0.96, 0.92, 0.78, 0.66,0.55]

# 95% Confidence Intervals (Lower, Upper)
cis = [
    (0.28, 3.32),
    (0.68, 1.24),
    (0.57, 1.06),
    (0.54, 0.80),
    (0.45, 0.67)
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
marker_color = "#F26868" # DeepSkyBlue
line_color = '#eeeeea'
grid_color = 'gray'

# Create the plot
fig, ax = plt.subplots(figsize=(44, 20), dpi=300)
fig.patch.set_facecolor(bg_color)
ax.set_facecolor(bg_color)

# Plot the data points with error bars
y_pos = range(len(labels))

ax.errorbar(hrs, y_pos, xerr=errors, fmt='o', markersize=45, color=marker_color, ecolor=line_color, capsize=20, linewidth=15, capthick=5)

# Add a vertical line at HR = 1
ax.axvline(x=1, color=grid_color, linestyle='--', linewidth=5)
#ax.axvline(x=0.5, color=grid_color, linestyle='--', linewidth=2, alpha=0.8)
#ax.axvline(x=3, color=grid_color, linestyle='--', linewidth=2, alpha=0.8)

# Remove default y-axis labels
ax.set_yticks([]) 
ax.set_ylim(-0.2, len(labels) - 0.7)
ax.set_xlim(0,4.0)

# Add column headers with white text
header_y = -0.7
#ax.text(-1, header_y, "Trials", fontweight='bold', ha='left', color=text_color)

#ax.text(-3, header_y, "Trials", fontweight='bold', ha='left', color=text_color)
ax.text(-1.4, header_y, "OAC (n, %)", fontweight='bold', ha='center', color=text_color)
ax.text(-0.4, header_y, "LAAO (n, %)", fontweight='bold', ha='center', color=text_color)
ax.text(4.0, header_y, "Hazard Ratio (95% CI)", fontweight='bold', ha='center', color=text_color)


# Add data rows
for i in range(len(labels)):
    y = i
    
    # Endpoint Label
    #ax.text(-3, y, labels[i], va='center', ha='left', color=text_color, fontweight='bold', fontsize=font_size+6)
    ax.text(-4, y, labels[i], va='center', ha='left', color=text_color, fontweight='bold', fontsize=font_size+6)
    
    # Intervention Data
    n_int, p_int = group_intervention[i]
    ax.text(-1.4, y, f"{n_int} ({p_int}%)", va='center', ha='center', color=text_color)
    
    # Control Data
    n_ctrl, p_ctrl = group_control[i]
    ax.text(-0.4, y, f"{n_ctrl} ({p_ctrl}%)", va='center', ha='center', color=text_color)
    
    # HR text
    hr, ci = hrs[i], cis[i]
    hr_text = f"{hr:.2f} ({ci[0]:.2f}-{ci[1]:.2f})"
    ax.text(4.0, y, hr_text, va='center', ha='center', color=text_color)

# Formatting
ax.invert_yaxis()  # Primary endpoint at the top
ax.set_xlabel('Hazard Ratio (log scale) of LAAO vs OAC', fontsize=font_size, labelpad=8, color=text_color)
#ax.set_title('Primary Endpoint and Components (POTCAST Trial)', fontsize=font_size+4, pad=40, fontweight='bold', color=text_color)

# Customize spines
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_color(line_color)
ax.spines['bottom'].set_linewidth(2)

# Customize ticks
ax.tick_params(axis='x', colors=text_color, width=4, length=10)
ax.set_xticks([0 ,0.5, 1.0, 2,4])
ax.set_xticklabels(['0', '0.5', '1.0', '2','4'])

# Allow drawing outside the axes
ax.set_clip_on(False)

# Adjust layout manually
plt.subplots_adjust(left=0.48, right=0.88, top=0.75, bottom=0.1)

# Save command to demonstrate transparency works
plt.savefig('playground/Plots/Potcast.png', dpi=300, transparent=True)
#plt.show()