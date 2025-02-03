import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator, FixedLocator, FixedFormatter
# Data
outcomes = ['Paroxysmal AF', 'Persistent', 'All']
mean_diff = [0.23, 0.26, 0.25]
ci_lower = [0.15, 0.14, 0.1]
ci_upper = [0.42, 0.46, 0.54]

# Convert mean differences and confidence intervals to positive values for log scale
mean_diff_log = np.exp(mean_diff)
ci_lower_log = np.exp(ci_lower)
ci_upper_log = np.exp(ci_upper)

# Set default font properties and background color
plt.rcParams.update({
    'font.size': 24,
    'font.family': 'inter',
    'axes.facecolor': 'none',  # Transparent background for axes
    'figure.facecolor': 'black',  # Transparent background for figure
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'axes.edgecolor': 'white',  # Edge color for axes
    'grid.color': 'lightgray',  # Color for grid lines
})

# Create the forest plot
fig, ax = plt.subplots(figsize=(12, 4))

# Plot the mean differences
#ax.errorbar(mean_diff_log, outcomes, xerr=[np.subtract(mean_diff_log, ci_lower_log), np.subtract(ci_upper_log, mean_diff_log)], fmt='o', color='black', ecolor='gray', capsize=5)
ax.errorbar(mean_diff, outcomes, xerr=[np.subtract(mean_diff, ci_lower), np.subtract(ci_upper, mean_diff)], fmt='o', color='cyan', ecolor='lightgray', capsize=10, markersize=20,elinewidth=4)

# Add a vertical line at one (log scale equivalent of zero)
ax.axvline(x=1, color='white', linestyle='--',linewidth=3)

# Set labels
ax.set_xlabel('Geometric Mean Difference at 6mos (log scale) with 95% CI', fontsize=20)
ax.set_xscale('log')

# Set x-axis limits
ax.set_xlim([0.04, 2])  # Adjust these values as needed

# Customize x-axis tick line size and number format
ax.tick_params(axis='x', which='both', length=8, width=1)  # Adjust tick line size
#ax.xaxis.set_major_formatter(ScalarFormatter())  # Set number format for major ticks
#ax.xaxis.set_minor_formatter(ScalarFormatter())  # Set number format for minor ticks

# Set specific x-axis labels
ax.xaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 2]))
ax.xaxis.set_major_formatter(FixedFormatter(['0.01', '0.1', '1', '2']))

# Reduce y-axis spacing
ax.set_yticks(np.arange(len(outcomes)))
ax.set_yticklabels(outcomes)

# Add labels for mean_diff and 95% CI
#for i, (md, cl, cu) in enumerate(zip(mean_diff_log, ci_lower_log, ci_upper_log)):
    #ax.text(md, outcomes[i], f'{mean_diff[i]:.2f}', va='center', ha='left', color='white', fontsize=16)
    #ax.text(md, outcomes[i], f'[{ci_lower[i]:.2f}, {ci_upper[i]:.2f}]', va='center', ha='right', color='white', fontsize=16)



# Show the plot
plt.show()