import matplotlib.pyplot as plt
import numpy as np

# Data from Table 2
years = ['2 Years', '4 Years', '7 Years']

# Event rates for Preventive PCI + OMT
tv_mi_pci = [0.1, 0.6, 1.0]  # Target-vessel-related MI
any_revasc_pci = [1.8, 4.6, 8.5]  # Any revascularisation
ischaemia_revasc_pci = [0.1, 1.7, 4.9]  # Ischaemia-driven target vessel revascularisation

# Event rates for OMT Alone
tv_mi_omt = [0.8, 0.9, 1.4]
any_revasc_omt = [3.7, 6.1, 12.4]
ischaemia_revasc_omt = [2.4, 4.4, 8.0]

# Bar width
bar_width = 0.25
x = np.arange(len(years))

# Create 3 separate bar charts
fig, axes = plt.subplots(1, 3, figsize=(10, 10))

# Set default font properties and background color
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'inter',
    'axes.facecolor': 'white',  # Black background for axes
    'figure.facecolor': 'white',  # Black background for figure
    'text.color': 'black',  # Text color
    'axes.labelcolor': 'darkgray',  # Label color for axes
    'xtick.color': 'darkgray',  # X-axis tick color
    'ytick.color': 'none',  # Y-axis tick color
    'axes.edgecolor': 'darkgray',  # Edge color for axes
    'grid.color': 'none',  # Color for grid lines
    'xtick.labelsize': 18,  # X-axis tick label size
    'ytick.labelsize': 18,  # Y-axis tick label size
})

# Set the same y-axis limits for all subplots
y_max = max(max(tv_mi_pci + tv_mi_omt), max(any_revasc_pci + any_revasc_omt), max(ischaemia_revasc_pci + ischaemia_revasc_omt)) * 1.2

# Target-Vessel-Related Myocardial Infarction
axes[0].bar(x - bar_width/2, tv_mi_pci, bar_width, label='Preventive PCI + OMT', color='r')
axes[0].bar(x + bar_width/2, tv_mi_omt, bar_width, label='OMT Alone', color='b')
axes[0].set_title('Target-Vessel-Related MI', pad=20, color='black')
axes[0].set_ylabel('Event Rate (%)', color='black')
axes[0].set_xticks(x)
axes[0].set_xticklabels(years, color='black')
axes[0].legend()
axes[0].grid(False)
axes[0].set_ylim(0, y_max)
axes[0].tick_params(axis='y', labelsize=16)
axes[0].tick_params(axis='x', labelsize=16)

# Add labels on each bar
for i in range(len(years)):
    axes[0].annotate(f'{tv_mi_pci[i]:.1f}', xy=(x[i] - bar_width/2, tv_mi_pci[i]), ha='center', va='bottom', color='black', fontsize=16)
    axes[0].annotate(f'{tv_mi_omt[i]:.1f}', xy=(x[i] + bar_width/2, tv_mi_omt[i]), ha='center', va='bottom', color='black', fontsize=16)

# Any Revascularisation
axes[1].bar(x - bar_width/2, any_revasc_pci, bar_width, color='r')
axes[1].bar(x + bar_width/2, any_revasc_omt, bar_width, color='b')
axes[1].set_title('Any Revascularisation', pad=20, color='black')
axes[1].set_xticks(x)
axes[1].set_xticklabels(years, color='black')
axes[1].grid(False)
axes[1].set_ylim(0, y_max)
axes[1].tick_params(axis='y', labelsize=16)
axes[1].tick_params(axis='x', labelsize=16)

# Add labels on each bar
for i in range(len(years)):
    axes[1].annotate(f'{any_revasc_pci[i]:.1f}', xy=(x[i] - bar_width/2, any_revasc_pci[i]), ha='center', va='bottom', color='black', fontsize=16)
    axes[1].annotate(f'{any_revasc_omt[i]:.1f}', xy=(x[i] + bar_width/2, any_revasc_omt[i]), ha='center', va='bottom', color='black', fontsize=16)

# Ischaemia-Driven Target Vessel Revascularisation
axes[2].bar(x - bar_width/2, ischaemia_revasc_pci, bar_width, color='r')
axes[2].bar(x + bar_width/2, ischaemia_revasc_omt, bar_width, color='b')
axes[2].set_title('Ischaemia-Driven Target Vessel Revascularisation', pad=20, color='black')
axes[2].set_xticks(x)
axes[2].set_xticklabels(years, color='black')
axes[2].grid(False)
axes[2].set_ylim(0, y_max)
axes[2].tick_params(axis='y', labelsize=16)
axes[2].tick_params(axis='x', labelsize=16)

# Add labels on each bar
for i in range(len(years)):
    axes[2].annotate(f'{ischaemia_revasc_pci[i]:.1f}', xy=(x[i] - bar_width/2, ischaemia_revasc_pci[i]), ha='center', va='bottom', color='black', fontsize=16)
    axes[2].annotate(f'{ischaemia_revasc_omt[i]:.1f}', xy=(x[i] + bar_width/2, ischaemia_revasc_omt[i]), ha='center', va='bottom', color='black', fontsize=16)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()