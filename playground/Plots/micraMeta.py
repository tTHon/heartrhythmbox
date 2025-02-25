import matplotlib.pyplot as plt

# Data from the forest plots (from the provided text)
outcomes = ['Total Complications', 'All-Cause Mortality', 'Device-Related Complications']
risk_ratios = [0.67, 0.8, 0.49]  # Example values, adjust as needed
confidence_intervals = [(0.47, 0.94), (0.63, 1.03), (0.43, 0.57)]  # Example values, adjust as needed

# Colors for the plot
colors = ['skyblue', 'lightcoral', 'lightgreen']

# Set default font properties
plt.rcParams.update({
    'font.size': 22,
    'font.family': 'inter',
    'axes.facecolor': 'none',
    'figure.facecolor': 'none',
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'legend.facecolor': 'black',
    'legend.edgecolor': 'black'
})

# Convert outcomes to numeric values for plotting with reduced vertical gap
outcome_positions = [i * 0.1 for i in range(len(outcomes))]  # Reduced spacing

# Plotting
plt.figure(figsize=(16, 8))
plt.scatter(risk_ratios, outcome_positions, marker='s', color=colors, s=100, edgecolor='black', zorder=3, linewidths=10)
plt.hlines(y=outcome_positions, xmin=[ci[0] for ci in confidence_intervals], xmax=[ci[1] for ci in confidence_intervals], color=colors, linewidth=10, zorder=2)
plt.axvline(x=1, color='gray', linestyle='--', linewidth=2, zorder=1)  # Line of no effect
plt.xscale('log')  # Set x-axis to logarithmic scale
#plt.xlabel('Risk Ratio', fontsize=18, fontweight='bold')
#plt.ylabel('Outcome', fontsize=18, fontweight='bold')
plt.title('Risk Ratios & 95% CI for Outcomes: MICRA vs. Conventional PMs', fontsize=22, fontweight='bold')
plt.yticks(outcome_positions, outcomes, fontsize=24, fontweight='bold')
plt.grid(axis='x', linestyle='--', linewidth=0.7)

# Set custom x-ticks in decimal format
xticks = [0.1, 0.2, 0.5, 1, 2]
plt.xticks(xticks, [f'{x:.2f}' for x in xticks], fontsize=16)

# Add data labels with adjusted y offset
for i, (rr, outcome, ci) in enumerate(zip(risk_ratios, outcome_positions, confidence_intervals)):
    if outcome == outcome_positions[2]:
        plt.text(rr, outcome - 0.02, f'{rr}', ha='center', va='bottom', fontsize=20, color='black', zorder=4)
        plt.text(rr+0.18, outcome - 0.02, f'[{ci[0]}, {ci[1]}]', ha='center', va='bottom', fontsize=20, color='black', zorder=4)
    else:
        plt.text(rr, outcome + 0.01, f'{rr}', ha='center', va='bottom', fontsize=20, color='black', zorder=4)
        plt.text(rr+0.27, outcome + 0.01, f'[{ci[0]}, {ci[1]}]', ha='center', va='bottom', fontsize=20, color='black', zorder=4)


plt.tight_layout()

# Show plot
plt.show()