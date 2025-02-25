import matplotlib.pyplot as plt

# Data from the forest plots (from the provided text)
outcomes = ['Total Complications', 'All-Cause Mortality', 'Device-Related Complications']
risk_ratios = [0.67, 0.8, 0.49]  # Example values, adjust as needed
confidence_intervals = [(0.47, 0.94), (0.63, 1.03), (0.43, 0.57)]  # Example values, adjust as needed

# Colors for the plot
colors = ['skyblue', 'lightcoral', 'lightgreen']

# Convert outcomes to numeric values for plotting
outcome_positions = [i * 0.5 for i in range(len(outcomes))]

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(risk_ratios, outcome_positions, marker='s', color=colors)
plt.hlines(y=outcome_positions, xmin=[ci[0] for ci in confidence_intervals], xmax=[ci[1] for ci in confidence_intervals], color=colors)
plt.axvline(x=1, color='gray', linestyle='--')  # Line of no effect
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.xlabel('Risk Ratio')
plt.ylabel('Outcome')
plt.title('Forest Plot of Three Outcomes')
plt.yticks(outcome_positions, outcomes)
plt.grid(axis='x', linestyle='--')

# Set custom x-ticks in decimal format
xticks = [0.1,0.2, 0.5, 1, 2]
plt.xticks(xticks, [f'{x:.2f}' for x in xticks])

# Add data labels with more y offset
for i, (rr, outcome, ci) in enumerate(zip(risk_ratios, outcome_positions, confidence_intervals)):
    plt.text(rr, outcome + 0.01, f'{rr}', ha='center', va='bottom', fontsize=10, color='black')
    plt.text(rr+0.2, outcome + 0.01, f'[{ci[0]}, {ci[1]}]', ha='center', va='bottom', fontsize=10, color='black')

plt.tight_layout()

# Show plot
plt.show()