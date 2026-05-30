import matplotlib.pyplot as plt

# Data from the forest plots
outcomes = ['Hemorrhagic', 'Ischemic','All Stroke']
risk_ratios = [0.96,1.61, 1.46]  
confidence_intervals = [(0.28, 3.32),(1.00, 2.59),(0.94, 2.27)]  

# Colors for the plot
colors = ['lightcoral','#FED154', '#2972b6']

# Set default font properties (Revised for visibility on dark/transparent backgrounds)
plt.rcParams.update({
    'font.size': 40,
    'font.family': 'inter',
    'axes.facecolor': 'none',
    'figure.facecolor': 'none',
    'text.color': 'white',          # Changed to white
    'axes.labelcolor': 'white',     # Changed to white
    'xtick.color': '#eeeeea',         # Changed to white
    'ytick.color': '#eeeeea',         # Changed to white
    'axes.edgecolor': '#ddddde'       # Added to make the border lines white
})

# Convert outcomes to numeric values for plotting
outcome_positions = [i * 0.1 for i in range(len(outcomes))]  

# Plotting
plt.figure(figsize=(20, 12))

# Changed edgecolor to white so the markers pop against a dark background
plt.scatter(risk_ratios, outcome_positions, marker='s', color=colors, s=500, edgecolor='white', zorder=3, linewidths=15)
plt.hlines(y=outcome_positions, xmin=[ci[0] for ci in confidence_intervals], xmax=[ci[1] for ci in confidence_intervals], color=colors, linewidth=10, zorder=2)

#--- NEW: Add vertical lines to act as caps on the confidence intervals ---
cap_height = 0.005  # Adjust this value to make the caps taller or shorter
ci_lower = [ci[0] for ci in confidence_intervals]
ci_upper = [ci[1] for ci in confidence_intervals]

# Left caps
plt.vlines(x=ci_lower, ymin=[y - cap_height for y in outcome_positions], ymax=[y + cap_height for y in outcome_positions], color=colors, linewidth=6, zorder=2)
# Right caps
plt.vlines(x=ci_upper, ymin=[y - cap_height for y in outcome_positions], ymax=[y + cap_height for y in outcome_positions], color=colors, linewidth=6, zorder=2)
# -------------------------------------------------------------------------

# Changed the line of no effect to lightgray so it shows up
plt.axvline(x=1, color='lightgray', linestyle='--', linewidth=2, zorder=1, alpha = 0.7)  
plt.xscale('log')  

#plt.title('Risk Ratios & 95% CI for Outcomes: MICRA vs. Conventional PMs', fontsize=22, fontweight='bold')
plt.xlabel('Hazard Ratio [95% CI] of LAAO vs OAC', fontsize=40, labelpad=15)
plt.yticks(outcome_positions, outcomes, fontsize=40)

# Changed grid line color slightly so it is visible but not overpowering
plt.grid(axis='x', linestyle='--', linewidth=0.7, color='gray', alpha=0.7)

# Set custom x-ticks in decimal format
xticks = [0.2, 0.5, 1, 2]
plt.xticks(xticks, [f'{x:.1f}' for x in xticks], fontsize=32)
plt.tick_params(axis='x', length=15, width=2, color='gray', pad=8)



# Add data labels with adjusted y offset (Changed text color to white)
for i, (rr, outcome, ci) in enumerate(zip(risk_ratios, outcome_positions, confidence_intervals)):
    if outcome == outcome_positions[2]:
        plt.text(rr, outcome - 0.025, f'{rr}', ha='center', va='bottom', fontsize=40, fontweight='bold', color='white', zorder=4)
        plt.text(rr+0.8, outcome - 0.025, f'[{ci[0]}, {ci[1]}]', ha='center', va='bottom', fontsize=40, color='white', zorder=4)
    if outcome == outcome_positions[1]:
        plt.text(rr, outcome + 0.01, f'{rr}', ha='center', va='bottom', fontsize=40, fontweight='bold', color='white', zorder=4)
        plt.text(rr+0.8, outcome + 0.01, f'[{ci[0]}, {ci[1]}]', ha='center', va='bottom', fontsize=40, color='white', zorder=4)
    if outcome == outcome_positions[0]:
        plt.text(rr, outcome + 0.01, f'{rr}', ha='center', va='bottom', fontsize=40, color='white',fontweight='bold', zorder=4)
        plt.text(rr+0.55, outcome + 0.01, f'[{ci[0]}, {ci[1]}]', ha='center', va='bottom', fontsize=40, color='white', zorder=4)

plt.tight_layout()

# Save plot with a transparent background
plt.savefig('playground/plots/champion.png', transparent=True, bbox_inches='tight')