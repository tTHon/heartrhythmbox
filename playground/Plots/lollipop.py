import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. Data Setup
data = {
    'Treatment': [
        'ARNI + BB + MRA + SGLT2i', 'ARNI + BB + MRA', 'ACEI + BB + MRA + SGLT2i',
        'ACEI + BB + Dig + H-ISDN', 'ARNI + BB + SGLT2i', 'ACEI + BB + MRA',
        'ARB + BB + MRA', 'ARNI + BB', 'ACEI + ARB + BB + Dig',
        'ACEI + BB + Dig', 'ACEI + BB', 'ARB + BB + Dig',
        'ARB + BB', 'BB'
    ],
    'HR': [0.39, 0.45, 0.46, 0.47, 0.52, 0.52, 0.56, 0.59, 0.65, 0.68, 0.69, 0.73, 0.74, 0.78],
    'Lower_CI': [0.32, 0.37, 0.38, 0.36, 0.43, 0.44, 0.47, 0.51, 0.55, 0.59, 0.62, 0.63, 0.66, 0.72],
    'Upper_CI': [0.49, 0.54, 0.56, 0.61, 0.63, 0.62, 0.65, 0.69, 0.76, 0.78, 0.77, 0.83, 0.82, 0.84]
}
df = pd.DataFrame(data)

# Sort: Worst (High HR/Long Bar) to Best (Low HR/Short Bar)
df = df.sort_values('HR', ascending=False).reset_index(drop=True)

# 2. Design Settings
BG_COLOR = '#1E1E1E'
TEXT_COLOR = '#FFFFFF'
STATS_COLOR = '#DDDDDD'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Arial', 'Helvetica', 'DejaVu Sans']

# 3. Radius Calculation (Reversed: Zero at Center)
# Now Radius is exactly the HR value.
df['Radius'] = df['HR']
df['Rad_Lower'] = df['Lower_CI']
df['Rad_Upper'] = df['Upper_CI']

# 4. Polar Setup
N = len(df)
start_angle = np.pi / 3  # Start at 2 o'clock
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
angles = start_angle - theta # Clockwise

# Figure Size
fig = plt.figure(figsize=(30, 26), facecolor=BG_COLOR)
ax = fig.add_subplot(111, projection='polar', facecolor=BG_COLOR)

# 5. Colors
norm = plt.Normalize(df['HR'].min(), df['HR'].max())
colors = plt.cm.cool(norm(df['HR']))

# 6. Grid Lines (Standard HR values)
# We plot lines at 0.2, 0.4, 0.6, 0.8, 1.0
grid_values = [0.2, 0.4, 0.6, 0.8, 1.0]
for val in grid_values:
    ax.plot(np.linspace(0, 2*np.pi, 100), [val]*100, color='white', linestyle='--', linewidth=4, alpha=0.3)
    # Label Grid
    ax.text(np.pi/2, val + 0.02, f"{val}", color='#ababab', fontsize=28, fontweight='bold', ha='center')

# 7. Shading (Wedges)
width = (2 * np.pi) / N
ax.bar(angles, df['Radius'], width=width, color=colors, alpha=0.3, zorder=1)

# 8. Data Plotting
for i, (angle, row) in enumerate(zip(angles, df.itertuples())):
    # Stick: From Center (0) to Data Point
    ax.plot([angle, angle], [0, row.Radius], color=colors[i], linewidth=5, alpha=0.8, zorder=5)
    
    # Error Bar
    ax.plot([angle, angle], [row.Rad_Lower, row.Rad_Upper], color='#EEEEEE', linewidth=3, zorder=6)
    
    # Dot
    ax.scatter(angle, row.Radius, color=colors[i], s=600, zorder=10, edgecolors='white', linewidth=3)

# 9. Labels
for angle, label, hr, ci_low, ci_high, radius in zip(angles, df['Treatment'], df['HR'], df['Lower_CI'], df['Upper_CI'], df['Radius']):
    
    # Rotation Logic
    norm_angle = (angle + np.pi) % (2 * np.pi) - np.pi
    if -np.pi/2 <= norm_angle <= np.pi/2:
        alignment = "left"
        rotation = np.rad2deg(norm_angle)
    else:
        alignment = "right"
        rotation = np.rad2deg(norm_angle) + 180

    # --- SPACING ---
    # With Scale Reversed, bars near the end are short (0.39).
    # We need to ensure labels don't crash into the center or each other.
    # We push them out relative to their specific bar length.
    
    label_r_drug = radius + 0.08 
    label_r_stats = radius + 0.26 
    
    # Drug Name
    ax.text(angle, label_r_drug, label, 
            rotation=rotation, ha=alignment, va='center', 
            fontsize=32, fontweight='bold', color=TEXT_COLOR)

    # Stats
    #stats_text = f"{hr:.2f} [{ci_low}-{ci_high}]"
    #ax.text(angle, label_r_stats, stats_text, 
            #rotation=rotation, ha=alignment, va='center', 
            #fontsize=28, color=STATS_COLOR)

# 10. Cleanup
ax.set_ylim(0, 1.1) # Set limit to accommodate the 1.0 grid line and labels
ax.spines['polar'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# Center Label (Origin)
ax.text(0, 0, "0", color='#666666', fontsize=24, ha='center', va='center')

plt.title("Treatment Effectiveness (All-Cause Mortality)\nCenter = 0.0 | Shortest Bar = Best Outcome", 
          y=1.10, fontsize=40, fontweight='bold', color=TEXT_COLOR)

plt.tight_layout()
#plt.show()

# Save with transparent background as requested
plt.savefig('playground/Plots/lollipop.png', dpi=300, transparent=True)
