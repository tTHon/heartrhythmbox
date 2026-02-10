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

# 2. Design Settings (Dark Mode)
BG_COLOR = '#1E1E1E'
TEXT_COLOR = '#FFFFFF'
STATS_COLOR = '#DDDDDD'
plt.rcParams['font.family'] = 'Inter'
plt.rcParams['font.sans-serif'] = ['Inter', 'Arial', 'Helvetica', 'DejaVu Sans']

# 3. Manual Radius Calculation (The Fix)
# We want HR 1.0 to be at the center (Radius ~ 0)
# We want HR 0.39 to be at the edge (Radius ~ High)
# Formula: Radius = (Reference - HR)
# We add a small offset (0.1) so 1.0 isn't a singularity at the exact center.
REF_VALUE = 1.1
df['Radius'] = REF_VALUE - df['HR']
df['Rad_Lower'] = REF_VALUE - df['Upper_CI'] # Invert CI logic
df['Rad_Upper'] = REF_VALUE - df['Lower_CI']

# 4. Polar Setup
N = len(df)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
angles = (np.pi / 2) - angles # Start at 12 o'clock

fig = plt.figure(figsize=(14, 14), facecolor=BG_COLOR)
ax = fig.add_subplot(111, projection='polar', facecolor=BG_COLOR)

# 5. Colors (Cool gradient - Inverted)
# Best drugs (Low HR, High Radius) get bright colors
norm = plt.Normalize(df['HR'].min(), df['HR'].max())
colors = plt.cm.cool(norm(df['HR']))

# 6. Plotting
# Draw "Fake" Grid Lines manually so we can label them correctly
grid_values = [1.0, 0.8, 0.6, 0.4]
for val in grid_values:
    r = REF_VALUE - val
    ax.plot(np.linspace(0, 2*np.pi, 100), [r]*100, color='white', linestyle='--', linewidth=1, alpha=0.3)
    # Label the grid lines
    ax.text(0, r + 0.02, f"{val}", color='#666666', fontsize=9, fontweight='bold', ha='center')

# Draw Data
for i, (angle, row) in enumerate(zip(angles, df.itertuples())):
    # Stick: From Center (Radius 0) to Data Point
    # Actually, let's start from the 1.0 ring (Radius = 0.1)
    r_start = REF_VALUE - 1.0
    
    ax.plot([angle, angle], [r_start, row.Radius], color=colors[i], linewidth=3, alpha=0.6)
    
    # Error Bar
    ax.plot([angle, angle], [row.Rad_Lower, row.Rad_Upper], color='#EEEEEE', linewidth=2)
    
    # Dot
    ax.scatter(angle, row.Radius, color=colors[i], s=250, zorder=10, edgecolors='white', linewidth=2)

# 7. Labels
for angle, label, hr, ci_low, ci_high, radius in zip(angles, df['Treatment'], df['HR'], df['Lower_CI'], df['Upper_CI'], df['Radius']):
    rotation = np.rad2deg(angle)
    alignment = "left"
    
    if angle < -np.pi/2 or angle > np.pi/2:
        rotation += 180
        alignment = "right"
    if rotation < 0: rotation += 360
    
    # Place Label OUTSIDE the point
    # Since we manually calculated radius, we just add a fixed amount to 'radius'
    label_r = radius + 0.05
    
    # Treatment Name
    ax.text(angle, label_r, label, 
            rotation=rotation, ha=alignment, va='center', fontsize=12, fontweight='bold', color=TEXT_COLOR)

    # Stats
    stats_text = f"{hr} [{ci_low}-{ci_high}]"
    ax.text(angle, label_r + 0.08, stats_text, 
            rotation=rotation, ha=alignment, va='center', fontsize=10, color=STATS_COLOR, fontfamily='monospace')

# 8. Cleanup
ax.set_ylim(0, 0.9) # Manual limit based on our Radius formula (1.1 - 0.39 = 0.71, so 0.9 is safe)
ax.spines['polar'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([]) # We drew manual grid lines

# Center Label
ax.text(0, 0, "Reference\n(1.0)", color='#666666', fontsize=10, ha='center', va='center')

plt.title("Treatment Effectiveness (All-Cause Mortality)\nOuter Edge = Best Outcome (Lowest HR)", 
          y=1.05, fontsize=20, fontweight='bold', color=TEXT_COLOR)

plt.tight_layout()


# Save with transparent background as requested
plt.savefig('playground/Plots/lollipop.png', dpi=300, transparent=True)
plt.show()