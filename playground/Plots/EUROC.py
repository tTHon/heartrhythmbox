import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Inverted Data Points (Converted from Survival to Risk %)
# ---------------------------------------------------------

# Graph A: Overall Cohort (Inverted: 100 - Survival%)
time_overall = [0, 40, 180, 250, 365, 550, 730]
risk_overall = [0.0, 5.0, 7.0, 8.5, 9.0, 11.0, 13.8]

# Graph C: Full DRT resolution (Inverted: 100 - Survival%)
time_full = [0, 100, 200, 365, 450, 580, 730]
risk_full = [0.0, 3.0, 6.0, 6.5, 8.0, 11.0, 12.3]

# Graph C: No DRT resolution (Inverted: 100 - Survival%)
time_none = [0, 30, 210, 365, 730]
risk_none = [0.0, 13.0, 17.6, 17.6, 17.6]

# ---------------------------------------------------------
# 2. Graph Styling & Base Setup
# ---------------------------------------------------------
plt.figure(figsize=(20, 16))

color1 = "#FFD400"  # "Minion Yellow" - Overall Cohort
color2 = "#469FF8"  # "Overalls Denim Blue" - Full DRT
color3 = "#CECFD0"  # "Goggles Silver" - No DRT
color4 = "#EEEEEA"  # Off-White/Light Gray for text readability
lineWidth = 8
axesLineWidth = 4
fontLabel = 42

plt.rcParams.update({
        'font.size': 36,
        'font.family': 'inter, sans-serif',
        'text.color': color4,
        'axes.labelcolor': color4,
        'xtick.color': color4,
        'ytick.color': color4,
        'axes.edgecolor': color4,
        'figure.facecolor': 'none', # Transparent Figure
        'axes.facecolor': 'none',   # Transparent Axis
        'savefig.facecolor': 'none' # Transparent Export
        })

# Setup the axes, limits, and vertical line first so they are present in all images
plt.axvline(x=365, color='#cccccc', linewidth=lineWidth-4, linestyle=':', alpha=0.6, zorder=1)

# Zoomed Y-axis to 0-25% so the risk curves fill the frame perfectly
plt.ylim(0.0, 25.0)
plt.xlim(0, 780)
plt.xlabel('Days after LAAC', labelpad=20, fontsize=40)
plt.ylabel('Stroke risk (%)', labelpad=20, fontsize=40)  # Updated Label

# Clean up axes
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Set Axis Linewidth ---
ax.spines['left'].set_linewidth(axesLineWidth)
ax.spines['bottom'].set_linewidth(axesLineWidth)
ax.tick_params(axis='both', which='major', length=10, width=axesLineWidth, color=color4, pad=8)

# Pushes the axes lines 25 points outward from the data area, creating the detached look
ax.spines['left'].set_position(('outward', 25))
ax.spines['bottom'].set_position(('outward', 25))

plt.tight_layout()

# ---------------------------------------------------------
# 3. Plotting & Saving Sequentially
# ---------------------------------------------------------

# --- IMAGE 1: Overall Cohort Only ---
plt.step(time_overall, risk_overall, where='post', label='Overall Cohort', 
         color=color1, linestyle='--', linewidth=lineWidth)

# Values inverted: 91.0% survival -> 9.0% risk | 86.2% survival -> 13.8% risk
plt.text(370, 9.1, '9.0%', color=color1, verticalalignment='bottom', fontweight='bold', fontsize=fontLabel)
plt.text(710, 13.8, '13.8%', color=color1, verticalalignment='bottom', fontweight='bold',fontsize=fontLabel)

plt.legend(loc='upper center', frameon=False, fontsize=40)
plt.savefig('playground/Plots/EUROC_step1.png', dpi=300, transparent=True)


# --- IMAGE 2: Add Full DRT Resolution ---
plt.step(time_full, risk_full, where='post', label='Full DRT resolution', 
         color=color2, linewidth=lineWidth) 

# Values inverted: 93.5% survival -> 6.5% risk | 87.7% survival -> 12.3% risk
# Placed text below lines ('top' alignment) to keep separation from the yellow line values
plt.text(370, 6.3, '6.5%', color=color2, verticalalignment='top', fontweight='bold', fontsize=fontLabel)
plt.text(710, 10.5, '12.3%', color=color2, verticalalignment='top', fontweight='bold', fontsize=fontLabel)

plt.legend(loc='upper center', frameon=False, fontsize=40) 
plt.savefig('playground/Plots/EUROC_step2.png', dpi=300, transparent=True)


# --- IMAGE 3: Add No DRT Resolution ---
plt.step(time_none, risk_none, where='post', label='No DRT resolution', 
         color=color3, linewidth=lineWidth) 

# Values inverted: 82.4% survival -> 17.6% risk
plt.text(370, 17.7, '17.6%', color=color3, verticalalignment='bottom', fontweight='bold', fontsize=fontLabel)
plt.text(710, 17.7, '17.6%', color=color3, verticalalignment='bottom', fontweight='bold', fontsize=fontLabel)

plt.legend(loc='upper left', frameon=False, fontsize=40) 
plt.savefig('playground/Plots/EUROC_step3.png', dpi=300, transparent=True)