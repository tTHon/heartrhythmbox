import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import numpy as np

# 1. STYLE CONSTANTS
BG_COLOR = 'none'      # Light Grey Background
LINE_COLOR = '#5a005a'    # Deep Purple for lines/text
MARKER_FILL = '#f8dc00'   # Bright Yellow for markers
MARKER_EDGE = '#5a005a'   # Deep Purple border for markers
TEXT_COLOR = '#1a1a1a'    # Dark text (almost black)
AXIS_COLOR = '#2a2a4a'    # Dark Blue/Purple for axis lines
GRID_COLOR = '#d0d0d0'    # Light grey for grid

# Font Setup
try:
    font_path = fm.findfont("Inter")
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Roboto'
except:
    plt.rcParams['font.family'] = 'sans-serif'

# 2. DATA PREPARATION
studies = [
    #{"name": "CAPITAL-RCT", "n": "801", "hr": 0.75, "lo": 0.47, "hi": 1.16, "size": 80},
    #{"name": "REDUCE-AMI", "n": "5,020", "hr": 0.96, "lo": 0.79, "hi": 1.16, "size": 180},
    #{"name": "REBOOT", "n": "8,438", "hr": 1.04, "lo": 0.89, "hi": 1.22, "size": 250},
    #{"name": "BETAMI-DANBLOCK", "n": "5,574", "hr": 0.85, "lo": 0.75, "hi": 0.98, "size": 200},
    {"name": "AßYSS", "n": "3,698", "hr": 0.86, "lo": 0.75, "hi": 0.99, "size": 140},
]

summary = {
    "name": "+ AßYSS", "n": "23,531", 
    "hr": 0.91, "lo": 0.84, "hi": 0.98,
    "p_fixed": "0.009", "p_random": "0.035",
    "heterogeneity": "I²≈26% (low-moderate)"
}

sensitivity = {
    "name": "ALL 4 RCTs", "n": "19,833", 
    "hr": 0.93, "lo": 0.82, "hi": 1.04,
    "note": "Attenuated benefit, increased heterogeneity"
}

# 3. PLOT SETUP
fig, ax = plt.subplots(figsize=(22, 11))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# Y-positions
y_studies = np.arange(len(studies))
summary_y = len(studies) + 1.4
sensitivity_y = summary_y - 1

# Set plot limits
ax.set_ylim(summary_y + 1.5, -1) 
# CHANGE 1: Set X limit start to 0.6
ax.set_xlim(0.6, 1.4)

# 4. DRAWING REFERENCE ELEMENTS
ax.axvline(x=1, color=AXIS_COLOR, linewidth=2.5, zorder=1)

# Vertical Grid lines
for x in [0.6, 0.8, 1.0, 1.2, 1.4]:
    if x != 1.0:
        ax.axvline(x=x, color=GRID_COLOR, linewidth=2.0, linestyle='--', zorder=0)

# 5. PLOTTING INDIVIDUAL STUDIES
for i, study in enumerate(studies):
    ax.plot([study['lo'], study['hi']], [i, i], color=LINE_COLOR, linewidth=4.0, zorder=2)
    
    ax.plot(study['hr'], i, marker='o', markersize=25, 
            markerfacecolor=MARKER_FILL, markeredgecolor=MARKER_EDGE, markeredgewidth=4.5, zorder=3)
    
    # CHANGE 2: Labels aligned to 0.6 (using 0.58 for padding)
    ax.text(0.1, i, f"{study['name']} (N={study['n']})", 
            fontweight='normal', fontsize=34, color=AXIS_COLOR, va='center', ha='right', transform=ax.get_yaxis_transform())
    
    ax.text(1.3, i, f"{study['hr']:.2f} ({study['lo']:.2f}-{study['hi']:.2f})", 
            fontsize=34, color=TEXT_COLOR, va='center', ha='center')

# 6. PLOTTING SUMMARIES (DIAMONDS)
def draw_diamond(data, y_pos):
    diamond_height = 0.5
    diamond_coords = [
        (data['lo'], y_pos),       
        (data['hr'], y_pos - diamond_height/2), 
        (data['hi'], y_pos),       
        (data['hr'], y_pos + diamond_height/2)  
    ]
    diamond = patches.Polygon(diamond_coords, closed=True, 
                              facecolor=MARKER_FILL, edgecolor=MARKER_EDGE, linewidth=3.5, zorder=3)
    ax.add_patch(diamond)

draw_diamond(summary, summary_y)
draw_diamond(sensitivity, sensitivity_y)

# Summary Text 
ax.text(0.2, summary_y, f"{summary['name']} (N={summary['n']})", 
        fontweight='bold', fontsize=32, color=LINE_COLOR, va='center', ha='right', transform=ax.get_yaxis_transform())

ax.text(1.1, summary_y - 0.1, f"HR {summary['hr']}, 95% CI: {summary['lo']}-{summary['hi']}", 
        fontsize=35, color=TEXT_COLOR, fontweight='bold', va='center', ha='left')
ax.text(1.1, summary_y + 0.3, f"p={summary['p_fixed']} (fixed); p={summary['p_random']} (random)", 
        fontsize=34, color=TEXT_COLOR, va='center', ha='left')
ax.text(1.1, summary_y + 0.6, f"Heterogeneity: {summary['heterogeneity']}", 
        fontsize=34, color=TEXT_COLOR, ha='left', va='center', style='italic')

# Sensitivity Text (Right aligned to 0.6 line)
ax.text(0.2, sensitivity_y, f"{sensitivity['name']} (N={sensitivity['n']})", 
        fontweight='bold', fontsize=35, color=LINE_COLOR, va='center', ha='right', transform=ax.get_yaxis_transform())

ax.text(1.1, sensitivity_y, f"HR {sensitivity['hr']}, 95% CI: {sensitivity['lo']}-{sensitivity['hi']}", 
        fontsize=33, color=TEXT_COLOR, fontweight='bold', va='center', ha='left')
#ax.text(1.45, sensitivity_y + 0.5, f"({sensitivity['note']})", 
        #fontsize=10, color=TEXT_COLOR, ha='right', va='center', style='italic')

# 7. AXIS LABELS AND TITLES
ax.set_title("Global effect of β-blockers after MI on trials' PRIMARY ENDPOINTS", 
             fontsize=40, fontweight='bold', color=AXIS_COLOR, pad=45, ha ='center')

ax.text(1.3, -0.8, "Hazard ratio (95% CI)", fontsize=34, color=AXIS_COLOR, ha='center')
ax.set_xlabel("Hazard ratio (95% CI)", fontsize=30, color=AXIS_COLOR)
ax.tick_params(axis='x', colors=AXIS_COLOR, labelsize=30, direction='out', length=8, width=3.0)

ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_color(AXIS_COLOR)
ax.spines['bottom'].set_linewidth(2)
ax.yaxis.set_ticks([]) 

plt.text(0.85, summary_y + 1.3, "β-blockers better", fontsize=30, fontweight='bold', color=AXIS_COLOR, ha='right')
plt.text(1.15, summary_y + 1.3, "β-blockers worse", fontsize=30, fontweight='bold', color=AXIS_COLOR, ha='left')

plt.tight_layout()
#plt.show()

# Save command to demonstrate transparency works
plt.savefig('playground/Plots/BBMeta.png', dpi=300, transparent=True)
