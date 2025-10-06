import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Set a visually appealing plot style
plt.style.use('seaborn-v0_8-whitegrid')

# --- 1. Data ---
plot_data = {
    'display_label': [
        'TVP (vs. LP)',
        'CCI (per 1 unit increase)'
    ],
    'HR_unadj':       [1.09, 1.37],
    'Lower_CI_unadj': [0.27, 0.91],
    'Upper_CI_unadj': [4.38, 2.08],
    'p_unadj':        [0.91, 0.14],
    'HR_adj':         [3.76, 1.74],
    'Lower_CI_adj':   [0.45, 0.95],
    'Upper_CI_adj':   [31.36, 3.16],
    'p_adj':          [0.22, 0.07]
}
df = pd.DataFrame(plot_data)

#--- Colors ---
colors = sns.color_palette("Set2", 2)
color1 = colors[0]
color2 = colors[1]

# --- 2. Create the Plot ---
fig, ax = plt.subplots(figsize=(16, 12))
y_positions = np.arange(len(df))+1
offset = 0.15
ax.axvline(x=1, color='gray', linestyle='--', linewidth=2)
ax.set_facecolor('whitesmoke')

# --- 3. Plot Data with Capping Logic ---
#X_LIMIT = 100
#ax.set_xlim(0.2, X_LIMIT)

# Unadjusted HRs (plotted higher on the inverted y-axis)
unadj_err = [df['HR_unadj'] - df['Lower_CI_unadj'], df['Upper_CI_unadj'] - df['HR_unadj']]
ax.errorbar(x=df['HR_unadj'], y=y_positions - offset, xerr=unadj_err, fmt='o', color=color1, elinewidth=7, markersize=12,markeredgecolor='#66B0A5', capsize=8, capthick=2, label='Unadjusted HR', zorder=5)

# Adjusted HRs (plotted lower on the inverted y-axis)
adj_hr_values = df['HR_adj'].copy()
adj_lower_ci = df['Lower_CI_adj'].copy()
adj_upper_ci = df['Upper_CI_adj'].copy()
tvp_index = df[df['display_label'] == 'TVP (vs. LP)'].index[0]
tvp_y_pos = y_positions[tvp_index] 
tvp_upper_ci_original = adj_upper_ci[tvp_index]

#if pd.notna(tvp_upper_ci_original) and tvp_upper_ci_original > X_LIMIT:
#    adj_upper_ci[tvp_index] = X_LIMIT
#    ax.annotate('', xy=(X_LIMIT + 0.9, tvp_y_pos), xytext=(X_LIMIT, tvp_y_pos),
 #               arrowprops=dict(arrowstyle="->, head_width=0.4, head_length=0.8", color=color2, lw=3),
  #              va='center')

adj_err = [adj_hr_values - adj_lower_ci, adj_upper_ci - adj_hr_values]
ax.errorbar(x=adj_hr_values, y=y_positions + offset, xerr=adj_err, fmt='s', color=color2, markersize=12, markeredgecolor='#FC8262', capsize=8, capthick=2, elinewidth=7, label='Adjusted* HR', zorder=5)


# --- 4. Add Row Shading and Text Annotations ---
# NEW: Add alternating row colors for readability
for i in range(len(df)):
    if i % 2 == 1: # Apply to every other row
        ax.axhspan(i - 0.4, i + 0.4, color='whitesmoke', zorder=0)

y_text_offset = 0.05

for i, row in df.iterrows():
    # Unadjusted HR annotation
    unadj_text = f"HR: {row['HR_unadj']:.2f} ({row['Lower_CI_unadj']:.2f}-{row['Upper_CI_unadj']:.2f}); $P$ = {row['p_unadj']:.2f}"
    # NEW: Conditional alignment to prevent text from going off-plot
    ha_unadj = 'left' if row['HR_unadj'] < 2 else 'center'
    ax.text(x=row['HR_unadj']-0.05, y=y_positions[i] - offset - y_text_offset, s=unadj_text,
            ha=ha_unadj, va='bottom', fontsize=24, color='black')

    # Adjusted HR annotation
    adj_text = f"HR: {row['HR_adj']:.2f} ({row['Lower_CI_adj']:.2f}-{row['Upper_CI_adj']:.2f}); $P$ = {row['p_adj']:.2f}"
    ha_adj = 'left' if row['HR_adj'] < 2 else 'center'
    ax.text(x=row['HR_adj'] - 0.05, y=y_positions[i] + offset - y_text_offset, s=adj_text,
            ha=ha_adj, va='bottom', fontsize=24, color='black')

# --- 5. Finalize Plot Aesthetics ---
ax.set_xlabel('HR (95% Confidence Interval)', fontsize=22, labelpad=15)
ax.set_title('Forest Plot of Unadjusted and Adjusted* Hazard Ratios for Major Complications', fontsize=26, weight='bold',pad=20)
ax.tick_params(labelsize=22, width=2, length=4, color = 'darkgray')
ax.set_yticks(y_positions)
ax.set_yticklabels(df['display_label'])
ax.invert_yaxis()
ax.set_xscale('log')
ax.set_xticks([0.1,1,10])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.legend(loc='upper right', frameon=True, edgecolor='darkgray',
          fontsize=20,facecolor='none',labelspacing=1.2,
          handleheight=1,handlelength=2,borderaxespad=0.5)

# NEW: Adjust layout to give more space on the left
plt.subplots_adjust(left=0.2, right=0.98, top=0.9, bottom=0.15)
plt.savefig('playground/Micra/forest.png', dpi=300, bbox_inches='tight')
#plt.show()