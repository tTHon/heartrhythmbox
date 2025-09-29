import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set a visually appealing plot style
plt.style.use('seaborn-v0_8-whitegrid')

# --- 1. Data ---
plot_data = {
    'display_label': [
        'TVP (vs. LP)',
        'CCI (per 1 unit increase)'
    ],
    'HR_unadj':       [1.18, 1.35],
    'Lower_CI_unadj': [0.29, 0.90],
    'Upper_CI_unadj': [4.79, 2.02],
    'p_unadj':        [0.816, 0.143],
    'HR_adj':         [5.89, 1.85],
    'Lower_CI_adj':   [0.48, 0.97],
    'Upper_CI_adj':   [72.67, 3.52],
    'p_adj':          [0.166, 0.060]
}
df = pd.DataFrame(plot_data)

# --- 2. Create the Plot ---
fig, ax = plt.subplots(figsize=(14, 5))
y_positions = np.arange(len(df))+1
offset = 0.15
ax.axvline(x=1, color='black', linestyle='--', linewidth=1)

# --- 3. Plot Data with Capping Logic ---
X_LIMIT = 100
ax.set_xlim(0.2, X_LIMIT)

# Unadjusted HRs (plotted higher on the inverted y-axis)
unadj_err = [df['HR_unadj'] - df['Lower_CI_unadj'], df['Upper_CI_unadj'] - df['HR_unadj']]
ax.errorbar(x=df['HR_unadj'], y=y_positions - offset, xerr=unadj_err, fmt='o', color='#007acc', capsize=5, label='Unadjusted HR', zorder=5)

# Adjusted HRs (plotted lower on the inverted y-axis)
adj_hr_values = df['HR_adj'].copy()
adj_lower_ci = df['Lower_CI_adj'].copy()
adj_upper_ci = df['Upper_CI_adj'].copy()
tvp_index = df[df['display_label'] == 'TVP (vs. LP)'].index[0]
tvp_y_pos = y_positions[tvp_index] + offset # Corrected y-position for the arrow
tvp_upper_ci_original = adj_upper_ci[tvp_index]

if pd.notna(tvp_upper_ci_original) and tvp_upper_ci_original > X_LIMIT:
    adj_upper_ci[tvp_index] = X_LIMIT
    ax.annotate('', xy=(X_LIMIT + 0.9, tvp_y_pos), xytext=(X_LIMIT, tvp_y_pos),
                arrowprops=dict(arrowstyle="->, head_width=0.4, head_length=0.8", color='#33a02c', lw=3),
                va='center')

adj_err = [adj_hr_values - adj_lower_ci, adj_upper_ci - adj_hr_values]
ax.errorbar(x=adj_hr_values, y=y_positions + offset, xerr=adj_err, fmt='s', color='#33a02c', capsize=5, label='Adjusted HR', zorder=5)


# --- 4. Add Row Shading and Text Annotations ---
# NEW: Add alternating row colors for readability
for i in range(len(df)):
    if i % 2 == 1: # Apply to every other row
        ax.axhspan(i - 0.4, i + 0.4, color='whitesmoke', zorder=0)

y_text_offset = 0.05

for i, row in df.iterrows():
    # Unadjusted HR annotation
    unadj_text = f"HR: {row['HR_unadj']:.2f} ({row['Lower_CI_unadj']:.2f}-{row['Upper_CI_unadj']:.2f}); p-value: {row['p_unadj']:.3f}"
    # NEW: Conditional alignment to prevent text from going off-plot
    ha_unadj = 'left' if row['HR_unadj'] < 2 else 'center'
    ax.text(x=row['HR_unadj'], y=y_positions[i] - offset - y_text_offset, s=unadj_text,
            ha=ha_unadj, va='bottom', fontsize=12, color='black')

    # Adjusted HR annotation
    adj_text = f"HR: {row['HR_adj']:.2f} ({row['Lower_CI_adj']:.2f}-{row['Upper_CI_adj']:.2f}); p-value: {row['p_adj']:.3f}"
    ha_adj = 'left' if row['HR_adj'] < 2 else 'center'
    ax.text(x=row['HR_adj'], y=y_positions[i] + offset - y_text_offset, s=adj_text,
            ha=ha_adj, va='bottom', fontsize=12, color='black')

# --- 5. Finalize Plot Aesthetics ---
ax.set_xlabel('HR (95% Confidence Interval)', fontsize=13)
ax.set_title('Forest Plot of Unadjusted and Adjusted Hazard Ratios for Complications', fontsize=15, weight='bold',pad=5)
ax.set_yticks(y_positions)
ax.set_yticklabels(df['display_label'], fontsize=12)
ax.invert_yaxis()
ax.set_xscale('log')
ax.set_xticks([0.2, 0.5, 1, 2, 5, 10, 100])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.legend(loc='upper left', frameon=True, edgecolor='black')

# NEW: Adjust layout to give more space on the left
plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.15)
#plt.savefig('final_forest_plot_revised.png', dpi=300, bbox_inches='tight')
plt.show()