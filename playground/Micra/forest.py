import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set a visually appealing plot style
plt.style.use('seaborn-v0_8-whitegrid')

# --- 1. Data with New, Explicit Labels and P-Values ---
plot_data = {
    'display_label': [
        'TVP (vs. LP)',
        'CCI (per 1 unit increase)'
    ],
    'HR_unadj':       [1.18, 1.35],
    'Lower_CI_unadj': [0.29, 0.90],
    'Upper_CI_unadj': [4.79, 2.02],
    'p_unadj':        [0.816, 0.143], # Unadjusted p-values
    'HR_adj':         [5.89, 1.85],
    'Lower_CI_adj':   [0.48, 0.97],
    'Upper_CI_adj':   [72.67, 3.52],
    'p_adj':          [0.166, 0.060]  # Adjusted p-values
}
df = pd.DataFrame(plot_data)

# --- 2. Create the Plot ---
# Increased figure width to make space for the p-value column
fig, ax = plt.subplots(figsize=(14, 5))
y_positions = np.arange(len(df))
offset = 0.15
ax.axvline(x=1, color='black', linestyle='--', linewidth=1, label='HR = 1 (No Effect)')

# --- 3. Plot Data with Capping Logic ---
X_LIMIT = 10
ax.set_xlim(0.1, X_LIMIT)

# Unadjusted HRs
unadj_err = [df['HR_unadj'] - df['Lower_CI_unadj'], df['Upper_CI_unadj'] - df['HR_unadj']]
ax.errorbar(x=df['HR_unadj'], y=y_positions + offset, xerr=unadj_err, fmt='o', color='#007acc', capsize=5, label='Unadjusted HR')

# Adjusted HRs
adj_hr_values = df['HR_adj'].copy()
adj_lower_ci = df['Lower_CI_adj'].copy()
adj_upper_ci = df['Upper_CI_adj'].copy()
tvp_index = df[df['display_label'] == 'TVP (vs. LP)'].index[0]
tvp_y_pos = y_positions[tvp_index] - offset
tvp_upper_ci_original = adj_upper_ci[tvp_index]

if pd.notna(tvp_upper_ci_original) and tvp_upper_ci_original > X_LIMIT:
    adj_upper_ci[tvp_index] = X_LIMIT
    ax.annotate('', xy=(X_LIMIT + 0.9, tvp_y_pos), xytext=(X_LIMIT, tvp_y_pos),
                arrowprops=dict(arrowstyle="->, head_width=0.4, head_length=0.8", color='#33a02c', lw=3),
                va='center')

adj_err = [adj_hr_values - adj_lower_ci, adj_upper_ci - adj_hr_values]
ax.errorbar(x=adj_hr_values, y=y_positions - offset, xerr=adj_err, fmt='s', color='#33a02c', capsize=5, label='Adjusted HR')


# --- 4. Add Text Annotations for HR, CI, and P-Value ---
# Define x-coordinates for the text columns
hr_ci_x_pos = X_LIMIT + 1.2
p_value_x_pos = X_LIMIT + 6.5

for i, row in df.iterrows():
    # Unadjusted HR/CI and p-value
    hr_ci_text_unadj = f"{row['HR_unadj']:.2f} ({row['Lower_CI_unadj']:.2f} - {row['Upper_CI_unadj']:.2f})"
    p_text_unadj = f"{row['p_unadj']:.3f}"
    ax.text(hr_ci_x_pos, y_positions[i] + offset, hr_ci_text_unadj, va='center', ha='left', fontsize=9)
    ax.text(p_value_x_pos, y_positions[i] + offset, p_text_unadj, va='center', ha='left', fontsize=9)
    
    # Adjusted HR/CI and p-value
    original_upper_ci = row['Upper_CI_adj']
    hr_ci_text_adj = f"{row['HR_adj']:.2f} ({row['Lower_CI_adj']:.2f} - {original_upper_ci:.2f})"
    p_text_adj = f"{row['p_adj']:.3f}"
    ax.text(hr_ci_x_pos, y_positions[i] - offset, hr_ci_text_adj, va='center', ha='left', fontsize=9)
    ax.text(p_value_x_pos, y_positions[i] - offset, p_text_adj, va='center', ha='left', fontsize=9)

# Add column headers for the text annotations
header_y_pos = len(df)
ax.text(hr_ci_x_pos, header_y_pos, "HR (95% CI)", va='center', ha='left', fontsize=9, weight='bold')
ax.text(p_value_x_pos, header_y_pos, "p-value", va='center', ha='left', fontsize=9, weight='bold')


# --- 5. Finalize Plot Aesthetics ---
ax.set_xlabel('Hazard Ratio (HR) and 95% Confidence Interval', fontsize=12)
ax.set_title('Forest Plot of Unadjusted and Adjusted Hazard Ratios', fontsize=14, weight='bold')
ax.set_yticks(y_positions)
ax.set_yticklabels(df['display_label'])
ax.invert_yaxis()
ax.set_xscale('log')
ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.legend(loc='upper left', frameon=True, edgecolor='black')

fig.text(0.5, 0.01, "An arrow indicates the 95% CI extends to the value shown.", ha='center', fontsize=9, style='italic', color='gray')

# Adjust layout to make space for the new p-value column
plt.subplots_adjust(right=0.8, bottom=0.15)
plt.savefig('final_forest_plot_with_pvalues.png', dpi=300, bbox_inches='tight')
plt.show()