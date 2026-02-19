import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load data
df = pd.read_csv('playground/dDimer/dDimer.csv')

# dDimer log10 transformation
df['log10_dDimer'] = np.log10(df['dDimer'])

# Variables for the plot in the order they were provided (not sorted by OR)
vars_to_plot = [
    ('age', 'Age (per year)'),
    ('Gender', 'Gender (2 vs 1)'),
    ('log10_dDimer', 'log10(d-Dimer)'),
    ('PW', 'LAA Flow (PW)'),
    ('CHADVA', 'CHA2DS2-VASc'),
    ('HASBLED', 'HASBLED'),
    ('GFR', 'GFR'),
    ('DM', 'Diabetes (yes/no)'),
    ('LAVI', 'LAVI'),
    ('LVEF', 'LVEF')
]

results = []

for var_col, var_label in vars_to_plot:
    temp_df = df[[var_col, 'Thrombus']].dropna()
    X = sm.add_constant(temp_df[var_col])
    y = temp_df['Thrombus']
    
    try:
        model = sm.Logit(y, X).fit(disp=0)
        params = model.params
        conf = model.conf_int()
        pvalues = model.pvalues
        
        or_val = np.exp(params.iloc[1])
        ci_low = np.exp(conf.iloc[1, 0])
        ci_high = np.exp(conf.iloc[1, 1])
        p_val = pvalues.iloc[1]
        
        results.append({
            'Variable': var_label,
            'OR': or_val,
            'Lower CI': ci_low,
            'Upper CI': ci_high,
            'P-value': p_val
        })
    except Exception as e:
        print(f"Error processing {var_label}: {e}")

# Create DataFrame. We reverse it so the first variable in the list is at the top.
forest_df = pd.DataFrame(results)[::-1].reset_index(drop=True)

# Styles
TEXT_COLOR = '#0F2043'
BAR_COLOR = '#2E0245'
MARKER_FILL = '#FFD700'

plt.rcParams.update({
    'font.size': 28,
    'font.family': 'sans-serif',
    'font.weight': 'normal',
    'text.color': TEXT_COLOR,
    'axes.labelcolor': TEXT_COLOR,
    'xtick.color': TEXT_COLOR,
    'ytick.color': TEXT_COLOR,
    'axes.edgecolor': TEXT_COLOR,
    'figure.facecolor': 'none',
    'axes.facecolor': 'none',
    'savefig.facecolor': 'none'
})

fig, ax = plt.subplots(figsize=(26, 18))
y_pos = np.arange(len(forest_df))

ax.errorbar(forest_df['OR'], y_pos, 
             xerr=[forest_df['OR'] - forest_df['Lower CI'], forest_df['Upper CI'] - forest_df['OR']],
             fmt='o', 
             ecolor=BAR_COLOR,
             markerfacecolor=MARKER_FILL,
             markeredgecolor=BAR_COLOR,
             markersize=22, 
             capsize=12, 
             elinewidth=5,
             markeredgewidth=4)

ax.axvline(x=1, color=TEXT_COLOR, linestyle='--', linewidth=3, alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(forest_df['Variable'])
ax.set_xlabel('Odds Ratio (Log Scale)')
ax.set_xscale('log')
ax.set_xlim(0.03, 25)

ticks = [0.1, 0.5, 1, 2, 5, 10, 25]   
ax.set_xticks(ticks)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

# Adding labels to the right: OR, 95% CI and P-value
for i, row in forest_df.iterrows():
    # Construct label text
    label_text = f"{row['OR']:.2f} ({row['Lower CI']:.2f}-{row['Upper CI']:.2f}), p={row['P-value']:.3f}"
    
    # Place text to the right of the plot area
    # We use a fixed x position in the data coordinates or figure coordinates?
    # Data coordinates are hard because of the log scale and varying CI widths.
    # Better to use axes coordinates for consistent alignment.
    ax.text(1.02, i, label_text, va='center', ha='left', transform=ax.get_yaxis_transform(), fontsize=24)

ax.set_title('Predictors for Thrombus: Univariate Odds Ratios', pad=40, fontweight='bold')
ax.grid(axis='x', linestyle=':', alpha=0.4, color=TEXT_COLOR)

# Adjust plot boundaries to make room for labels on the right
plt.subplots_adjust(right=0.7)
ax.set_xlim(0.05, 25)

plt.savefig('playground/dDimer/forest_plot.png', transparent=True, facecolor='#efffff', bbox_inches='tight')