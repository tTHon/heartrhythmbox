import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data from the user's text
data = {
    'Variable': [
        'Diabetes Mellitus',
        'Higher BMI (per 5kg/m2)',
        'Higher Corrected Serum Calcium (per 1mg/dL)',
        'Higher Serum Phosphorous (per 1mg/dL)',
        'Higher PTH (per 100pg/mL)',
        'Nutritional Vitamin D Use',
        'Cinacalcet Use',
        'Warfarin Use',
        'Higher Hemoglobin (per 1g/dL)',
        'Use of Erythropoiesis-Stimulating Agents'
    ],
    'OR': [2.16, 1.38, 1.33, 1.11, 1.12, 2.11, 2.12, 3.22, 0.81, 0.69],
    'CI_Lower': [1.69, 1.29, 1.12, 1.03, 1.03, 1.41, 1.10, 2.11, 0.91, 0.51],
    'CI_Upper': [2.59, 1.51, 1.61, 1.19, 1.21, 2.95, 3.15, 4.65, 0.99, 0.99]
}

df = pd.DataFrame(data)

# Calculate log odds and their confidence intervals
df['log_OR'] = np.log(df['OR'])
df['log_CI_Lower'] = np.log(df['CI_Lower'])
df['log_CI_Upper'] = np.log(df['CI_Upper'])

# Sort the dataframe by OR for better visualization
df_sorted = df.sort_values(by='OR', ascending=False).reset_index(drop=True)

# Set font to Inter and larger font sizes, black background, and adjust colors
plt.rcParams['font.family'] = 'Inter'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.facecolor'] = '#222222'
plt.rcParams['figure.facecolor'] = '#222222'
plt.rcParams['savefig.facecolor'] = '#222222'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['xtick.color'] = '#cccccc'
plt.rcParams['ytick.color'] = '#cccccc'
plt.rcParams['text.color'] = '#ffffff'
plt.rcParams['axes.labelcolor'] = '#ffffff'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 20

fig, ax = plt.subplots(figsize=(12, 9))

# Plot the confidence intervals (grey)
for i, row in df_sorted.iterrows():
    ax.hlines(y=i, xmin=row['CI_Lower'], xmax=row['CI_Upper'], color='#bbbbbb', linestyle='-', linewidth=4)

# Plot the odds ratios (red)
ax.plot(df_sorted['OR'], range(len(df_sorted)), 'o', color='#FE4D51', markersize=14, markeredgecolor='#ffffff', markeredgewidth=2)

# Add a vertical line at OR = 1 (grey)
ax.axvline(x=1.0, color='#888888', linestyle='--', linewidth=2)

# Set y-axis labels and ticks
ax.set_yticks(range(len(df_sorted)))
ax.set_yticklabels(df_sorted['Variable'], color='#ffffff')

# Set x-axis labels (Odds Ratios) and use a log scale
ax.set_xlabel('Odds Ratio (OR)', fontsize=18, color='#ffffff')
ax.set_xscale('log')

# Add annotations for OR and CI (white)
for i, row in df_sorted.iterrows():
    or_text = f"OR: {row['OR']:.2f} (95% CI: {row['CI_Lower']:.2f}-{row['CI_Upper']:.2f})"
    ax.text(x=row['CI_Upper'] + 0.15, y=i, s=or_text, va='center', color='#ffffff', fontsize=14)

# Title
ax.set_title('Forest Plot of Odds Ratios and 95% Confidence Intervals for CUA Development', color='#FE4D51', pad=20)

plt.tight_layout()
plt.savefig('playground/Plots/forest_plot.png')

print("Forest plot saved as forest_plot.png")