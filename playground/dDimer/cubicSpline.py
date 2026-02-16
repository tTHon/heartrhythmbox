import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from patsy import dmatrix
import matplotlib.ticker as ticker

# Load data
df = pd.read_csv('playground/dDimer/dDimer.csv')
data = df[['dDimer', 'Thrombus']].dropna()
data = data[data['dDimer'] > 0]
data['log_dDimer'] = np.log(data['dDimer'])

# Fit Spline Model
model = smf.logit(formula='Thrombus ~ cr(log_dDimer, df=4)', data=data).fit(disp=0)

# Reference value (Median)
ref_val_orig = data['dDimer'].median()
ref_val_log = np.log(ref_val_orig)

# Helper function for OR/CI
def get_or_ci(dDimer_orig):
    dDimer_log = np.log(dDimer_orig)
    pred_df = pd.DataFrame({'log_dDimer': dDimer_log})
    ref_df = pd.DataFrame({'log_dDimer': [ref_val_log]})
    
    design_info = model.model.data.design_info
    X_pred = dmatrix(design_info.builder, pred_df)
    X_ref = dmatrix(design_info.builder, ref_df)
    X_diff = X_pred - X_ref
    
    params = model.params.values
    cov = model.cov_params().values
    
    log_or = X_diff @ params
    var_log_or = np.sum((X_diff @ cov) * X_diff, axis=1)
    se_log_or = np.sqrt(var_log_or)
    
    return np.exp(log_or), np.exp(log_or - 1.96*se_log_or), np.exp(log_or + 1.96*se_log_or)

# Plot Data Generation
x_range_log = np.linspace(data['log_dDimer'].min(), data['log_dDimer'].max(), 500)
x_range_orig = np.exp(x_range_log)
or_vals, ci_low, ci_high = get_or_ci(x_range_orig)

# Define Markers (Points of Interest)
marker_points = [250, 500, 1000, 2000, 4000]
marker_points = [p for p in marker_points if p <= data['dDimer'].max() and p >= data['dDimer'].min()]
# Ensure median is included
if ref_val_orig not in marker_points:
    marker_points.append(ref_val_orig)
marker_points.sort()

# Calculate OR for markers
marker_ors = []
for p in marker_points:
    val, _, _ = get_or_ci([p])
    marker_ors.append(val[0])

# Plotting
plt.figure(figsize=(10, 6))

# Main Curve
plt.plot(x_range_orig, or_vals, color='#005a32', lw=2, label='Odds Ratio')
plt.fill_between(x_range_orig, ci_low, ci_high, color='#41ae76', alpha=0.2, label='95% CI')

# Reference Lines
plt.axhline(1, color='gray', ls='--', lw=1)
plt.axvline(ref_val_orig, color='gray', ls=':', lw=1, label=f'Median ({ref_val_orig:.0f})')

# Add Markers and Annotations
plt.scatter(marker_points, marker_ors, color='black', s=40, zorder=5) # Black dots
for x, y in zip(marker_points, marker_ors):
    plt.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, fontweight='bold')

# Rug Plot (Large)
y_min_limit = 0.08
plt.ylim(y_min_limit, 200) # Ensure space at bottom
rug_y_pos_0 = y_min_limit * 1.2
rug_y_pos_1 = y_min_limit * 1.5

plt.plot(data.loc[data.Thrombus==0, 'dDimer'], [rug_y_pos_0]*len(data[data.Thrombus==0]), '|', color='blue', alpha=0.6, markersize=15, markeredgewidth=1.5, label='No Thrombus')
plt.plot(data.loc[data.Thrombus==1, 'dDimer'], [rug_y_pos_1]*len(data[data.Thrombus==1]), '|', color='red', alpha=0.6, markersize=15, markeredgewidth=1.5, label='Thrombus')

# Formatting
plt.xscale('log')
plt.yscale('log')
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))

plt.title('Cubic Spline Plot: Odds Ratio of Thrombus vs d-Dimer')
plt.xlabel('d-Dimer (Log Scale)')
plt.ylabel('Odds Ratio (Log Scale)')
plt.grid(True, which='both', alpha=0.2)
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig('playground/dDimer/splinePlot.png')
#plt.show()