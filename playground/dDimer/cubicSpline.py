import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from patsy import dmatrix
import matplotlib.ticker as ticker

# Load data
df = pd.read_csv('playground/dDimer/dDimer.csv')
data = df[['dDimer', 'Thrombus']].dropna()

# Filter non-positive values for log transformation
data = data[data['dDimer'] > 0]

# Log-transform
data['log_dDimer'] = np.log(data['dDimer'])

# Fit model
# Using cubic spline on log_dDimer with 4 degrees of freedom
model = smf.logit(formula='Thrombus ~ cr(log_dDimer, df=4)', data=data).fit(disp=0)

# Reference value (Median)
ref_val_orig = data['dDimer'].median()
ref_val_log = np.log(ref_val_orig)

# Define function to calculate OR and CI
def calculate_or_ci(dDimer_values_orig):
    # Log transform input
    dDimer_values_log = np.log(dDimer_values_orig)
    
    # Create prediction DataFrame
    pred_df = pd.DataFrame({'log_dDimer': dDimer_values_log})
    ref_df = pd.DataFrame({'log_dDimer': [ref_val_log]})
    
    # Design matrices
    design_info = model.model.data.design_info
    X_pred = dmatrix(design_info.builder, pred_df)
    X_ref = dmatrix(design_info.builder, ref_df)
    
    # Difference
    X_diff = X_pred - X_ref
    
    # Calculate
    params = model.params.values
    cov_params = model.cov_params().values
    
    log_or = X_diff @ params
    var_log_or = np.sum((X_diff @ cov_params) * X_diff, axis=1)
    se_log_or = np.sqrt(var_log_or)
    
    or_val = np.exp(log_or)
    ci_low = np.exp(log_or - 1.96 * se_log_or)
    ci_up = np.exp(log_or + 1.96 * se_log_or)
    
    return or_val, ci_low, ci_up

# --- Generate Plot Data ---
x_range_log = np.linspace(data['log_dDimer'].min(), data['log_dDimer'].max(), 500)
x_range_orig = np.exp(x_range_log)
or_plot, ci_low_plot, ci_up_plot = calculate_or_ci(x_range_orig)

# --- Plotting ---
plt.figure(figsize=(10, 6))

# Plot curve and CI
plt.plot(x_range_orig, or_plot, color='#005a32', lw=2, label='Odds Ratio')
plt.fill_between(x_range_orig, ci_low_plot, ci_up_plot, color='#41ae76', alpha=0.2, label='95% CI')

# Reference lines
plt.axhline(y=1, color='gray', linestyle='--', linewidth=1)
plt.axvline(x=ref_val_orig, color='gray', linestyle=':', linewidth=1, label=f'Ref (Median: {ref_val_orig:.0f})')

# Rug plot for data density
plt.plot(data.loc[data.Thrombus==0, 'dDimer'], [or_plot.max()*0.01]*len(data[data.Thrombus==0]), '|', color='blue', alpha=0.5, label='No Thrombus')
plt.plot(data.loc[data.Thrombus==1, 'dDimer'], [or_plot.max()*0.02]*len(data[data.Thrombus==1]), '|', color='red', alpha=0.5, label='Thrombus')

# Axes Log Scale
plt.xscale('log')
plt.yscale('log')

# Format ticks to be numerical strings (not scientific notation)
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))

plt.title('Cubic Spline: Odds Ratio of Thrombus vs d-Dimer')
plt.xlabel('d-Dimer (Log Scale)')
plt.ylabel('Odds Ratio (Log Scale)')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()

plt.tight_layout()
#plt.savefig('cubic_spline_or_numerical.png')
plt.show()