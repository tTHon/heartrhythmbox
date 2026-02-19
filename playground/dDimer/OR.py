import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv('playground/dDimer/dDimer.csv')

# dDimer log10 for OR calculation
df['log10_dDimer'] = np.log10(df['dDimer'])

# Variables of interest (the ones used in the forest plot)
vars_info = [
    ('age', 'Age (years)', 'continuous'),
    ('Gender', 'Gender (1=Male, 2=Female)', 'categorical'),
    ('dDimer', 'd-Dimer (ng/mL)', 'continuous'),
    ('log10_dDimer', 'log10(d-Dimer)', 'continuous'),
    ('PW', 'LAA Flow Velocity (PW)', 'continuous'),
    ('CHADVA', 'CHA2DS2-VASc Score', 'continuous'),
    ('HASBLED', 'HASBLED Score', 'continuous'),
    ('GFR', 'GFR (mL/min/1.73m2)', 'continuous'),
    ('DM', 'Diabetes Mellitus', 'categorical'),
    ('LAVI', 'LAVI (mL/m2)', 'continuous'),
    ('LVEF', 'LVEF (%)', 'continuous')
]

table_data = []

for col, label, var_type in vars_info:
    temp_df = df[[col, 'Thrombus']].dropna()
    group0 = temp_df[temp_df['Thrombus'] == 0][col]
    group1 = temp_df[temp_df['Thrombus'] == 1][col]
    
    # Descriptive Stats
    if var_type == 'continuous':
        desc0 = f"{group0.median():.1f} ({group0.quantile(0.25):.1f}-{group0.quantile(0.75):.1f})"
        desc1 = f"{group1.median():.1f} ({group1.quantile(0.25):.1f}-{group1.quantile(0.75):.1f})"
    else:
        # For categorical, show N (%)
        count0 = group0.value_counts().to_dict()
        total0 = len(group0)
        desc0 = ", ".join([f"{k}: {v} ({v/total0*100:.1f}%)" for k, v in count0.items()])
        
        count1 = group1.value_counts().to_dict()
        total1 = len(group1)
        desc1 = ", ".join([f"{k}: {v} ({v/total1*100:.1f}%)" for k, v in count1.items()])

    # Univariate Logistic Regression for OR and P-value
    try:
        X = sm.add_constant(temp_df[col])
        y = temp_df['Thrombus']
        model = sm.Logit(y, X).fit(disp=0)
        or_val = np.exp(model.params.iloc[1])
        ci_low = np.exp(model.conf_int().iloc[1, 0])
        ci_high = np.exp(model.conf_int().iloc[1, 1])
        p_val = model.pvalues.iloc[1]
        or_str = f"{or_val:.3f} ({ci_low:.3f}-{ci_high:.3f})"
    except:
        or_str = "N/A"
        p_val = np.nan

    table_data.append({
        'Variable': label,
        'No Thrombus (N=146)': desc0,
        'Thrombus (N=8)': desc1,
        'Odds Ratio (95% CI)': or_str,
        'P-value': f"{p_val:.3f}" if not np.isnan(p_val) else "N/A"
    })

summary_table = pd.DataFrame(table_data)
summary_table.to_csv('playground/dDimer/ORsummary.csv', index=False)
print(summary_table.to_string())