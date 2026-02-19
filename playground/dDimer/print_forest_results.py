import pandas as pd
import numpy as np
import statsmodels.api as sm

predictors = {
    'age': 'Age',
    'Gender': 'Gender (Female)',
    'Log_dDimer': 'dDimer (Log)',
    'CHADVA': 'CHA2DS2-VASc',
    'HASBLED': 'HAS-BLED',
    'LAVI': 'LAVI',
    'LAdiameter': 'LA Diameter',
    'LVEF': 'LVEF',
    'PW': 'PW Velocity'
}

df = pd.read_csv('playground/dDimer/dDimer.csv')
df['Log_dDimer'] = np.log(df['dDimer'])

results = []
for var, label in predictors.items():
    if var not in df.columns:
        print(f"{var} not in df, skipping")
        continue
    temp_df = df[[var, 'Thrombus']].dropna()
    X = temp_df[var]
    X.name = var
    y = temp_df['Thrombus']
    X = sm.add_constant(X)
    try:
        model = sm.Logit(y, X).fit(disp=0)
        params = model.params
        conf = model.conf_int()
        p_val = model.pvalues[var]
        or_val = np.exp(params[var])
        lower_ci = np.exp(conf.loc[var][0])
        upper_ci = np.exp(conf.loc[var][1])
        print(f"{label}: OR={or_val:.3f} 95%CI=({lower_ci:.3f}-{upper_ci:.3f}) p={p_val:.4f}")
    except Exception as e:
        print(f"{label}: model failed: {e}")
