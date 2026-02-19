import pandas as pd
import numpy as np
import statsmodels.api as sm

if __name__ == '__main__':
    df = pd.read_csv('playground/dDimer/dDimer.csv')
    print('Thrombus counts:')
    print(df['Thrombus'].value_counts(dropna=False))
    print('\nMean age by Thrombus:')
    print(df.groupby('Thrombus')['age'].mean())

    temp = df[['age','Thrombus']].dropna()
    X = sm.add_constant(temp['age'])
    y = temp['Thrombus']
    model = sm.Logit(y, X).fit(disp=0)
    params = model.params
    conf = model.conf_int()
    or_val = np.exp(params['age'])
    lower = np.exp(conf.loc['age'][0])
    upper = np.exp(conf.loc['age'][1])
    p = model.pvalues['age']

    print('\nLogit (age per year)')
    print(f'OR = {or_val:.3f}')
    print(f'95% CI = ({lower:.3f}, {upper:.3f})')
    print(f'p = {p:.4f}')
