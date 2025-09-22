import pandas as pd
import io
import statsmodels.api as sm
from patsy import dmatrix
import numpy as np
import matplotlib.pyplot as plt

# Data provided by the user
data_string = """BSAYu	MajCom30D
1.299620507	0
1.438589086	0
1.644521324	0
1.5125397	0
0	0
0	0
0	0
1.61070964	0
1.850317794	0
1.301083221	0
1.588833028	0
0	0
1.444881139	0
0	0
1.27896688	0
1.936312361	0
1.346015819	0
1.333570188	0
1.460852948	0
1.958192291	0
1.623647274	0
1.308372259	0
1.802707186	0
1.302349577	0
1.423636215	1
1.485385114	0
1.372232402	0
1.612126064	0
1.649218096	0
1.603608754	0
1.346015819	0
1.86056894	0
1.520816656	0
1.280750242	1
1.26601246	0
1.654134587	0
1.388108359	0
1.249884255	0
1.419917152	0
1.623647274	0
1.284901793	0
1.664219646	1
1.392868634	0
1.711029905	0
1.352219123	1
1.2582161	0
1.531037815	0
1.764946994	0
1.712215239	0
1.356899715	0
1.359233967	0
1.553728401	0
1.49686529	0
1.540697456	0
1.146157554	0
1.498558577	0
1.787576817	0
1.289826694	0
1.470973553	0
"""
df = pd.read_csv(io.StringIO(data_string), sep='\t')

# Remove rows where 'BSAYu' is 0, as they are likely placeholders and not informative for the model
df_filtered = df[df['BSAYu'] > 0]

# Create a design matrix with a cubic spline for BSAYu
# The degrees of freedom (df) can be tuned. df=4 is a reasonable starting point.
X = dmatrix("bs(BSAYu, df=4, degree=3, include_intercept=False)", data=df_filtered, return_type='dataframe')
y = df_filtered['MajCom30D']

# Fit a logistic regression model
model = sm.GLM(y, X, family=sm.families.Binomial()).fit()

# Create a sequence of BSAYu values to plot the spline curve
x_vals = np.linspace(df_filtered['BSAYu'].min(), df_filtered['BSAYu'].max(), 100)

# Create a design matrix for the new BSAYu values
X_plot = dmatrix("bs(x_vals, df=4, degree=3, include_intercept=False)", return_type='dataframe')

# Predict the probabilities for these new values
y_pred = model.predict(X_plot)

# Calculate the odds ratio for a unit change at a representative value, e.g., the mean
# This is for explanation purposes, as the OR is not constant
mean_bsayu = df_filtered['BSAYu'].mean()
x_mean = dmatrix(f"bs({mean_bsayu}, df=4, degree=3, include_intercept=False)", return_type='dataframe')
x_mean_plus_1 = dmatrix(f"bs({mean_bsayu + 1}, df=4, degree=3, include_intercept=False)", return_type='dataframe')

log_odds_mean = np.dot(x_mean, model.params)
log_odds_mean_plus_1 = np.dot(x_mean_plus_1, model.params)

# Calculate odds ratio at mean
or_at_mean = np.exp(log_odds_mean_plus_1 - log_odds_mean)[0]

# Plotting the cubic spline
fig, ax = plt.subplots(figsize=(10, 6))

# Jitter the data points for better visibility since MajCom30D is binary
jittered_y = df_filtered['MajCom30D'] + np.random.uniform(-0.02, 0.02, size=len(df_filtered))
ax.scatter(df_filtered['BSAYu'], jittered_y, alpha=0.6, label='Observed Data (with jitter)')

# Plot the predicted probability curve
ax.plot(x_vals, y_pred, color='red', linewidth=2, label='Fitted Cubic Spline Curve')

ax.set_title('Cubic Spline Logistic Regression for MajCom30D vs. BSAYu')
ax.set_xlabel('BSAYu')
ax.set_ylabel('Probability of MajCom30D = 1')
ax.legend()
plt.grid(True)
plt.savefig('cubic_spline_plot.png')
plt.show()

# Calculate OR and 95% CI for each BSAYu value in x_vals
# OR is calculated as the change in odds for a 1-unit increase in BSAYu
# For each x, compare x and x+1
or_vals = []
lower_ci = []
upper_ci = []
for x in x_vals:
    X1 = dmatrix(f"bs({x}, df=4, degree=3, include_intercept=False)", return_type='dataframe')
    X2 = dmatrix(f"bs({x+1}, df=4, degree=3, include_intercept=False)", return_type='dataframe')
    log_odds1 = np.dot(X1, model.params)
    log_odds2 = np.dot(X2, model.params)
    or_val = np.exp(log_odds2 - log_odds1)[0]
    # Delta method for CI
    diff = X2.values - X1.values
    se = np.sqrt(np.dot(np.dot(diff, model.cov_params()), diff.T))[0][0]
    ci_low = np.exp((log_odds2 - log_odds1) - 1.96 * se)[0]
    ci_up = np.exp((log_odds2 - log_odds1) + 1.96 * se)[0]
    or_vals.append(or_val)
    lower_ci.append(ci_low)
    upper_ci.append(ci_up)

# Plot OR and 95% CI
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(x_vals, or_vals, color='blue', label='Odds Ratio (OR)')
ax2.fill_between(x_vals, lower_ci, upper_ci, color='blue', alpha=0.2, label='95% CI')
ax2.set_title('Odds Ratio (OR) for MajCom30D by BSAYu (Cubic Spline)')
ax2.set_xlabel('BSAYu')
ax2.set_ylabel('Odds Ratio for 1-unit increase in BSAYu')
ax2.axhline(1, color='grey', linestyle='--')
ax2.legend()
plt.grid(True)
plt.savefig('cubic_spline_OR_CI_plot.png')
plt.show()