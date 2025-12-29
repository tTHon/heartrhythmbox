import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Load your CSV
df = pd.read_csv("playground/Micra/LPBaseline.csv")

# Show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Convert T2FU from days to months
df['T2FU_months'] = df['T2FU'] / 30.44  # 30.44 is the average number of days in a month
# Convert T2FU from days to years
df['T2FU_years'] = df['T2FU'] / 365.25  # 365.25 accounts for leap years

def summarize_continuous(series):
    return f"{series.mean():.2f} ± {series.std():.2f} ({series.min():.1f}-{series.max():.1f})"

def summarize_categorical(series):
    counts = series.value_counts(dropna=False)
    total = len(series)
    return "; ".join([f"{idx}: {val} ({val/total*100:.1f}%)" for idx, val in counts.items()])

def summarize_nonParametric(series):
    return f"{series.median():.2f} ({series.quantile(0.25):.2f}-{series.quantile(0.75):.2f}) ({series.min():.2f}-{series.max():.2f})"

continuous_vars = ["Age", "Weight", "Height", "BSA","CCI","BMI","T2FU_years","CKD"]
#continuous_vars = []
#categorical_vars = ["Sex", "MI","PCI/CABG","CKD","CHF","PAD","CVA","Dementia","COPD","CNT","PU","Liver","DM","CKD","Malignancy","TV","SigValve","AF","HTN","CCISev","Antiplatelet","OAC","Access","Position","Model","Hemostasis"] 

categorical_vars = ["AcuteCom","ChronicCom",'Death'] 
#categorical_vars = ["CKD"]

nonParametric_vars = ["Age", "BMI", "Weight", "Height","BSA","CCI"]
# Overall
summary_all = {}
for var in continuous_vars:
    if var in df.columns:
        summary_all[var] = summarize_continuous(df[var].dropna())
for var in categorical_vars:
    if var in df.columns:
        summary_all[var] = summarize_categorical(df[var].dropna())
for var in nonParametric_vars:
    if var in df.columns:
        summary_all[var] = summarize_nonParametric(df[var].dropna())

num_patients = len(df)
summary_all_df = pd.DataFrame.from_dict(summary_all, orient="index", columns=["All Patients"+f' (N={num_patients})'])

# Final table
baseline_table = summary_all_df
#print(baseline_table)

# Print the table in a markdown format
print(baseline_table.to_markdown())

# --- Shapiro-Wilk normality tests ---
def run_shapiro_on_vars(var_list):
    results = []
    for var in var_list:
        if var in df.columns:
            data = df[var].dropna()
            n = len(data)
            if n < 3:
                results.append({'variable': var, 'n': n, 'W': np.nan, 'p_value': np.nan, 'normal': False, 'note': 'n<3'})
                continue
            # scipy.stats.shapiro requires n <= 5000; if larger, sample 5000 randomly
            sample = data if n <= 5000 else data.sample(5000, random_state=0)
            try:
                W, p = stats.shapiro(sample)
            except Exception as e:
                W, p = np.nan, np.nan
            normal = False if np.isnan(p) else (p > 0.05)
            results.append({'variable': var, 'n': n, 'W': W, 'p_value': p, 'normal': normal, 'note': ''})
    return pd.DataFrame(results)

shapiro_vars = continuous_vars + nonParametric_vars
shapiro_df = run_shapiro_on_vars(shapiro_vars)

# Print the tables in markdown format
print(baseline_table.to_markdown())
print("\nShapiro-Wilk normality test results:")
print(shapiro_df.to_markdown(index=False, floatfmt=".4f"))

# --- QQ plots for variables tested with Shapiro-Wilk ---
vars_to_plot = list(dict.fromkeys(shapiro_vars))  # preserve order, remove duplicates
vars_to_plot = [v for v in vars_to_plot if v in df.columns]

if vars_to_plot:
    n = len(vars_to_plot)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes_flat = axes.ravel() if n > 1 else [axes]

    for i, var in enumerate(vars_to_plot):
        data = df[var].dropna()
        ax = axes_flat[i]
        if len(data) < 3:
            ax.text(0.5, 0.5, f"{var}\n(n < 3)", ha='center', va='center')
            ax.set_axis_off()
            continue

        # QQ plot against normal distribution
        stats.probplot(data, dist="norm", plot=ax)

        # add Shapiro result to title if available
        row = shapiro_df[shapiro_df['variable'] == var]
        if not row.empty:
            W = row['W'].values[0]
            p = row['p_value'].values[0]
            ax.set_title(f"{var} — Shapiro W={W:.3f}, p={p:.3f}")
        else:
            ax.set_title(var)

    # hide any unused subplots
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    out_path = "playground/Micra/qq_plots.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()