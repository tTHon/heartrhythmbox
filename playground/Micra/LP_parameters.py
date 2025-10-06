import pandas as pd
import numpy as np

# Load your CSV
df = pd.read_csv("playground/Micra/LP_parameters.csv")

# Show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def summarize_continuous(series):
    return f"Mean: {series.mean():.1f} ± {series.std():.1f} ({series.min():.1f}-{series.max():.1f}), Median: {series.median():.1f} (IQR: {series.quantile(0.25):.1f}-{series.quantile(0.75):.1f})"

def summarize_categorical(series):
    counts = series.value_counts(dropna=False)
    total = len(series)
    return "; ".join([f"{idx}: {val} ({val/total*100:.1f}%)" for idx, val in counts.items()])

categorical_vars = [""]
continuous_vars = ["Threshold0","Imp0","Threshold6","Imp02","VP0","VP6"]
# Overall
summary_all = {}
for var in continuous_vars:
    if var in df.columns:
        summary_all[var] = summarize_continuous(df[var].dropna())
for var in categorical_vars:
    if var in df.columns:
        summary_all[var] = summarize_categorical(df[var].dropna())

num_patients = len(df)
summary_all_df = pd.DataFrame.from_dict(summary_all, orient="index", columns=["All Patients"+f' (N={num_patients})'])

# Final table
baseline_table = summary_all_df
#print(baseline_table)

# Print the table in a markdown format
print(baseline_table.to_markdown())
