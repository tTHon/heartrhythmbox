import pandas as pd
import numpy as np

# Load your CSV
df = pd.read_csv("playground/Micra/LPBaseline.csv")

# Show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Convert T2FU from days to months
df['T2FU_months'] = df['T2FU'] / 30.44  # 30.44 is the average number of days in a month


def summarize_continuous(series):
    return f"{series.mean():.1f} ± {series.std():.1f} ({series.min():.1f}-{series.max():.1f})"

def summarize_categorical(series):
    counts = series.value_counts(dropna=False)
    total = len(series)
    return "; ".join([f"{idx}: {val} ({val/total*100:.1f}%)" for idx, val in counts.items()])

#continuous_vars = ["Age", "Weight", "Height", "BSAYu","CCI","T2FU","CKD"]
continuous_vars = ["BMI"]
#categorical_vars = ["Sex", "MI","PCI/CABG","CHF","PAD","CVA","Dementia","COPD","CNT","PU","Liver","DM","CKD","Malignancy","TV","SigValve","AF","HTN","CCISev","Antiplatelet","OAC","Access","Position","Model","Hemostasis"] 

#categorical_vars = ["IndicationPPM","IndicationforMicra","AcuteCom","ChronicCom",'Death'] 
categorical_vars = ["CKD"]
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
