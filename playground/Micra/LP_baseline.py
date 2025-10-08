import pandas as pd

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

#continuous_vars = ["Age", "Weight", "Height", "BSA","CCI","BMI","T2FU_years","CKD"]
continuous_vars = []
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
