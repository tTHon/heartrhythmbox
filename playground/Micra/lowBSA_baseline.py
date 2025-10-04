import pandas as pd
import statsmodels.api as sm
import numpy as np
import warnings

# Suppress warnings from the statsmodels library for a cleaner output
warnings.filterwarnings('ignore')

# --- 1. Load and Prepare Data ---

# Load the dataset from your local file
file_path = 'playground/Micra/lowBSA.csv'
try:
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"ERROR: The file '{file_path}' was not found. Please make sure it's in the correct directory.")
    exit()

# calculate follow-up time for the whole cohort
df['T2FU_year'] = df['T2FU'] / 365.25
print(f"Mean F/U time (year): {df['T2FU_year'].mean():.2f} ({df['T2FU_year'].std():.2f}) ({df['T2FU_year'].min():.2f}-{df['T2FU_year'].max():.2f})")
print(f"Medien F/U time (year): {df['T2FU_year'].median():.1f} ({df['T2FU_year'].quantile(0.25):.1f})-({df['T2FU_year'].quantile(0.75):.1f}")


# Filter for the matched cohort (lowBSA LP=1 patients and their TV=0 controls)
lp_lowbsa_match_ids = df[(df['Type'] == 1)]['MatchID'].unique()
df_matched = df[df['MatchID'].isin(lp_lowbsa_match_ids)].copy()

# Note: We no longer need to create a 'group_numeric' column
# The code now assumes 'Type' is already coded as 1=LP and 0=TV.
print(f"Filtered to matched cohort. Records: {df_matched.shape[0]}, Matched Sets: {len(lp_lowbsa_match_ids)}")


# --- 2. Define Variables for Table 1 ---

continuous_vars = ['Age', 'BSA', 'CCI', 'Weight', 'Height','CKDStage','BMI','T2FU']
categorical_vars = {
    'Sex': {'F': 0, 'M': 1},
    'CCISev': {0: 0, 1: 1},
    'CKDStage': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1},
    'CHF': {0: 0, 1: 1},
    'PAD': {0: 0, 1: 1},
    'HTN': {0: 0, 1: 1},
    'CVA': {"No": 0, "Stroke/TIA": 1,"Hemiplegia": 1},
    'Dementia': {0: 0, 1: 1},
    'COPD': {0: 0, 1: 1},
    'CNT': {0: 0, 1: 1},
    'PU': {0: 0, 1: 1},
    'Liver': {'No': 0, 'Mild': 1,'Mod/Sev': 1},
    'DM ': {'No': 0, 'Mild': 1,"EndOrgan": 1},
    'Malignancy': {'No': 0, 'Solid w/o Met': 1,'Solid w Met': 1},
    'TV_Disease': {'None/Mild': 0,'Mod/Sev': 1},
    'AF': {0: 0, 1: 1},
    'Complication': {0: 0, 1: 1},
    'Death': {0: 0, 1: 1},
    'IndicationforPPM' : {'AVB': 0, 'SSS': 1},
    'AnyEvents': {0: 0, 1: 1}
}

# --- 3. Perform Analysis and Build Table ---

results = []
# Get group sizes directly from the 'Type' column
n_lp = df_matched[df_matched['Type'] == 1].shape[0]
n_tv = df_matched[df_matched['Type'] == 0].shape[0]

print("\nProcessing variables to generate Table 1...")

# --- Analysis for Continuous Variables ---
for var in continuous_vars:
    lp_stats = df_matched[df_matched['Type'] == 1][var]
    tv_stats = df_matched[df_matched['Type'] == 0][var]
    
    p_value = np.nan
    try:
        # Add a constant (intercept) to the model
        X = sm.add_constant(df_matched[var])
        # Conditional logistic regression using the numeric 'Type' column
        #model = sm.Logit(df_matched['Type'], df_matched[var], groups=df_matched['MatchID'])
        model = sm.Logit(df_matched['Type'], X, groups=df_matched['MatchID'])
        result = model.fit(disp=0)
        # The p-value is in the second position (index 1)
        p_value = result.pvalues[1]
        #p_value = result.pvalues[var]
    except Exception as e:
        print(f"Could not calculate p-value for '{var}'. Reason: {e}")

    results.append({
        'Characteristic': var,
        f'LP (n={n_lp})': f'{lp_stats.mean():.2f} ({lp_stats.std():.2f}) ({lp_stats.min():.2f}-{lp_stats.max():.2f})',
        f'TV (n={n_tv})': f'{tv_stats.mean():.2f} ({tv_stats.std():.2f}) ({tv_stats.min():.2f}-{tv_stats.max():.2f})',
        'p-value': f'{p_value:.3f}' if pd.notna(p_value) else 'N/A'
    })

# --- Analysis for Categorical Variables ---
for var, mapping in categorical_vars.items():
    df_matched[var + '_numeric'] = pd.to_numeric(df_matched[var].replace(mapping), errors='coerce').fillna(0)
    
    lp_stats = df_matched[df_matched['Type'] == 1][var + '_numeric']
    tv_stats = df_matched[df_matched['Type'] == 0][var + '_numeric']

    p_value = np.nan
    try:
        model = sm.Logit(df_matched['Type'], df_matched[var + '_numeric'], groups=df_matched['MatchID'])
        result = model.fit(disp=0)
        p_value = result.pvalues[var + '_numeric']
    except Exception as e:
        print(f"Could not calculate p-value for '{var}'. Reason: {e}")

    lp_sum = int(lp_stats.sum())
    tv_sum = int(tv_stats.sum())
    lp_display = f'{lp_sum} ({lp_sum / n_lp * 100:.2f}%)' if n_lp > 0 else '0 (0.0%)'
    tv_display = f'{tv_sum} ({tv_sum / n_tv * 100:.2f}%)' if n_tv > 0 else '0 (0.0%)'
    char_name = f'{var} (Yes)' if var not in ['Sex'] else 'Sex (Male)'
    
    results.append({
        'Characteristic': char_name,
        f'LP (n={n_lp})': lp_display,
        f'TV (n={n_tv})': tv_display,
        'p-value': f'{p_value:.3f}' if pd.notna(p_value) else 'N/A'
    })

# --- 4. Format and Display Final Table ---
from tabulate import tabulate
table1_df = pd.DataFrame(results).set_index('Characteristic')

#print("\n" + "="*60)
#print("      Table 1: Baseline Characteristics of Matched Cohort")
#print("="*60)
#print(table1_df.to_string())
#print("="*60)

# REPLACE your old print statement with this one
print(tabulate(table1_df, headers='keys', tablefmt='pretty'))

# --- 5. Save Table to a File ---
#try:
    #save_csv = input("\nSave table to a CSV file? (y/n): ").lower().strip()
    #if save_csv == 'y':
        #output_filename = 'Table_1_Baseline_Characteristics.csv'
        #table1_df.to_csv(output_filename)
        #print(f"Table successfully saved as '{output_filename}'")
#except Exception as e:
    #print(f"Could not save file. Reason: {e}")