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

# Filter for the matched cohort (lowBSA LP=1 patients and their TV=0 controls)
lp_lowbsa_match_ids = df[(df['Type'] == 1) & (df['lowBSA'] == 1)]['MatchID'].unique()
df_matched = df[df['MatchID'].isin(lp_lowbsa_match_ids)].copy()

# Note: We no longer need to create a 'group_numeric' column
# The code now assumes 'Type' is already coded as 1=LP and 0=TV.
print(f"Filtered to matched cohort. Records: {df_matched.shape[0]}, Matched Sets: {len(lp_lowbsa_match_ids)}")


# --- 2. Define Variables for Table 1 ---

continuous_vars = ['Age', 'BSA', 'CCI', 'Weight', 'Height']
categorical_vars = {
    'Sex': {'F': 0, 'M': 1},
    'CCISev': {0: 0, 1: 1},
    'CHF': {0: 0, 1: 1},
    'HTN': {0: 0, 1: 1}
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
        # Conditional logistic regression using the numeric 'Type' column
        model = sm.Logit(df_matched['Type'], df_matched[var], groups=df_matched['MatchID'])
        result = model.fit(disp=0)
        p_value = result.pvalues[var]
    except Exception as e:
        print(f"Could not calculate p-value for '{var}'. Reason: {e}")

    results.append({
        'Characteristic': var,
        f'LP (n={n_lp})': f'{lp_stats.mean():.2f} ({lp_stats.std():.2f})',
        f'TV (n={n_tv})': f'{tv_stats.mean():.2f} ({tv_stats.std():.2f})',
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
table1_df = pd.DataFrame(results).set_index('Characteristic')

print("\n" + "="*60)
print("      Table 1: Baseline Characteristics of Matched Cohort")
print("="*60)
print(table1_df.to_string())
print("="*60)

# --- 5. Save Table to a File ---
#try:
    #save_csv = input("\nSave table to a CSV file? (y/n): ").lower().strip()
    #if save_csv == 'y':
        #output_filename = 'Table_1_Baseline_Characteristics.csv'
        #table1_df.to_csv(output_filename)
        #print(f"Table successfully saved as '{output_filename}'")
#except Exception as e:
    #print(f"Could not save file. Reason: {e}")