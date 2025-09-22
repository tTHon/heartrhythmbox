import pandas as pd
from scipy import stats
import numpy as np

# Load your CSV file
# NOTE: Make sure your file path is correct.
try:
    df = pd.read_csv("playground/Micra/lowBSA.csv")
except FileNotFoundError:
    print("Error: The file 'LPBaseline.csv' was not found. Please check the file path.")
    exit()

def compare_continuous_vars(df, group_var, lp_label, tv_label, continuous_vars):
    """
    Compares continuous variables between two groups using a t-test.
    """
    results = []
    
    lp_group = df[df[group_var] == lp_label].dropna(subset=continuous_vars)
    tv_group = df[df[group_var] == tv_label].dropna(subset=continuous_vars)
    
    for var in continuous_vars:
        # Check if the variable exists and has data
        if var in df.columns and len(lp_group[var]) > 1 and len(tv_group[var]) > 1:
            lp_mean = lp_group[var].mean()
            lp_std = lp_group[var].std()
            tv_mean = tv_group[var].mean()
            tv_std = tv_group[var].std()
            
            # Perform independent samples t-test
            t_stat, p_val = stats.ttest_ind(lp_group[var], tv_group[var], equal_var=False)
            
            results.append({
                'Variable': var,
                f'{lp_label} (N={len(lp_group)})': f"{lp_mean:.1f} \u00B1 {lp_std:.1f}",
                f'{tv_label} (N={len(tv_group)})': f"{tv_mean:.1f} \u00B1 {tv_std:.1f}",
                'P-value': p_val
            })
    return results

def compare_categorical_vars(df, group_var, lp_label, tv_label, categorical_vars):
    """
    Compares categorical variables between two groups using a Chi-square test.
    """
    results = []
    
    lp_group = df[df[group_var] == lp_label]
    tv_group = df[df[group_var] == tv_label]
    
    for var in categorical_vars:
        if var in df.columns:
            contingency_table = pd.crosstab(df[group_var], df[var])
            
            # Chi-square test
            chi2, p_val, _, _ = stats.chi2_contingency(contingency_table)

            # Format the counts and percentages for display
            lp_counts = contingency_table.loc[lp_label]
            tv_counts = contingency_table.loc[tv_label]
            
            lp_str = "; ".join([f"{idx}: {val} ({val/lp_counts.sum()*100:.1f}%)" for idx, val in lp_counts.items()])
            tv_str = "; ".join([f"{idx}: {val} ({val/tv_counts.sum()*100:.1f}%)" for idx, val in tv_counts.items()])
            
            results.append({
                'Variable': var,
                f'{lp_label} (N={len(lp_group)})': lp_str,
                f'{tv_label} (N={len(tv_group)})': tv_str,
                'P-value': p_val
            })
    return results

def compare_ordinal_vars(df, group_var, lp_label, tv_label, ordinal_vars):
    """
    Compares ordinal variables between two groups using a Mann-Whitney U test.
    """
    results = []
    
    lp_group = df[df[group_var] == lp_label]
    tv_group = df[df[group_var] == tv_label]
    
    for var in ordinal_vars:
        if var in df.columns:
            # Drop NaN to avoid errors in the test
            lp_data = lp_group[var].dropna()
            tv_data = tv_group[var].dropna()
            
            # Perform Mann-Whitney U test
            u_stat, p_val = stats.mannwhitneyu(lp_data, tv_data, alternative='two-sided')
            
            # Calculate Median and IQR for display
            lp_median = np.median(lp_data)
            lp_q1, lp_q3 = np.percentile(lp_data, [25, 75])
            
            tv_median = np.median(tv_data)
            tv_q1, tv_q3 = np.percentile(tv_data, [25, 75])
            
            results.append({
                'Variable': var,
                f'{lp_label} (N={len(lp_data)})': f"{lp_median} ({lp_q1}-{lp_q3})",
                f'{tv_label} (N={len(tv_data)})': f"{tv_median} ({tv_q1}-{tv_q3})",
                'P-value': p_val
            })
    return results

# --- Main Analysis ---
group_variable = "Type"
lp_label = "LP"
tv_label = "TV"

continuous_vars = ["Age", "CCI", "BSAYu"]
categorical_vars = ["Sex", "CCISev","TV","AF", "MajCom30D", "Death","composite"]
ordinal_vars = ["CKD"] # CKD moved here as it's an ordinal variable

# Perform comparisons for all variable types
continuous_results = compare_continuous_vars(df, group_variable, lp_label, tv_label, continuous_vars)
categorical_results = compare_categorical_vars(df, group_variable, lp_label, tv_label, categorical_vars)
ordinal_results = compare_ordinal_vars(df, group_variable, lp_label, tv_label, ordinal_vars)

# Combine results into a single DataFrame
results_df = pd.DataFrame(continuous_results + categorical_results + ordinal_results)

# Format P-value column
results_df['P-value'] = results_df['P-value'].apply(lambda x: f"{x:.4f}" if x >= 0.0001 else "<0.0001")

# Print the final table
print("Comparison of LP vs TV Pacemaker Groups")
print(results_df.to_markdown(index=False))