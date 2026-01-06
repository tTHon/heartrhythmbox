import pandas as pd
import numpy as np
from scipy import stats

def compare_thrombus_groups(csv_file):
    # Load data
    df = pd.read_csv('playground/dDimer/dDimer.csv')
    
    # Define variables
    group_col = 'Thrombus'
    categorical_cols = [
        'Gender', 'procedure', 'typeAF', 'OAC', 'HTN', 
        'IHD', 'CVA', 'HF', 'DM', 'Shape', 'Complication'
    ]
    continuous_cols = [
        'age', 'dDimer', 'CHADVA', 'HASBLED', 'BMI', 
        'creatinine', 'GFR', 'LAVI', 'LAdiameter', 'LVEF', 'PW'
    ]

    # Split dataset
    group0 = df[df[group_col] == 0]
    group1 = df[df[group_col] == 1]
    
    results = []
    
    # --- 1. Analyze Continuous Variables ---
    for col in continuous_cols:
        if col not in df.columns: continue
            
        # Calculate Mean ± SD
        mean0, std0 = group0[col].mean(), group0[col].std()
        mean1, std1 = group1[col].mean(), group1[col].std()
        
        # Mann-Whitney U Test (Non-parametric, safer for small/skewed data)
        # Drop NaNs before testing
        g0_clean = group0[col].dropna()
        g1_clean = group1[col].dropna()
        
        if len(g0_clean) > 0 and len(g1_clean) > 0:
            stat, p_val = stats.mannwhitneyu(g0_clean, g1_clean)
        else:
            p_val = np.nan

        results.append({
            'Variable': col,
            'Category': '(Mean ± SD)',
            'Thrombus=0 (n={})'.format(len(group0)): f"{mean0:.2f} ± {std0:.2f}",
            'Thrombus=1 (n={})'.format(len(group1)): f"{mean1:.2f} ± {std1:.2f}",
            'P-value': p_val
        })

    # --- 2. Analyze Categorical Variables ---
    for col in categorical_cols:
        if col not in df.columns: continue
            
        # Create contingency table
        contingency = pd.crosstab(df[col], df[group_col])
        
        # Statistical Test
        # Use Fisher's Exact for 2x2, Chi-Square for larger tables
        if contingency.shape == (2, 2):
            stat, p_val = stats.fisher_exact(contingency)
        else:
            try:
                stat, p_val, dof, expected = stats.chi2_contingency(contingency)
            except:
                p_val = np.nan

        # Calculate counts and percentages for each category
        categories = sorted(df[col].unique())
        for i, cat in enumerate(categories):
            n0 = len(group0[group0[col] == cat])
            p0 = (n0 / len(group0)) * 100
            
            n1 = len(group1[group1[col] == cat])
            p1 = (n1 / len(group1)) * 100
            
            results.append({
                'Variable': col if i == 0 else "", # Only show variable name once
                'Category': str(cat),
                'Thrombus=0 (n={})'.format(len(group0)): f"{n0} ({p0:.1f}%)",
                'Thrombus=1 (n={})'.format(len(group1)): f"{n1} ({p1:.1f}%)",
                'P-value': p_val if i == 0 else "" # Only show p-value once
            })

    # Create DataFrame and format P-values
    results_df = pd.DataFrame(results)
    
    # Helper to format p-values nicely
    def clean_p(val):
        if isinstance(val, str) or pd.isna(val): return ""
        if val < 0.001: return "<0.001"
        return f"{val:.3f}"
        
    results_df['P-value'] = results_df['P-value'].apply(clean_p)
    
    return results_df

# Run the function
comparison_table = compare_thrombus_groups('dDimer.csv')
print(comparison_table.to_string())