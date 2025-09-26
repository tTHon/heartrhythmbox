import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
file_path = 'playground/Micra/lowBSA.csv'
try:
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"ERROR: The file '{file_path}' was not found.")
    exit()

# Filter for matched cohort
lp_lowbsa_match_ids = df[df['Type'] == 1]['MatchID'].unique()
df_matched = df[df['MatchID'].isin(lp_lowbsa_match_ids)].copy()
print(f"Matched cohort: {df_matched.shape[0]} records, {len(lp_lowbsa_match_ids)} matched sets")

# ============================================================================
# PART 1: PREPARE DATA FOR ANALYSIS
# ============================================================================

# Step 1: Create a clear multi-level status column for competing risks
# 0 = Censored, 1 = Complication (event), 2 = Death (competing risk)
outcome = np.zeros(len(df_matched))
outcome[df_matched['Complication'] == 1] = 1
outcome[(df_matched['Death'] == 1) & (df_matched['Complication'] == 0)] = 2
df_matched['status'] = outcome.astype(int)

# Step 2: Create the specific event column for the cause-specific model
df_matched['is_complication'] = (df_matched['status'] == 1).astype(int)

# Use T2Events for duration
df_matched['duration'] = df_matched['T2Events']

# Create numeric group variable for the model (TVP=1, LP=0)
df_matched['group_tvp'] = (df_matched['Type'] == 0).astype(int)

print("\n--- Data prepared for Adjusted Cox Model ---")
# Display the necessary columns, including CCI
print(df_matched[['MatchID', 'Type', 'duration', 'is_complication', 'CCI']].head())


# ============================================================================
# PART 2: RUN THE ADJUSTED STRATIFIED CAUSE-SPECIFIC COX MODEL
# ============================================================================

print("\n" + "="*80)
print("     ADJUSTED STRATIFIED CAUSE-SPECIFIC MODEL (Adjusting for CCI)")
print("="*80)

# Initialize the Cox model
cph = CoxPHFitter()

try:
    # Fit the model, stratifying by MatchID and adjusting for group and CCI
    cph.fit(df_matched,
            duration_col='duration',
            event_col='is_complication',
            strata=['MatchID'],
            formula="group_tvp + CCI") # CCI is added as a covariate

    print("\n--- Adjusted Stratified Cox Model Results ---")
    cph.print_summary()

    # Extract key results for a clean summary
    # Note: We are interested in the HR for the group variable
    hr_adj = cph.summary.loc['group_tvp', 'exp(coef)']
    p_value_adj = cph.summary.loc['group_tvp', 'p']
    ci_lower_adj = cph.summary.loc['group_tvp', 'exp(coef) lower 95%']
    ci_upper_adj = cph.summary.loc['group_tvp', 'exp(coef) upper 95%']

    print("\n--- Summary of Findings ---")
    print(f"Adjusted Hazard Ratio (aHR) for TVP vs LP (adjusting for CCI):")
    print(f"aHR = {hr_adj:.2f} (95% CI: {ci_lower_adj:.2f}-{ci_upper_adj:.2f})")
    print(f"P-value: {p_value_adj:.3f}")

except Exception as e:
    print(f"\nCould not fit the model. Error: {e}")