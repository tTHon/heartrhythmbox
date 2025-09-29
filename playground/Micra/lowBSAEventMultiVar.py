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
#df_matched['is_complication'] = (df_matched['status'] == 1).astype(int)
#df_matched['is_death'] = (df_matched['status'] == 2).astype(int)

# Use T2Events for duration
df_matched['duration'] = df_matched['T2Events']

# Create numeric group variable for the model (TVP=1, LP=0)
df_matched['group_tvp'] = (df_matched['Type'] == 0).astype(int)

print("\n--- Data prepared for Analysis ---")
print(df_matched[['MatchID', 'Type', 'duration', 'is_complication', 'CCI']].head())


# ============================================================================
# PART 2A: UNADJUSTED MODEL FOR DEVICE GROUP (group_tvp)
# ============================================================================

print("\n" + "="*80)
print("       UNADJUSTED STRATIFIED MODEL (for Device Group)")
print("="*80)

cph_unadj_group = CoxPHFitter()
cph_unadj_group.fit(df_matched,
                    duration_col='duration',
                    event_col='is_complication',
                    strata=['MatchID'],
                    formula="group_tvp")

hr_unadj = cph_unadj_group.summary.loc['group_tvp', 'exp(coef)']
p_value_unadj = cph_unadj_group.summary.loc['group_tvp', 'p']
ci_lower_unadj = cph_unadj_group.summary.loc['group_tvp', 'exp(coef) lower 95%']
ci_upper_unadj = cph_unadj_group.summary.loc['group_tvp', 'exp(coef) upper 95%']

print("\n--- Summary of Unadjusted Findings for Device Group ---")
print(f"Unadjusted Hazard Ratio (uHR) for TVP vs LP:")
print(f"uHR = {hr_unadj:.2f} (95% CI: {ci_lower_unadj:.2f}-{ci_upper_unadj:.2f}), P-value: {p_value_unadj:.3f}")


# ============================================================================
# PART 2B: UNADJUSTED MODEL FOR COMORBIDITY (CCI)
# ============================================================================

print("\n" + "="*80)
print("       UNADJUSTED STRATIFIED MODEL (for CCI)")
print("="*80)

# Initialize a new Cox model for the CCI-only analysis
cph_unadj_cci = CoxPHFitter()

try:
    # Fit the model using only the CCI variable
    cph_unadj_cci.fit(df_matched,
                      duration_col='duration',
                      event_col='is_complication',
                      strata=['MatchID'],
                      formula="CCI") # ✅ Note: Only CCI is included

    print("\n--- Unadjusted Stratified Cox Model Results for CCI ---")
    cph_unadj_cci.print_summary()

    # Extract key results for CCI
    hr_unadj_cci = cph_unadj_cci.summary.loc['CCI', 'exp(coef)']
    p_value_unadj_cci = cph_unadj_cci.summary.loc['CCI', 'p']
    ci_lower_unadj_cci = cph_unadj_cci.summary.loc['CCI', 'exp(coef) lower 95%']
    ci_upper_unadj_cci = cph_unadj_cci.summary.loc['CCI', 'exp(coef) upper 95%']

    print("\n--- Summary of Unadjusted Findings for CCI ---")
    print(f"Unadjusted Hazard Ratio (uHR) for each one-point increase in CCI:")
    print(f"uHR = {hr_unadj_cci:.2f} (95% CI: {ci_lower_unadj_cci:.2f}-{ci_upper_unadj_cci:.2f}), P-value: {p_value_unadj_cci:.3f}")

except Exception as e:
    print(f"\nCould not fit the unadjusted CCI model. Error: {e}")


# ============================================================================
# PART 2C: ADJUSTED MODEL (group_tvp + CCI)
# ============================================================================

print("\n" + "="*80)
print("      ADJUSTED STRATIFIED MODEL (Adjusting for CCI)")
print("="*80)

cph_adj = CoxPHFitter()
cph_adj.fit(df_matched,
              duration_col='duration',
              event_col='is_complication',
              strata=['MatchID'],
              formula="group_tvp + CCI")

print("\n--- Adjusted Stratified Cox Model Results ---")
cph_adj.print_summary()
cph_adj.print_summary(decimals=3)

hr_adj = cph_adj.summary.loc['group_tvp', 'exp(coef)']
p_value_adj = cph_adj.summary.loc['group_tvp', 'p']
ci_lower_adj = cph_adj.summary.loc['group_tvp', 'exp(coef) lower 95%']
ci_upper_adj = cph_adj.summary.loc['group_tvp', 'exp(coef) upper 95%']

print("\n--- Summary of Adjusted Findings ---")
print(f"Adjusted Hazard Ratio (aHR) for TVP vs LP (adjusting for CCI):")
print(f"aHR = {hr_adj:.2f} (95% CI: {ci_lower_adj:.2f}-{ci_upper_adj:.2f}), P-value: {p_value_adj:.3f}")