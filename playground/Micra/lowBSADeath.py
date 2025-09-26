import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
file_path = 'playground/Micra/lowBSA.csv'
df = pd.read_csv(file_path)

# Filter for matched cohort
lp_lowbsa_match_ids = df[df['Type'] == 1]['MatchID'].unique()
df_matched = df[df['MatchID'].isin(lp_lowbsa_match_ids)].copy()

# ============================================================================
# PART 1: PREPARE DATA FOR ANALYSIS OF DEATH
# ============================================================================

# Step 1: Create a clear multi-level status column where DEATH is the event
# 0 = Censored, 1 = Death (event), 2 = Complication (competing risk)
outcome_death = np.zeros(len(df_matched))
outcome_death[df_matched['Death'] == 1] = 1
outcome_death[(df_matched['Complication'] == 1) & (df_matched['Death'] == 0)] = 2
df_matched['status_death'] = outcome_death.astype(int)

# Step 2: Create the specific event column for the cause-specific model
# The event is 'death' (status_death == 1). Complication (status_death == 2) becomes a non-event (0).
df_matched['is_death'] = (df_matched['status_death'] == 1).astype(int)

# Use T2Events for duration and create numeric group variable
df_matched['duration'] = df_matched['T2Events']
df_matched['group_tvp'] = (df_matched['Type'] == 0).astype(int)

print("\n--- Data prepared for Adjusted Cox Model for DEATH ---")
print(df_matched[['MatchID', 'Type', 'duration', 'is_death', 'CCI']].head())

# ============================================================================
# PART 2: RUN THE ADJUSTED STRATIFIED MODEL FOR DEATH
# ============================================================================

print("\n" + "="*80)
print("     ADJUSTED STRATIFIED CAUSE-SPECIFIC MODEL FOR DEATH (Adjusting for CCI)")
print("="*80)

cph_death = CoxPHFitter()

try:
    cph_death.fit(df_matched,
                  duration_col='duration',
                  event_col='is_death', # Use the new event column for death
                  strata=['MatchID'],
                  formula="group_tvp + CCI")

    print("\n--- Adjusted Stratified Cox Model Results for DEATH ---")
    cph_death.print_summary()

    # Extract key results for summary
    hr_adj_death = cph_death.summary.loc['group_tvp', 'exp(coef)']
    p_value_adj_death = cph_death.summary.loc['group_tvp', 'p']
    ci_lower_adj_death = cph_death.summary.loc['group_tvp', 'exp(coef) lower 95%']
    ci_upper_adj_death = cph_death.summary.loc['group_tvp', 'exp(coef) upper 95%']

    print("\n--- Summary of Findings for DEATH ---")
    print(f"Adjusted Hazard Ratio (aHR) for TVP vs LP (adjusting for CCI):")
    print(f"aHR = {hr_adj_death:.2f} (95% CI: {ci_lower_adj_death:.2f}-{ci_upper_adj_death:.2f})")
    print(f"P-value: {p_value_adj_death:.3f}")

except Exception as e:
    print(f"\nCould not fit the model. Error: {e}")