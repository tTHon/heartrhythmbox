import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import AalenJohansenFitter, CoxPHFitter
from lifelines.plotting import add_at_risk_counts


# Load the dataset
file_path = 'playground/Micra/lowBSA.csv'
try:
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"ERROR: The file '{file_path}' was not found.")
    exit()
# Drop rows with missing T2Events or AnyEvents
df.dropna(subset=['T2Events', 'AnyEvents'], inplace=True)

# Filter for matched cohort
lp_lowbsa_match_ids = df[(df['Type'] == 1)]['MatchID'].unique()
df_matched = df[df['MatchID'].isin(lp_lowbsa_match_ids)].copy()
print(f"Matched cohort: {df_matched.shape[0]} records, {len(lp_lowbsa_match_ids)} matched sets")

# ============================================================================
# PART 1: PREPARE DATA FOR TIME-TO-EVENT ANALYSIS
# ============================================================================

# Step 1: Create a clear multi-level status column for competing risks
# 0 = Censored, 1 = Complication (event), 2 = Death (competing risk)
outcome = np.zeros(len(df_matched))
outcome[df_matched['Complication'] == 1] = 1
outcome[(df_matched['Death'] == 1) & (df_matched['Complication'] == 0)] = 2
df_matched['status'] = outcome.astype(int)

# Step 2: Create the specific event column for the cause-specific model
df_matched['is_complication'] = (df_matched['status'] == 1).astype(int)
df_matched['is_death'] = (df_matched['status'] == 2).astype(int)

# Use T2Events for duration
df_matched['duration'] = df_matched['T2Events']

# Create numeric group variable for the model
df_matched['group_tvp'] = (df_matched['Type'] == 0).astype(int)

#print("\n--- Data prepared for all analyses ---")
#print(df_matched[['MatchID', 'Type', 'duration', 'status', 'is_complication']].head())


# ============================================================================
# PART 2: AALEN-JOHANSEN CUMULATIVE INCIDENCE PLOT
# ============================================================================

print("\n" + "="*80)
print("        AALEN-JOHANSEN CUMULATIVE INCIDENCE OF COMPLICATIONS")
print("="*80)

# Separate data for each group to plot their curves
df_lp = df_matched[df_matched['Type'] == 1]
df_tvp = df_matched[df_matched['Type'] == 0]

# Initialize Aalen-Johansen Fitters for each group
ajf_comp = AalenJohansenFitter()
ajf_lp = AalenJohansenFitter()
ajf_tvp = AalenJohansenFitter()

ajf_comp.fit(df_matched['duration'], df_matched['status'], event_of_interest=2)
print("\n--- Aalen-Johansen Fit Summary for Death---")

ci_comp = ajf_comp.cumulative_density_.iloc[-1,0]
ci_comp_lower = ajf_comp.confidence_interval_cumulative_density_.iloc[-1,0]
ci_comp_upper = ajf_comp.confidence_interval_cumulative_density_.iloc[-1,1]
print(f"Cumulative incidence of complications (all patients): {ci_comp:.2f} (95% CI: {ci_comp_lower:.2f}-{ci_comp_upper:.2f})")


# Fit the model for each group for the event of interest (complication, status==1)
ajf_lp.fit(df_lp['duration'], df_lp['status'], event_of_interest=1)
ajf_tvp.fit(df_tvp['duration'], df_tvp['status'], event_of_interest=1)

#print("\n--- Aalen-Johansen Fit Summary for LP Group ---")
ci_complications = ajf_lp.cumulative_density_.iloc[-1,0]
ci_complications_lower = ajf_lp.confidence_interval_cumulative_density_.iloc[-1,0]
ci_complications_upper = ajf_lp.confidence_interval_cumulative_density_.iloc[-1,1]
print(f"Cumulative incidence of complications LP group: {ci_complications:.2f} (95% CI: {ci_complications_lower:.2f}-{ci_complications_upper:.2f})")

#print("\n--- Aalen-Johansen Fit Summary for TVP Group ---")
ci_complications = ajf_tvp.cumulative_density_.iloc[-1,0]
ci_complications_lower = ajf_tvp.confidence_interval_cumulative_density_.iloc[-1,0]
ci_complications_upper = ajf_tvp.confidence_interval_cumulative_density_.iloc[-1,1]
print(f"Cumulative incidence of complications TV group: {ci_complications:.2f} (95% CI: {ci_complications_lower:.2f}-{ci_complications_upper:.2f})")


# Plot the cumulative incidence curves
plt.figure(figsize=(10, 6))
ax = plt.gca()

ajf_lp.plot_cumulative_density(ax=ax, label=f'LP Group (N={len(df_lp)})', ci_show=True)
ajf_tvp.plot_cumulative_density(ax=ax, label=f'TVP Group (N={len(df_tvp)})', ci_show=True)

plt.title('Cumulative Incidence of Complications (Aalen-Johansen)')
plt.xlabel('Time (days)')
plt.ylabel('Cumulative Incidence')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plotting Section ---

# Use a professional and clean plot style
sns.set_style("whitegrid")

plt.figure(figsize=(12, 8))
ax = plt.gca()

# Plot the cumulative incidence for the LP group
ajf_lp.plot_cumulative_density(
    ax=ax,
    label='LP Group (N={})'.format(len(df_lp)),
    ci_show=True,
    color='#1f77b4'  # Professional blue
)

# Plot the cumulative incidence for the TVP group
ajf_tvp.plot_cumulative_density(
    ax=ax,
    label='TVP Group (N={})'.format(len(df_tvp)),
    ci_show=True,
    color='#ff7f0e'  # Professional orange
)

# --- Add Censor Markers ---
# We need to manually plot the censored events on the curves

# For LP Group
lp_censored = df_lp[df_lp['status'] == 0]
lp_cif_at_censoring = ajf_lp.cumulative_density_.loc[lp_censored['duration'].values]
ax.scatter(lp_cif_at_censoring.index, lp_cif_at_censoring.values,
           marker='+', color='#1f77b4', s=30, label='LP Censored')

# For TVP Group
tvp_censored = df_tvp[df_tvp['status'] == 0]
tvp_cif_at_censoring = ajf_tvp.cumulative_density_.loc[tvp_censored['duration'].values]
ax.scatter(tvp_cif_at_censoring.index, tvp_cif_at_censoring.values,
           marker='+', color='#ff7f0e', s=30, label='TVP Censored')


# --- Customize the Plot ---

# Add titles and labels with better formatting
plt.title('Cumulative Incidence of Complications', fontsize=16, fontweight='bold')
plt.xlabel('Time (days)', fontsize=12)
plt.ylabel('Cumulative Incidence', fontsize=12)
plt.ylim(0, 0.5) # Adjust y-axis limit for better visualization if incidence is low

# Improve the legend
handles, labels = ax.get_legend_handles_labels()
# Manually re-order legend to group censored markers with their lines
order = [0, 2, 1, 3]
ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper left', fontsize=10)


# --- Add At-Risk Table ---
# This table shows the number of patients remaining at risk over time
add_at_risk_counts(ajf_lp, ajf_tvp, ax=ax)

plt.tight_layout()
plt.show()

# Fit for death
ajf_death = AalenJohansenFitter()
ajf_death.fit(df_matched['duration'], df_matched['status'], event_of_interest=2)
ajf_lp.fit(df_lp['duration'], df_lp['status'], event_of_interest=2)
ajf_tvp.fit(df_tvp['duration'], df_tvp['status'], event_of_interest=2)

print("\n--- Aalen-Johansen Fit Summary for Death---")
ci_death = ajf_death.cumulative_density_.iloc[-1,0]
ci_death_lower = ajf_death.confidence_interval_cumulative_density_.iloc[-1,0]
ci_death_upper = ajf_death.confidence_interval_cumulative_density_.iloc[-1,1]
ci_death_lp = ajf_lp.cumulative_density_.iloc[-1,0]
ci_death_lp_lower = ajf_lp.confidence_interval_cumulative_density_.iloc[-1,0]
ci_death_lp_upper = ajf_lp.confidence_interval_cumulative_density_.iloc[-1,1]
ci_death_tvp = ajf_tvp.cumulative_density_.iloc[-1,0]
ci_death_tvp_lower = ajf_tvp.confidence_interval_cumulative_density_.iloc[-1,0]
ci_death_tvp_upper = ajf_tvp.confidence_interval_cumulative_density_.iloc[-1,1]     
print(f"Cumulative incidence of death (all patients): {ci_death:.2f} (95% CI: {ci_death_lower:.2f}-{ci_death_upper:.2f})")
print(f"Cumulative incidence of death (LP group): {ci_death_lp:.2f} (95% CI: {ci_death_lp_lower:.2f}-{ci_death_lp_upper:.2f})")
print(f"Cumulative incidence of death (TVP group): {ci_death_tvp:.2f} (95% CI: {ci_death_tvp_lower:.2f}-{ci_death_tvp_upper:.2f})")


# ============================================================================
# PART 3: STRATIFIED CAUSE-SPECIFIC COX MODEL
# ============================================================================

print("\n" + "="*80)
print("             STRATIFIED CAUSE-SPECIFIC MODEL FOR TIME-TO-COMPLICATION")
print("="*80)

cph = CoxPHFitter()

try:
    cph.fit(df_matched,
            duration_col='duration',
            event_col='is_complication',
            strata=['MatchID'],
            formula="group_tvp")

    #print("\n--- Stratified Cox Model Results ---")
    #cph.print_summary()

    # Extract key results for summary
    p_value = cph.summary.loc['group_tvp', 'p']
    hr = cph.summary.loc['group_tvp', 'exp(coef)']
    ci_lower = cph.summary.loc['group_tvp', 'exp(coef) lower 95%']
    ci_upper = cph.summary.loc['group_tvp', 'exp(coef) upper 95%']

    print("\n--- Summary of Findings ---")
    print(f"Cause-Specific Hazard Ratio (TVP vs LP): {hr:.2f} (95% CI: {ci_lower:.2f}-{ci_upper:.2f})")
    print(f"P-value: {p_value:.2f}")

except Exception as e:
    print(f"\nCould not fit the model. Error: {e}")