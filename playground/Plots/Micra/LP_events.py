import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import AalenJohansenFitter
from scipy import stats

# Load the data
df = pd.read_csv('playground/Micra/LP_events.csv')

# Drop rows with missing T2Events or AnyEvents
df.dropna(subset=['T2Events', 'AnyEvents'], inplace=True)

# Convert T2Events to numeric and AnyEvents to integer
df['T2Events'] = pd.to_numeric(df['T2Events'], errors='coerce')
df['AnyEvents'] = df['AnyEvents'].astype(int)

# Create a new event variable for competing risk analysis
# 0 = censored, 1 = Complication, 2 = Death
df['status'] = df['AnyEvents'].apply(lambda x: 1 if x == 1 else 2 if x == 2 else 0)

# Create a new duration variable
df['duration'] = df['T2Events']

# Drop any rows that have missing values after coercion
df.dropna(subset=['duration', 'status'], inplace=True)

# **FIX: Clean the lowBSA column**
# Drop rows with missing lowBSA values and ensure it only contains 0 or 1
df.dropna(subset=['lowBSA'], inplace=True)
df['lowBSA'] = df['lowBSA'].astype(int)

# Debug: Check unique values in lowBSA
print(f"Unique values in lowBSA: {df['lowBSA'].unique()}")
print(f"Value counts in lowBSA:\n{df['lowBSA'].value_counts()}")
print("-" * 50)

# Manual implementation of Gray's test
def grays_test(durations, groups, events, event_of_interest):
    """
    Manual implementation of Gray's test for comparing cumulative incidence functions
    """
    unique_groups = np.unique(groups)
    
    print(f"Debug: Unique groups found: {unique_groups}")
    print(f"Debug: Number of unique groups: {len(unique_groups)}")
    
    if len(unique_groups) != 2:
        raise ValueError(f"Gray's test implemented for 2 groups only. Found {len(unique_groups)} groups: {unique_groups}")
    
    # Get unique event times
    event_times = np.unique(durations[events == event_of_interest])
    event_times = event_times[event_times > 0]
    
    observed_diff = []
    expected_diff = []
    variance_terms = []
    
    for t in event_times:
        at_risk_0 = np.sum((durations >= t) & (groups == unique_groups[0]))
        at_risk_1 = np.sum((durations >= t) & (groups == unique_groups[1]))
        total_at_risk = at_risk_0 + at_risk_1
        
        if total_at_risk == 0:
            continue
            
        events_0 = np.sum((durations == t) & (groups == unique_groups[0]) & (events == event_of_interest))
        events_1 = np.sum((durations == t) & (groups == unique_groups[1]) & (events == event_of_interest))
        total_events = events_0 + events_1
        
        if total_events == 0:
            continue
        
        expected_0 = (at_risk_0 * total_events) / total_at_risk
        observed_diff.append(events_0 - expected_0)
        
        # Variance calculation
        if total_at_risk > 1:
            variance = (at_risk_0 * at_risk_1 * total_events * (total_at_risk - total_events)) / \
                      (total_at_risk**2 * (total_at_risk - 1))
        else:
            variance = 0
        variance_terms.append(variance)
    
    if len(observed_diff) == 0:
        return 0, 1.0
        
    U = np.sum(observed_diff)
    V = np.sum(variance_terms)
    
    if V == 0:
        return 0, 1.0
        
    test_statistic = (U**2) / V
    p_value = 1 - stats.chi2.cdf(test_statistic, df=1)
    
    return test_statistic, p_value

# --- Perform Aalen-Johansen (Competing Risk) Analysis ---
print("\n## Cumulative Incidence for the whole LP cohort (Aalen-Johansen)")
ajf = AalenJohansenFitter(seed=42)

# Fit for complications
ajf.fit(df['duration'], df['status'], event_of_interest=1)
ci_complications = ajf.cumulative_density_.iloc[-1,0]
ci_complications_lower = ajf.confidence_interval_cumulative_density_.iloc[-1,0]
ci_complications_upper = ajf.confidence_interval_cumulative_density_.iloc[-1,1]
print(f"Cumulative incidence of complications: {ci_complications:.4f} (95% CI: {ci_complications_lower:.4f}-{ci_complications_upper:.4f})")

# Fit for death
ajf_death = AalenJohansenFitter(seed=42).fit(df['duration'], df['AnyEvents'], event_of_interest=2)
ci_death = ajf_death.cumulative_density_.iloc[-1,0]
ci_death_lower = ajf_death.confidence_interval_cumulative_density_.iloc[-1,0]
ci_death_upper = ajf_death.confidence_interval_cumulative_density_.iloc[-1,1]
print(f"Cumulative incidence of death: {ci_death:.4f} (95% CI: {ci_death_lower:.4f}-{ci_death_upper:.4f})")
print("-" * 50)

# --- Perform competing risk analysis for BSA groups for complications ---
print("\n## Cumulative Incidence for different BSA groups (Aalen-Johansen)")
df_low_bsa = df[df['lowBSA'] == 1]
df_other_bsa = df[df['lowBSA'] == 0]

# Fit for low BSA group death
ajf_low_bsa = AalenJohansenFitter(seed=42)
ajf_low_bsa.fit(df_low_bsa['duration'], df_low_bsa['status'], event_of_interest=2)
ci_low_bsa_death = ajf_low_bsa.cumulative_density_.iloc[-1,0]
ci_low_bsa_death_lower = ajf_low_bsa.confidence_interval_cumulative_density_.iloc[-1,0]
ci_low_bsa_death_upper = ajf_low_bsa.confidence_interval_cumulative_density_.iloc[-1,1]
print(f"Low BSA group (N={len(df_low_bsa)}):")
print(f"Cumulative incidence of death: {ci_low_bsa_death:.4f} (95% CI: {ci_low_bsa_death_lower:.4f}-{ci_low_bsa_death_upper:.4f})")

# Fit for low BSA group complications
ajf_low_bsa = AalenJohansenFitter(seed=42)
ajf_low_bsa.fit(df_low_bsa['duration'], df_low_bsa['status'], event_of_interest=1)
ci_low_bsa = ajf_low_bsa.cumulative_density_.iloc[-1,0]
ci_low_bsa_lower = ajf_low_bsa.confidence_interval_cumulative_density_.iloc[-1,0]
ci_low_bsa_upper = ajf_low_bsa.confidence_interval_cumulative_density_.iloc[-1,1]
print(f"Cumulative incidence of complications: {ci_low_bsa:.4f} (95% CI: {ci_low_bsa_lower:.4f}-{ci_low_bsa_upper:.4f})")

# Fit for other BSA group death
ajf_other_bsa = AalenJohansenFitter(seed=42)
ajf_other_bsa.fit(df_other_bsa['duration'], df_other_bsa['status'], event_of_interest=2)
ci_other_bsa_death = ajf_other_bsa.cumulative_density_.iloc[-1,0]
ci_other_bsa_death_lower = ajf_other_bsa.confidence_interval_cumulative_density_.iloc[-1,0]
ci_other_bsa_death_upper = ajf_other_bsa.confidence_interval_cumulative_density_.iloc[-1,1]
print(f"\nOther BSA group (N={len(df_other_bsa)}):")
print(f"Cumulative incidence of death: {ci_other_bsa_death:.4f} (95% CI: {ci_other_bsa_death_lower:.4f}-{ci_other_bsa_death_upper:.4f})")

# Fit for other BSA group complications
ajf_other_bsa = AalenJohansenFitter(seed=42)
ajf_other_bsa.fit(df_other_bsa['duration'], df_other_bsa['status'], event_of_interest=1)
ci_other_bsa = ajf_other_bsa.cumulative_density_.iloc[-1,0]
ci_other_bsa_lower = ajf_other_bsa.confidence_interval_cumulative_density_.iloc[-1,0]
ci_other_bsa_upper = ajf_other_bsa.confidence_interval_cumulative_density_.iloc[-1,1]
print(f"Cumulative incidence of complications: {ci_other_bsa:.4f} (95% CI: {ci_other_bsa_lower:.4f}-{ci_other_bsa_upper:.4f})")
print("-" * 50)

# --- Perform Gray's test for competing risks ---
print("\n## Gray's Test for Comparing Cumulative Incidence of Complications")
print("(Appropriate statistical test for competing risk analysis)")

try:
    duration = df['duration'].values
    group = df['lowBSA'].values
    event = df['status'].values
    
    # Perform Gray's test for complications
    test_stat_com, p_val_com = grays_test(duration, group, event, event_of_interest=1)
    
    print(f"Gray's Test Statistic for complication: {test_stat_com:.4f}")
    print(f"p-value: {p_val_com:.4f}")

    # Perform Gray's test for complications
    test_stat_death, p_val_death = grays_test(duration, group, event, event_of_interest=2)
    
    print(f"Gray's Test Statistic for death: {test_stat_death:.4f}")
    print(f"p-value: {p_val_death:.4f}")
    

        
except Exception as e:
    print(f"Error running Gray's test: {e}")
    print("Falling back to descriptive comparison")
    difference = abs(ci_low_bsa - ci_other_bsa)
    print(f"Difference in cumulative incidence: {difference:.4f}")

print("-" * 50)

# --- PLOT 1: Cumulative Incidence of Death (Whole Cohort) ---
plt.figure(figsize=(10, 6))
ajf_death.plot_cumulative_density(label=f'Aalen-Johansen (All Patients, N={len(df)})', ci_show=True)
plt.title('Cumulative Incidence of Death (Competing Risk)')
plt.xlabel('Time (days)')
plt.ylabel('Cumulative Incidence')
plt.grid(True)
plt.legend()
plt.tight_layout()
#plt.show()

# --- PLOT 2: Comparison of Complication Incidence by BSA Group ---
plt.figure(figsize=(10, 6))
ax = plt.gca()

ajf_low_bsa.plot_cumulative_density(ax=ax, label=f'Low BSA (N={len(df_low_bsa)})', linestyle='--', ci_show=True)
ajf_other_bsa.plot_cumulative_density(ax=ax, label=f'Other BSA (N={len(df_other_bsa)})', linestyle='-', ci_show=True)

plt.title('Comparison of Complication Incidence by BSA Group')
plt.xlabel('Time (days)')
plt.ylabel('Cumulative Incidence of Complications')
plt.grid(True)
plt.legend()
plt.tight_layout()
#plt.show()

# --- Summary of Key Results ---
print("\n" + "="*60)
print("SUMMARY OF KEY RESULTS")
print("="*60)
print(f"Overall cohort (N={len(df)}):")
print(f"  Complications: {ci_complications:.4f} (95% CI: {ci_complications_lower:.4f}-{ci_complications_upper:.4f})")
print(f"  Death: {ci_death:.4f} (95% CI: {ci_death_lower:.4f}-{ci_death_upper:.4f})")
print()
print("="*60)

# --- Crude Incidence Calculation (simple proportion) ---
print("\n## Crude Incidence Rates (Simple Proportions)")
print("(Does not account for time or competing risks)")

df['death'] = (df['Death']).astype(int)
# Overall cohort
total_patients = len(df)
total_complications = (df['status'] == 1).sum()
total_deaths = (df['death'] == 1).sum()
crude_comp_overall = total_complications / total_patients
crude_death_overall = total_deaths / total_patients
print(f"Overall Cohort (N={total_patients}):")
print(f"  Crude Incidence of Complications: {crude_comp_overall:.4f} ({total_complications}/{total_patients})")
print(f"  Crude Incidence of Death: {crude_death_overall:.4f} ({total_deaths}/{total_patients})")

# Low BSA group
total_low_bsa = len(df_low_bsa)
comp_low_bsa = (df_low_bsa['status'] == 1).sum()
crude_comp_low_bsa = comp_low_bsa / total_low_bsa
print(f"\nLow BSA Group (N={total_low_bsa}):")
print(f"  Crude Incidence of Complications: {crude_comp_low_bsa:.4f} ({comp_low_bsa}/{total_low_bsa})")

# Other BSA group
total_other_bsa = len(df_other_bsa)
comp_other_bsa = (df_other_bsa['status'] == 1).sum()
crude_comp_other_bsa = comp_other_bsa / total_other_bsa
print(f"\nOther BSA Group (N={total_other_bsa}):")
print(f"  Crude Incidence of Complications: {crude_comp_other_bsa:.4f} ({comp_other_bsa}/{total_other_bsa})")

print("-" * 50)