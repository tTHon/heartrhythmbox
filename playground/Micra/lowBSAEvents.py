import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import AalenJohansenFitter, KaplanMeierFitter
from scipy import stats

# Load the data
df = pd.read_csv('playground/Micra/lowBSA_events.csv')

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

# Manual implementation of Gray's test
def grays_test(durations, groups, events, event_of_interest=1):
    """
    Manual implementation of Gray's test for comparing cumulative incidence functions
    """
    unique_groups = np.unique(groups)
    
    if len(unique_groups) != 2:
        raise ValueError("Gray's test implemented for 2 groups only")
    
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
print("## Cumulative Incidence for the whole low BSA cohort (Aalen-Johansen)")
ajf = AalenJohansenFitter(seed=42)

# Fit for complications
ajf.fit(df['duration'], df['status'], event_of_interest=1)
ci_complications = ajf.cumulative_density_.iloc[-1,0]
ci_complications_lower = ajf.confidence_interval_cumulative_density_.iloc[-1,0]
ci_complications_upper = ajf.confidence_interval_cumulative_density_.iloc[-1,1]
print(f"Cumulative incidence of complications of low BSA LP and TV: {ci_complications:.4f} (95% CI: {ci_complications_lower:.4f}-{ci_complications_upper:.4f})")

# Fit for death
ajf_death = AalenJohansenFitter(seed=42).fit(df['duration'], df['status'], event_of_interest=2)
ci_death = ajf_death.cumulative_density_.iloc[-1,0]
ci_death_lower = ajf_death.confidence_interval_cumulative_density_.iloc[-1,0]
ci_death_upper = ajf_death.confidence_interval_cumulative_density_.iloc[-1,1]
print(f"Cumulative incidence of death of low BSA LP and TV: {ci_death:.4f} (95% CI: {ci_death_lower:.4f}-{ci_death_upper:.4f})")
print("-" * 50)

# --- Perform competing risk analysis for BSA groups ---
print("\n## Cumulative Incidence for LP vs TV (Aalen-Johansen)")
df_LP = df[df['Type'] == "LP"]
df_TV = df[df['Type'] == "TV"]

# Fit for LP
ajf_LP = AalenJohansenFitter(seed=42)
ajf_LP.fit(df_LP['duration'], df_LP['status'], event_of_interest=1)
ci_LP = ajf_LP.cumulative_density_.iloc[-1,0]
ci_LP_lower = ajf_LP.confidence_interval_cumulative_density_.iloc[-1,0]
ci_LP_upper = ajf_LP.confidence_interval_cumulative_density_.iloc[-1,1]
print(f"LP group (N={len(df_LP)}):")
print(f"Cumulative incidence of complications: {ci_LP:.4f} (95% CI: {ci_LP_lower:.4f}-{ci_LP_upper:.4f})")

# Fit for TV
ajf_TV = AalenJohansenFitter(seed=42)
ajf_TV.fit(df_TV['duration'], df_TV['status'], event_of_interest=1)
ci_TV = ajf_TV.cumulative_density_.iloc[-1,0]
ci_TV_lower = ajf_TV.confidence_interval_cumulative_density_.iloc[-1,0]
ci_TV_upper = ajf_TV.confidence_interval_cumulative_density_.iloc[-1,1]
print(f"\nTV (N={len(df_TV)}):")
print(f"Cumulative incidence of complications: {ci_TV:.4f} (95% CI: {ci_TV_lower:.4f}-{ci_TV_upper:.4f})")
print("-" * 50)

# --- Perform Gray's test for competing risks ---
print("\n## Gray's Test for Comparing Cumulative Incidence of Complications")
print("(Appropriate statistical test for competing risk analysis)")

try:
    duration = df['duration'].values
    group = df['Type'].values
    event = df['status'].values
    
    # Perform Gray's test
    test_stat, p_val = grays_test(duration, group, event, event_of_interest=1)
    
    print(f"Gray's Test Statistic: {test_stat:.4f}")
    print(f"p-value: {p_val:.4f}")
    
    if p_val < 0.05:
        print("Result: Statistically significant difference between groups (p < 0.05)")
    else:
        print("Result: No statistically significant difference between groups (p ≥ 0.05)")
        
except Exception as e:
    print(f"Error running Gray's test: {e}")
    print("Falling back to descriptive comparison")
    difference = abs(ci_LP - ci_TV)
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
plt.show()


# --- PLOT 2: Comparison of Complication Incidence by BSA Group ---
plt.figure(figsize=(10, 6))
ax = plt.gca() # Get current axes

ajf_LP.plot_cumulative_density(ax=ax, label=f'LP (N={len(df_LP)})', linestyle='--', ci_show=True)
ajf_TV.plot_cumulative_density(ax=ax, label=f'TV (N={len(df_LP)})', linestyle='-', ci_show=True)

plt.title('Comparison of Complication Incidence by BSA Group')
plt.xlabel('Time (days)')
plt.ylabel('Cumulative Incidence of Complications')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# --- Summary of Key Results ---
print("\n" + "="*60)
print("SUMMARY OF KEY RESULTS")
print("="*60)
print(f"Overall cohort (N={len(df)}):")
print(f"  Complications: {ci_complications:.4f} (95% CI: {ci_complications_lower:.4f}-{ci_complications_upper:.4f})")
print(f"  Death: {ci_death:.4f} (95% CI: {ci_death_lower:.4f}-{ci_death_upper:.4f})")
print()
print("Group comparison (Aalen-Johansen):")
print(f"  LP: {ci_LP:.4f} (95% CI: {ci_LP_lower:.4f}-{ci_LP_upper:.4f})")
print(f"  TV: {ci_TV:.4f} (95% CI: {ci_TV_lower:.4f}-{ci_TV_upper:.4f})")
print(f"  Gray's test p-value: {p_val:.4f}")
print("="*60)