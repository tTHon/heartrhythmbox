import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import AalenJohansenFitter

# ============================================================================
# PART 0: LOAD AND PREPARE DATA
# ============================================================================
# Load the dataset
file_path = 'playground/Micra/lowBSA.csv'
df = pd.read_csv(file_path)

# Filter for matched cohort
lp_lowbsa_match_ids = df[df['Type'] == 1]['MatchID'].unique()
df_matched = df[df['MatchID'].isin(lp_lowbsa_match_ids)].copy()

# Prepare the multi-level status variable
outcome_comp = np.zeros(len(df_matched))
outcome_comp[df_matched['Complication'] == 1] = 1
outcome_comp[(df_matched['Death'] == 1) & (df_matched['Complication'] == 0)] = 2
df_matched['status_comp'] = outcome_comp.astype(int)

# General variables
df_matched['duration'] = df_matched['T2Events']

# Separate dataframes for each group
df_lp = df_matched[df_matched['Type'] == 1].copy()
df_tvp = df_matched[df_matched['Type'] == 0].copy()

print("Data preparation complete.")
print(f"LP Group size: {len(df_lp)}, TVP Group size: {len(df_tvp)}")

# ============================================================================
# PLOT 1: STACKED CUMULATIVE INCIDENCE PLOT
# ============================================================================
def build_cif_dataframe(duration, status, group_name):
    print(f"\nProcessing {group_name}...")

    # Fit once for event 1
    ajf_event1 = AalenJohansenFitter().fit(duration, status, event_of_interest=1)
    # Fit once for event 2
    ajf_event2 = AalenJohansenFitter().fit(duration, status, event_of_interest=2)

    # Extract cumulative incidence series, only 1 column per fit
    cif1_raw = ajf_event1.cumulative_density_.iloc[:, 0] if not ajf_event1.cumulative_density_.empty else pd.Series(dtype=float)
    cif2_raw = ajf_event2.cumulative_density_.iloc[:, 0] if not ajf_event2.cumulative_density_.empty else pd.Series(dtype=float)

    # Create a unified timeline that contains all time points
    timeline = pd.Index([0]).union(cif1_raw.index).union(cif2_raw.index)

    # Reindex both cumulative incidence series to the timeline, forward fill and fill initial NaNs with 0
    cif1_aligned = cif1_raw.reindex(timeline, method='ffill').fillna(0)
    cif2_aligned = cif2_raw.reindex(timeline, method='ffill').fillna(0)

    # Combine into a single DataFrame
    df_cif = pd.DataFrame({'Complication': cif1_aligned, 'Death': cif2_aligned})

    print(f"--- {group_name} Data for Plotting ---")
    print(df_cif.tail())

    return df_cif



# --- Build the dataframes for each group using the helper function ---
cif_lp = build_cif_dataframe(df_lp['duration'], df_lp['status_comp'], "LP Group")
cif_tvp = build_cif_dataframe(df_tvp['duration'], df_tvp['status_comp'], "TVP Group")

# --- Create the plot with Seaborn styling ---
sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#f7f7f7"})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

# Define professional colors
color_comp = '#0072B2' # Blue
color_death = '#D55E00' # Vermillion

# Plot for LP Group
ax1.fill_between(cif_lp.index, 0, cif_lp['Complication'], color=color_comp, alpha=0.8, label='Complication')
ax1.fill_between(cif_lp.index, cif_lp['Complication'], cif_lp['Complication'] + cif_lp['Death'], color=color_death, alpha=0.8, label='Death')
ax1.set_title(f'Leadless Pacemaker (LP) Group (N={len(df_lp)})', fontsize=14, fontweight='bold')
ax1.set_xlabel('Time (days)', fontsize=12)
ax1.set_ylabel('Cumulative Incidence', fontsize=12)
ax1.tick_params(axis='both', which='major', labelsize=10)
ax1.legend(frameon=True, facecolor='white', framealpha=0.7)
ax1.set_ylim(0, 0.6) # Adjust Y-axis for better visibility
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)


# Plot for TVP Group
ax2.fill_between(cif_tvp.index, 0, cif_tvp['Complication'], color=color_comp, alpha=0.8, label='Complication')
ax2.fill_between(cif_tvp.index, cif_tvp['Complication'], cif_tvp['Complication'] + cif_tvp['Death'], color=color_death, alpha=0.8, label='Death')
ax2.set_title(f'Transvenous Pacemaker (TVP) Group (N={len(df_tvp)})', fontsize=14, fontweight='bold')
ax2.set_xlabel('Time (days)', fontsize=12)
ax2.tick_params(axis='both', which='major', labelsize=10)
ax2.legend(frameon=True, facecolor='white', framealpha=0.7)
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

# Overall Title
plt.suptitle('Stacked Cumulative Incidence of Competing Events', fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()