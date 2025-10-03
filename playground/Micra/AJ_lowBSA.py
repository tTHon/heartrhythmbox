import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import AalenJohansenFitter


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

#print("\n" + "="*80)
#print("        AALEN-JOHANSEN CUMULATIVE INCIDENCE OF COMPLICATIONS")
#print("="*80)

# Separate data for each group to plot their curves
df_lp = df_matched[df_matched['Type'] == 1]
df_tvp = df_matched[df_matched['Type'] == 0]

# Initialize Aalen-Johansen Fitters for each group
ajf_comp = AalenJohansenFitter()
ajf_comp_lp = AalenJohansenFitter()
ajf_comp_tvp = AalenJohansenFitter()

ajf_comp.fit(df_matched['duration'], df_matched['status'], event_of_interest=1)
print("\n--- Aalen-Johansen Fit Summary for complication---")

ci_comp = ajf_comp.cumulative_density_.iloc[-1,0]
ci_comp_lower = ajf_comp.confidence_interval_cumulative_density_.iloc[-1,0]
ci_comp_upper = ajf_comp.confidence_interval_cumulative_density_.iloc[-1,1]
print(f"Cumulative incidence of complications (all patients): {ci_comp:.4f} (95% CI: {ci_comp_lower:.4f}-{ci_comp_upper:.4f})")


# Fit the model for each group for the event of interest (complication, status==1)
ajf_comp_lp.fit(df_lp['duration'], df_lp['status'], event_of_interest=1)
ajf_comp_tvp.fit(df_tvp['duration'], df_tvp['status'], event_of_interest=1)

#print("\n--- Aalen-Johansen Fit Summary for LP Group ---")
ci_complications = ajf_comp_lp.cumulative_density_.iloc[-1,0]
ci_complications_lower = ajf_comp_lp.confidence_interval_cumulative_density_.iloc[-1,0]
ci_complications_upper = ajf_comp_lp.confidence_interval_cumulative_density_.iloc[-1,1]
print(f"Cumulative incidence of complications LP group: {ci_complications:.4f} (95% CI: {ci_complications_lower:.4f}-{ci_complications_upper:.4f})")

#print("\n--- Aalen-Johansen Fit Summary for TVP Group ---")
ci_complications = ajf_comp_tvp.cumulative_density_.iloc[-1,0]
ci_complications_lower = ajf_comp_tvp.confidence_interval_cumulative_density_.iloc[-1,0]
ci_complications_upper = ajf_comp_tvp.confidence_interval_cumulative_density_.iloc[-1,1]
print(f"Cumulative incidence of complications TV group: {ci_complications:.4f} (95% CI: {ci_complications_lower:.4f}-{ci_complications_upper:.4f})")

# Fit the model for the event of interest (death, status==2)
ajf_death = AalenJohansenFitter()
ajf_death.fit(df_matched['duration'], df_matched['status'], event_of_interest=2)
print("\n--- Aalen-Johansen Fit Summary for Death ---")
ci_death = ajf_death.cumulative_density_.iloc[-1,0]
ci_death_lower = ajf_death.confidence_interval_cumulative_density_.iloc[-1,0]
ci_death_upper = ajf_death.confidence_interval_cumulative_density_.iloc[-1,1]
print(f"Cumulative incidence of death (all patients): {ci_death:.4f} (95% CI: {ci_death_lower:.4f}-{ci_death_upper:.4f})")

# Fit the model for each group for the event of interest (death, status==2)
ajf_lp_death = AalenJohansenFitter()
ajf_tvp_death = AalenJohansenFitter()
ajf_lp_death.fit(df_lp['duration'], df_lp['status'], event_of_interest=2)
ajf_tvp_death.fit(df_tvp['duration'], df_tvp['status'], event_of_interest=2)
print("\n--- Aalen-Johansen Fit Summary for Death in LP Group ---")    
ci_death = ajf_lp_death.cumulative_density_.iloc[-1,0]
ci_death_lower = ajf_lp_death.confidence_interval_cumulative_density_.iloc[-1,0]
ci_death_upper = ajf_lp_death.confidence_interval_cumulative_density_.iloc[-1,1]
print(f"Cumulative incidence of death LP group: {ci_death:.4f} (95% CI: {ci_death_lower:.4f}-{ci_death_upper:.4f})")
print("\n--- Aalen-Johansen Fit Summary for Death in TVP Group ---")
ci_death = ajf_tvp_death.cumulative_density_.iloc[-1,0]
ci_death_lower = ajf_tvp_death.confidence_interval_cumulative_density_.iloc[-1,0]
ci_death_upper = ajf_tvp_death.confidence_interval_cumulative_density_.iloc[-1,1]
print(f"Cumulative incidence of death TVP group: {ci_death:.4f} (95% CI: {ci_death_lower:.4f}-{ci_death_upper:.4f})")

# 4. Plot the results
# Set a professional theme
sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#FDFDFD", "grid.linestyle": ":"})
colors = sns.color_palette("Set2", 2)
color_complication = colors[0]
color_death = colors[1]

# 5 Prepare cumulative incidence results for plotting

# Get time points and cumulative incidence for complications and death (LP)
cif_results = pd.DataFrame({
    'time': ajf_comp_lp.cumulative_density_.index/365.25,
    'cif_complication': ajf_comp_lp.cumulative_density_.iloc[:, 0].values,
    'cif_death': ajf_lp_death.cumulative_density_.iloc[:, 0].values
})

fig, ax = plt.subplots(figsize=(10, 6))

# Plot CIF for Complication
ax.step(cif_results['time'], cif_results['cif_complication'], where='post', label='Complication', color=color_complication)
# Plot CIF for Death
ax.step(cif_results['time'], cif_results['cif_death'], where='post', label='Death', color=color_death)

# Create a stacked plot for better visualization
ax.fill_between(cif_results['time'], 0, cif_results['cif_complication'], step='post', alpha=0.7, color=color_complication)
ax.fill_between(cif_results['time'], cif_results['cif_complication'], 
                cif_results['cif_complication'] + cif_results['cif_death'], step='post', alpha=0.7, color=color_death)

# Customize titles, labels, and ticks
N = len(df_matched)
main_title = f'Cumulative Incidence of Complications and Death for low BSA LP (N={N})'
ax.set_title(label=main_title, loc='center', fontsize=15, fontweight='bold', pad=10)
ax.set_xlabel('Time (years)', fontsize=14)
ax.set_ylabel('Cumulative Incidence', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left', fontsize=14)
ax.set_ylim(0, 1.0)
ax.set_xlim(left=0)

# *** ANNOTATION WITH REVISED TEXT ***
max_time = cif_results['time'].max()
final_comp_inc = cif_results.loc[cif_results['time'] == max_time, 'cif_complication'].values[0]
final_death_inc = cif_results.loc[cif_results['time'] == max_time, 'cif_death'].values[0]
textstr = (f"Cumulative Incidence by Aalen-Johansen at {max_time:.1f} years:\n"
            f"Complication: {final_comp_inc:.2%}\n"
            f"Death: {final_death_inc:.2%}")
props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='lightgray')
ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=13,
        verticalalignment='top', horizontalalignment='right', bbox=props)
#plt.show()

# Get time points and cumulative incidence for complications and death (TVP)
cif_results = pd.DataFrame({
    'time': ajf_comp_tvp.cumulative_density_.index/365.25,
    'cif_complication': ajf_comp_tvp.cumulative_density_.iloc[:, 0].values,
    'cif_death': ajf_tvp_death.cumulative_density_.iloc[:, 0].values
})

fig, ax = plt.subplots(figsize=(10, 6))

# Plot CIF for Complication
ax.step(cif_results['time'], cif_results['cif_complication'], where='post', label='Complication', color=color_complication)
# Plot CIF for Death
ax.step(cif_results['time'], cif_results['cif_death'], where='post', label='Death', color=color_death)

# Create a stacked plot for better visualization
ax.fill_between(cif_results['time'], 0, cif_results['cif_complication'], step='post', alpha=0.7, color=color_complication)
ax.fill_between(cif_results['time'], cif_results['cif_complication'], 
                cif_results['cif_complication'] + cif_results['cif_death'], step='post', alpha=0.7, color=color_death)

# Customize titles, labels, and ticks
N = len(df_matched)
main_title = f'Cumulative Incidence of Complications and Death for TVP (N={N})'
ax.set_title(label=main_title, loc='center', fontsize=15, fontweight='bold', pad=10)
ax.set_xlabel('Time (years)', fontsize=14)
ax.set_ylabel('Cumulative Incidence', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left', fontsize=14)
ax.set_ylim(0, 1.0)
ax.set_xlim(left=0)

# *** ANNOTATION WITH REVISED TEXT ***
max_time = cif_results['time'].max()
final_comp_inc = cif_results.loc[cif_results['time'] == max_time, 'cif_complication'].values[0]
final_death_inc = cif_results.loc[cif_results['time'] == max_time, 'cif_death'].values[0]
textstr = (f"Cumulative Incidence by Aalen-Johansen at {max_time:.1f} years:\n"
            f"Complication: {final_comp_inc:.2%}\n"
            f"Death: {final_death_inc:.2%}")
props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='lightgray')
ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=13,
        verticalalignment='top', horizontalalignment='right', bbox=props)
#plt.show()

# ============================================================================
# PART 3: COMBINED PLOT FOR LP vs. TVP
# ============================================================================

# Prepare data for both groups again for clarity
# LP Data
cif_results_lp = pd.DataFrame({
    'time': ajf_comp_lp.cumulative_density_.index / 365.25,
    'cif_complication': ajf_comp_lp.cumulative_density_.iloc[:, 0].values,
    'cif_death': ajf_lp_death.cumulative_density_.iloc[:, 0].values
})

# TVP Data
cif_results_tvp = pd.DataFrame({
    'time': ajf_comp_tvp.cumulative_density_.index / 365.25,
    'cif_complication': ajf_comp_tvp.cumulative_density_.iloc[:, 0].values,
    'cif_death': ajf_tvp_death.cumulative_density_.iloc[:, 0].values
})


# --- Create the Combined Plot ---
fig, ax = plt.subplots(figsize=(12, 8))
colors_lp = sns.color_palette("Blues_d", 2)
colors_tvp = sns.color_palette("Reds_d", 2)



# Plot LP Group Curves
ax.step(cif_results_lp['time'], cif_results_lp['cif_complication'],
        where='post', label='LP - Complication', color=colors_lp[0], linestyle='-')
ax.step(cif_results_lp['time'], cif_results_lp['cif_death'],
        where='post', label='LP - Death', color=colors_lp[1], linestyle='-')

# Plot TVP Group Curves
ax.step(cif_results_tvp['time'], cif_results_tvp['cif_complication'],
        where='post', label='TVP - Complication', color=colors_tvp[0], linestyle='--')
ax.step(cif_results_tvp['time'], cif_results_tvp['cif_death'],
        where='post', label='TVP - Death', color=colors_tvp[1], linestyle='--')


# Customize titles, labels, and ticks for the combined plot
N_lp = len(df_lp)
N_tvp = len(df_tvp)
main_title = f'Cumulative Incidence of Complications and Death\nLow BSA LP (N={N_lp}) vs. TVP (N={N_tvp})'
ax.set_title(label=main_title, loc='center', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Time (years)', fontsize=14)
ax.set_ylabel('Cumulative Incidence', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left', fontsize=14)
ax.set_ylim(0, 1.0)
ax.set_xlim(left=0)
ax.grid(True, which='both', linestyle=':', linewidth=0.6)

# Save the combined plot to a file
plt.tight_layout()
plt.savefig("cif_combined_plot.png")
plt.show()
