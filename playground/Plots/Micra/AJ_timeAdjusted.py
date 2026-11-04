import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import AalenJohansenFitter
import sys

# Load the dataset
file_path = 'playground/Micra/lowBSA.csv'
try:
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"ERROR: The file '{file_path}' was not found.")
    sys.exit()

# Drop rows with missing T2Events or AnyEvents
df.dropna(subset=['T2Events', 'AnyEvents'], inplace=True)

# Filter for matched cohort
lp_lowbsa_match_ids = df[(df['Type'] == 1)]['MatchID'].unique()
df_matched = df[df['MatchID'].isin(lp_lowbsa_match_ids)].copy()
print(f"Matched cohort: {df_matched.shape[0]} records, {len(lp_lowbsa_match_ids)} matched sets")

# PART 1: PREPARE DATA FOR TIME-TO-EVENT ANALYSIS
# ============================================================================
outcome = np.zeros(len(df_matched))
outcome[df_matched['Complication'] == 1] = 1
outcome[(df_matched['Death'] == 1) & (df_matched['Complication'] == 0)] = 2
df_matched['status'] = outcome.astype(int)
df_matched['duration'] = df_matched['T2Events']

# Separate data for each group
df_lp = df_matched[df_matched['Type'] == 1].copy()
df_tvp = df_matched[df_matched['Type'] == 0].copy()

# PART 2: AALEN-JOHANSEN FITTING
# ============================================================================
ajf_comp_lp = AalenJohansenFitter()
ajf_lp_death = AalenJohansenFitter()
ajf_comp_tvp = AalenJohansenFitter()
ajf_tvp_death = AalenJohansenFitter()

ajf_comp_lp.fit(df_lp['duration'], df_lp['status'], event_of_interest=1)
ajf_lp_death.fit(df_lp['duration'], df_lp['status'], event_of_interest=2)
ajf_comp_tvp.fit(df_tvp['duration'], df_tvp['status'], event_of_interest=1)
ajf_tvp_death.fit(df_tvp['duration'], df_tvp['status'], event_of_interest=2)

# ADJUSTED: Determine the maximum follow-up time from the TVP group
# ============================================================================
max_tvp_time_days = df_tvp['duration'].max()
max_tvp_time_years = max_tvp_time_days / 365.25
print(f"\nAdjusting all plots to TVP maximum follow-up time: {max_tvp_time_years:.2f} years.")

# Set plotting theme
sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#FDFDFD", "grid.linestyle": ":"})
color_complication = sns.color_palette("Set2", 2)[0]
color_death = sns.color_palette("Set2", 2)[1]

# PART 3: PLOT FOR LP GROUP (ADJUSTED TO TVP TIME)
# ============================================================================
cif_results_lp = pd.DataFrame({
    'time': ajf_comp_lp.cumulative_density_.index / 365.25,
    'cif_complication': ajf_comp_lp.cumulative_density_.iloc[:, 0].values,
    'cif_death': ajf_lp_death.cumulative_density_.iloc[:, 0].values
})

fig, ax = plt.subplots(figsize=(15, 8))
ax.step(cif_results_lp['time'], cif_results_lp['cif_complication'], where='post', label='Complication', color=color_complication)
ax.step(cif_results_lp['time'], cif_results_lp['cif_death'], where='post', label='Death', color=color_death)
ax.fill_between(cif_results_lp['time'], 0, cif_results_lp['cif_complication'], step='post', alpha=0.7, color=color_complication)
ax.fill_between(cif_results_lp['time'], cif_results_lp['cif_complication'], cif_results_lp['cif_complication'] + cif_results_lp['cif_death'], step='post', alpha=0.7, color=color_death)

N = len(df_lp)
ax.set_title(label=f'Cumulative Incidence for low BSA LP (N=25)', loc='center', fontsize=18, fontweight='bold', pad=10)
ax.set_xlabel('Time (years)*', fontsize=16)
ax.set_ylabel('Cumulative Incidence', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left', fontsize=16)
ax.set_ylim(0, 1.0)
ax.set_xlim(0, max_tvp_time_years) # ADJUSTED: Set x-axis limit

# ADJUSTED: Annotation at the new time limit
lp_annotation_df = cif_results_lp[cif_results_lp['time'] <= max_tvp_time_years]
max_time_adjusted = lp_annotation_df['time'].max()
final_comp_inc = lp_annotation_df.loc[lp_annotation_df['time'] == max_time_adjusted, 'cif_complication'].values[0]
final_death_inc = lp_annotation_df.loc[lp_annotation_df['time'] == max_time_adjusted, 'cif_death'].values[0]

textstr = (f"Cumulative Incidence at {max_time_adjusted:.1f} years*:\n"
           f"Complication: {final_comp_inc:.2%}\n"
           f"Death: {final_death_inc:.2%}")
props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='lightgray')
ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=16, verticalalignment='top', horizontalalignment='right', bbox=props)
plt.tight_layout()
#plt.savefig("cif_lp_plot_adjusted.png")
plt.show()
plt.close()
print("Generated cif_lp_plot_adjusted")

# PART 4: PLOT FOR TVP GROUP
# ============================================================================
cif_results_tvp = pd.DataFrame({
    'time': ajf_comp_tvp.cumulative_density_.index / 365.25,
    'cif_complication': ajf_comp_tvp.cumulative_density_.iloc[:, 0].values,
    'cif_death': ajf_tvp_death.cumulative_density_.iloc[:, 0].values
})

fig, ax = plt.subplots(figsize=(15, 8))
ax.step(cif_results_tvp['time'], cif_results_tvp['cif_complication'], where='post', label='Complication', color=color_complication)
ax.step(cif_results_tvp['time'], cif_results_tvp['cif_death'], where='post', label='Death', color=color_death)
ax.fill_between(cif_results_tvp['time'], 0, cif_results_tvp['cif_complication'], step='post', alpha=0.7, color=color_complication)
ax.fill_between(cif_results_tvp['time'], cif_results_tvp['cif_complication'], cif_results_tvp['cif_complication'] + cif_results_tvp['cif_death'], step='post', alpha=0.7, color=color_death)

N = len(df_tvp)
ax.set_title(label=f'Cumulative Incidence for TVP (N=75)', loc='center', fontsize=18, fontweight='bold', pad=10)
ax.set_xlabel('Time (years)', fontsize=16)
ax.set_ylabel('Cumulative Incidence', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left', fontsize=14)
ax.set_ylim(0, 1.0)
ax.set_xlim(0, max_tvp_time_years) # ADJUSTED: Set x-axis limit (ensures consistency)

max_time = cif_results_tvp['time'].max()
final_comp_inc = cif_results_tvp.loc[cif_results_tvp['time'] == max_time, 'cif_complication'].values[0]
final_death_inc = cif_results_tvp.loc[cif_results_tvp['time'] == max_time, 'cif_death'].values[0]
textstr = (f"Cumulative Incidence at {max_time:.1f} years:\n"
           f"Complication: {final_comp_inc:.2%}\n"
           f"Death: {final_death_inc:.2%}")
props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='lightgray')
ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=16, verticalalignment='top', horizontalalignment='right', bbox=props)
plt.tight_layout()
#plt.savefig("cif_tvp_plot_adjusted.png")
plt.show()
plt.close()
print("Generated cif_tvp_plot_adjusted")

# PART 5: COMBINED PLOT (ADJUSTED TO TVP TIME)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 8))
colors_lp = sns.color_palette("Blues_d", 2)
colors_tvp = sns.color_palette("Reds_d", 2)

ax.step(cif_results_lp['time'], cif_results_lp['cif_complication'], where='post', label='LP - Complication', color=colors_lp[0], linestyle='-')
ax.step(cif_results_lp['time'], cif_results_lp['cif_death'], where='post', label='LP - Death', color=colors_lp[1], linestyle='-')
ax.step(cif_results_tvp['time'], cif_results_tvp['cif_complication'], where='post', label='TVP - Complication', color=colors_tvp[0], linestyle='--')
ax.step(cif_results_tvp['time'], cif_results_tvp['cif_death'], where='post', label='TVP - Death', color=colors_tvp[1], linestyle='--')

N_lp = len(df_lp)
N_tvp = len(df_tvp)
ax.set_title(label=f'Cumulative Incidence of Complications and Death\nLow BSA LP (N={N_lp}) vs. TVP (N={N_tvp})', loc='center', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Time (years)', fontsize=14)
ax.set_ylabel('Cumulative Incidence', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left', fontsize=14)
ax.grid(True, which='both', linestyle=':', linewidth=0.6)
ax.set_ylim(0, 1.0)
ax.set_xlim(0, max_tvp_time_years) # ADJUSTED: Set x-axis limit

#plt.tight_layout()
#plt.savefig("cif_combined_plot_adjusted.png")
#plt.show()
#plt.close()
#print("Generated cif_combined_plot_adjusted.png")