import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import AalenJohansenFitter

# ============================================================================
# PART 0: Data Preparation
# ============================================================================
# Load the data
df = pd.read_csv('playground/Micra/LP_events.csv')

# Data cleaning and preparation
df.dropna(subset=['T2Events', 'AnyEvents'], inplace=True)
df['T2Events'] = pd.to_numeric(df['T2Events'], errors='coerce')
df['AnyEvents'] = df['AnyEvents'].astype(int)
df['status'] = df['AnyEvents'].apply(lambda x: 1 if x == 1 else 2 if x == 2 else 0)
df['duration'] = df['T2Events']
df.dropna(subset=['duration', 'status'], inplace=True)

# *** CREATE DURATION IN YEARS ***
df['duration_years'] = df['duration'] / 365.25

# Separate into groups
df_low_bsa = df[df['lowBSA'] == 1].copy()
df_other_bsa = df[df['lowBSA'] == 0].copy()

# ============================================================================
# Corrected Helper Function to Build CIF DataFrame
# ============================================================================
def build_cif_dataframe(duration, status):
    """
    Builds a clean, aligned DataFrame for plotting stacked cumulative incidence,
    robustly handling cases with zero events.
    """
    # Fit once for event 1 (Complication)
    ajf_event1 = AalenJohansenFitter().fit(duration, status, event_of_interest=1)
    # Fit once for event 2 (Death)
    ajf_event2 = AalenJohansenFitter().fit(duration, status, event_of_interest=2)

    # Extract cumulative incidence series; creates an empty series if no events occurred
    cif1_raw = ajf_event1.cumulative_density_.iloc[:, 0] if not ajf_event1.cumulative_density_.empty else pd.Series(dtype=float)
    cif2_raw = ajf_event2.cumulative_density_.iloc[:, 0] if not ajf_event2.cumulative_density_.empty else pd.Series(dtype=float)

    # Create a unified timeline that contains all time points from both event types
    timeline = pd.Index([0]).union(cif1_raw.index).union(cif2_raw.index)

    # Reindex both series to the unified timeline, forward-filling values
    cif1_aligned = cif1_raw.reindex(timeline, method='ffill').fillna(0)
    cif2_aligned = cif2_raw.reindex(timeline, method='ffill').fillna(0)

    # Combine into a single, perfectly aligned DataFrame
    df_cif = pd.DataFrame({'Complication': cif1_aligned, 'Death': cif2_aligned})
    return df_cif

# --- Build the dataframes for each group ---
cif_low_bsa = build_cif_dataframe(df_low_bsa['duration_years'], df_low_bsa['status'])
cif_other_bsa = build_cif_dataframe(df_other_bsa['duration_years'], df_other_bsa['status'])
cif_all_lp = build_cif_dataframe(df['duration_years'], df['status'])


# ============================================================================
# PLOTTING SECTION
# ============================================================================

# --- Reusable plotting function for consistent, eye-pleasing style ---
def create_stacked_plot(ax, cif_data, group_name, N):
    # Set a professional theme
    sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#FDFDFD", "grid.linestyle": ":"})
    colors = sns.color_palette("Set2", 2)
    color_complication = colors[0]
    color_death = colors[1]
    print(f"Colors used - Complication: {color_complication}, Death: {color_death}")

    # Plot the stacked areas
    ax.fill_between(cif_data.index, 0, cif_data['Complication'], color=color_complication, alpha=0.9, label='Complication')
    ax.fill_between(cif_data.index, cif_data['Complication'], cif_data['Complication'] + cif_data['Death'], color=color_death, alpha=0.9, label='Death')
    
    # Customize titles, labels, and ticks
    main_title = f'Cumulative Incidence of Complications and Death for all Leadless Pacemaker Patients (N={N})'
    ax.set_title(label=main_title, loc='center', fontsize=15, fontweight='bold', pad=10)
    ax.set_xlabel('Time (years)', fontsize=14)
    ax.set_ylabel('Cumulative Incidence', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left', fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(left=0)

    # *** ANNOTATION WITH REVISED TEXT ***
    max_time = cif_data.index.max()
    final_comp_inc = cif_data.loc[max_time, 'Complication']
    final_death_inc = cif_data.loc[max_time, 'Death']
    textstr = (f"Cumulative Incidence by Aalen-Johansen at {max_time:.1f} years:\n"
               f"Complication: {final_comp_inc:.2%}\n"
               f"Death: {final_death_inc:.2%}")
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='lightgray')
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=13,
            verticalalignment='top', horizontalalignment='right', bbox=props)

# --- Generate and show the three separate plots ---

# Plot 1: Low BSA Group
#fig1, ax1 = plt.subplots(figsize=(10, 7))
#create_stacked_plot(ax1, cif_low_bsa, "Low BSA Group", len(df_low_bsa))
#plt.tight_layout()
#plt.show()

# Plot 2: Other BSA Group
#fig2, ax2 = plt.subplots(figsize=(10, 7))
#create_stacked_plot(ax2, cif_other_bsa, "Other BSA Group", len(df_other_bsa))
#plt.tight_layout()
#plt.show()

# Plot 3: All LP Patients
fig3, ax3 = plt.subplots(figsize=(10, 7))
create_stacked_plot(ax3, cif_all_lp, "All LP Patients", len(df))
plt.tight_layout()
plt.show()