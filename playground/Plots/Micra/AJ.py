import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_aalen_johansen(data):
    """
    Calculates the Aalen-Johansen estimator for cumulative incidence functions
    in a competing risks setting.

    Args:
        data (pd.DataFrame): DataFrame with 'time' and 'event' columns.
                             'event' codes: 0=censored, 1=complication, 2=death.

    Returns:
        pd.DataFrame: A DataFrame containing the cumulative incidence for each event type.
    """
    # Sort data by time, which is essential for the calculation
    df = data.sort_values(by='time').reset_index(drop=True)

    # Find unique times where an event (1 or 2) occurred
    event_times = sorted(df[df['event'] > 0]['time'].unique())

    # Initialize variables
    overall_survival = 1.0
    cif_1 = 0.0  # Cumulative incidence for event 1 (complication)
    cif_2 = 0.0  # Cumulative incidence for event 2 (death)

    results = [{'time': 0, 'cif_complication': 0, 'cif_death': 0}]

    # Iterate through each unique event time
    for t in event_times:
        at_risk = (df['time'] >= t).sum()
        d1 = ((df['time'] == t) & (df['event'] == 1)).sum()
        d2 = ((df['time'] == t) & (df['event'] == 2)).sum()
        
        if at_risk > 0:
            # Update CIFs with the survival probability *before* this time point
            cif_1 += overall_survival * (d1 / at_risk)
            cif_2 += overall_survival * (d2 / at_risk)
            
            # Update overall survival for the *next* time point
            overall_survival *= (1 - ((d1 + d2) / at_risk))

        results.append({'time': t, 'cif_complication': cif_1, 'cif_death': cif_2})

    return pd.DataFrame(results)

# --- Main Script ---

# 1. Load Data from CSV File
# IMPORTANT: Replace 'my_data.csv' with the actual name of your CSV file.
file_path = 'playground/Micra/LP_events_NA.csv' 

try:
    df_from_csv = pd.read_csv(file_path)

    # Rename columns to the standard 'time' and 'event' used in the function
    # T2Event -> time
    # AnyEvent -> event
    df_competing_risks = df_from_csv.rename(columns={
        'T2Events': 'time',
        'AnyEvents': 'event'
    })

    print(f"Successfully loaded data from '{file_path}'.")
    print("Found {} records.".format(len(df_competing_risks)))
    print("-" * 30)

    # 2. Calculate the cumulative incidence functions
    cif_results = calculate_aalen_johansen(df_competing_risks)

    # 3. Display the results table
    print("Aalen-Johansen Cumulative Incidence Functions")
    print(cif_results.round(4))
    print("-" * 30)


    # 4. Plot the results
    # Set a professional theme
    sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#FDFDFD", "grid.linestyle": ":"})
    colors = sns.color_palette("Set2", 2)
    color_complication = colors[0]
    color_death = colors[1]

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
    N = len(df_competing_risks)
    main_title = f'Cumulative Incidence of Complications and Death for all Leadless Pacemaker Patients (N={N})'
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
    plt.show()

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    print("Please make sure your CSV file is in the same directory as the script or provide the full file path.")
except KeyError:
    print(f"Error: The CSV file '{file_path}' must contain the columns 'T2Event' and 'AnyEvent'.")