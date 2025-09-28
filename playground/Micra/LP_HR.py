import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 1. DATA LOADING AND PREPROCESSING
# ----------------------------------------------------
try:
    df = pd.read_csv('playground/Micra/LPBaseline.csv')
except FileNotFoundError:
    print("Error: The file 'playground/Micra/LPBaseline.csv' was not found.")
    exit()

# Standard preprocessing steps
df.dropna(subset=['T2Events', 'AnyEvents'], inplace=True)
df['T2Events'] = pd.to_numeric(df['T2Events'], errors='coerce')
df['AnyEvents'] = df['AnyEvents'].astype(int)
df['status'] = df['AnyEvents'].apply(lambda x: 1 if x == 1 else 2 if x == 2 else 0)
df['duration'] = df['T2Events']
df.dropna(subset=['duration', 'status'], inplace=True)

print("Data loaded and preprocessed successfully.")
print("-" * 50)

# ----------------------------------------------------
# 2. CAUSE-SPECIFIC DATA PREPARATION FOR COMPLICATIONS
# ----------------------------------------------------
# For the Cause-Specific model of complications, we treat death as a censored event.
df['event_for_comp_model'] = (df['status'] == 1).astype(int)

# ----------------------------------------------------
# 3. HELPER FUNCTION FOR DESCRIPTIVE STATISTICS
# ----------------------------------------------------
def get_descriptive_stats(data, variable):
    """
    Generate descriptive statistics for a variable.
    Returns appropriate statistics based on variable type.
    """
    var_data = data[variable].dropna()
    
    if var_data.empty:
        return "No data"
    
    # Check if variable is categorical (has few unique values or is object/category type)
    is_categorical = (
        var_data.dtype == 'object' or 
        var_data.dtype.name == 'category' or
        len(var_data.unique()) <= 10 
    )
    
    if is_categorical:
        # For categorical variables, show counts and percentages
        value_counts = var_data.value_counts().sort_index()
        total = len(var_data)
        
        desc_parts = []
        for value, count in value_counts.items():
            percentage = (count / total) * 100
            desc_parts.append(f"{value}: {count} ({percentage:.1f}%)")
        
        return "; ".join(desc_parts)
    
    else:
        # For continuous variables, show mean ± SD, median (IQR)
        mean_val = var_data.mean()
        std_val = var_data.std()
        median_val = var_data.median()
        q25 = var_data.quantile(0.25)
        q75 = var_data.quantile(0.75)
        
        return f"Mean: {mean_val:.2f} ± {std_val:.2f}; Median: {median_val:.2f} ({q25:.2f}-{q75:.2f})"

# ----------------------------------------------------
# 4. CHECK AVAILABLE COLUMNS AND PREPARE ANALYSIS
# ----------------------------------------------------
print("\n## Checking Available Columns ##")
print("Available columns in dataset:")
print(df.columns.tolist())

# **IMPORTANT**: Replace these with your actual variable names.
variables_to_analyze = [
    'Age',
    'BSA',
    'BMI',
    'Sex',
    'CCI',
    'CKD'
]

print(f"\nVariables to analyze: {variables_to_analyze}")

existing_variables = [var for var in variables_to_analyze if var in df.columns]
missing_variables = [var for var in variables_to_analyze if var not in df.columns]

print(f"Variables found in dataset: {existing_variables}")
if missing_variables:
    print(f"Variables NOT found in dataset: {missing_variables}")

# ----------------------------------------------------
# 5. UNIVARIATE CAUSE-SPECIFIC ANALYSIS LOOP
# ----------------------------------------------------
all_results = []
failed_analyses = []

print("\n" + "="*80)
print("## Running Univariate Cause-Specific Models for Complications ##")
print("="*80)

for variable in existing_variables:
    print(f"\n--- Analyzing Risk Factor: {variable} ---")
    
    total_rows = len(df)
    missing_count = df[variable].isna().sum()
    print(f"Total rows: {total_rows}, Missing values in {variable}: {missing_count}")
    
    analysis_df = df[[variable, 'duration', 'event_for_comp_model']].dropna()
    analysis_n = len(analysis_df)
    events_n = analysis_df['event_for_comp_model'].sum()
    
    print(f"Analysis sample size: {analysis_n}, Events: {events_n}")

    if analysis_df.empty:
        print(f"Warning: No data remaining after removing missing values for '{variable}'. Skipping.")
        failed_analyses.append({'variable': variable, 'reason': 'No data after removing missing values'})
        continue

    if events_n == 0:
        print(f"Warning: No events (complications) in the data for '{variable}'. Cannot fit Cox model.")
        failed_analyses.append({'variable': variable, 'reason': 'No events to analyze'})
        continue

    descriptive_stats = get_descriptive_stats(analysis_df, variable)
    print(f"Descriptive stats: {descriptive_stats}")
    
    try:
        if analysis_df[variable].dtype == 'object' or analysis_df[variable].dtype.name == 'category':
            analysis_df = pd.get_dummies(analysis_df, columns=[variable], drop_first=True)
            dummy_cols = [col for col in analysis_df.columns if col.startswith(f'{variable}_')]
            if len(dummy_cols) > 0:
                formula = dummy_cols[0]
                plot_name = f"{variable} ({dummy_cols[0]})"
            else:
                raise ValueError("No dummy variable created for the categorical variable.")
        else:
            formula = variable
            plot_name = variable
        
        cph = CoxPHFitter()
        cph.fit(analysis_df,
                duration_col='duration',
                event_col='event_for_comp_model',
                formula=formula)

        print(f"Model fitted successfully. Summary shape: {cph.summary.shape}")
        
        for idx, row in cph.summary.iterrows():
            result = {
                'Risk Factor': plot_name,
                'Descriptive Statistics': descriptive_stats,
                'N': len(analysis_df),
                'Events': analysis_df['event_for_comp_model'].sum(),
                'Hazard Ratio (HR)': row['exp(coef)'],
                '95% CI Lower': row['exp(coef) lower 95%'],
                '95% CI Upper': row['exp(coef) upper 95%'],
                'p-value': row['p']
            }
            all_results.append(result)
            print(f"Added result for: {idx}")

    except Exception as e:
        print(f"ERROR: Could not fit model for '{variable}'. Error: {e}")
        failed_analyses.append({'variable': variable, 'reason': str(e)})

print(f"\nCompleted analysis. Total results collected: {len(all_results)}")

if failed_analyses:
    print(f"\nFailed analyses: {len(failed_analyses)}")
    for failure in failed_analyses:
        print(f" - {failure['variable']}: {failure['reason']}")

print("\n" + "="*100)
print("### Consolidated Summary of Univariate Cause-Specific Analyses ###")
print("="*100)

if not all_results:
    print("No results were generated.")
else:
    results_df = pd.DataFrame(all_results)
    
    results_df['Hazard Ratio (HR)'] = pd.to_numeric(results_df['Hazard Ratio (HR)'], errors='coerce')
    results_df['95% CI Lower'] = pd.to_numeric(results_df['95% CI Lower'], errors='coerce')
    results_df['95% CI Upper'] = pd.to_numeric(results_df['95% CI Upper'], errors='coerce')
    results_df['p-value'] = pd.to_numeric(results_df['p-value'], errors='coerce')
    
    print("\n" + "="*100)
    print("### Creating Forest Plot for Visualization ###")
    print("="*100)

    plot_df = results_df.dropna(subset=['Hazard Ratio (HR)', '95% CI Lower', '95% CI Upper']).copy()
    plot_df = plot_df.sort_values(by='Hazard Ratio (HR)', ascending=True)

    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        line_color = '#1f77b4'
        marker_color = '#ff7f0e'
        ref_line_color = 'gray'
        
        ax.hlines(y=plot_df['Risk Factor'], xmin=plot_df['95% CI Lower'], xmax=plot_df['95% CI Upper'],
                  color=line_color, linestyle='-', linewidth=2, alpha=0.8, label='95% CI')
        
        ax.scatter(plot_df['Hazard Ratio (HR)'], plot_df['Risk Factor'], color=marker_color, zorder=3,
                   s=100, marker='o', edgecolors='black', linewidth=0.8, label='Hazard Ratio')
        
        ax.axvline(x=1, color=ref_line_color, linestyle='--', linewidth=1.5, alpha=0.7, label='No Effect (HR=1)')
        
        ax.set_xscale('log')
        
        x_ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(t) for t in x_ticks])

        ax.set_xlabel('Hazard Ratio (Log Scale)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Risk Factor', fontsize=12, fontweight='bold')
        ax.set_title('Forest Plot of Univariate Cox Regression Results', fontsize=16, fontweight='bold', pad=20)
        
        ax.grid(axis='x', linestyle=':', alpha=0.6)
        
        for i, row in plot_df.iterrows():
            hr_text = f"HR: {row['Hazard Ratio (HR)']:.2f}"
            p_text = f"p: {row['p-value']:.3f}"
            text = f"{hr_text}, {p_text}"
            ax.text(row['95% CI Upper'] * 1.05, row['Risk Factor'], text,
                    va='center', ha='left', fontsize=9, color='darkgreen')
        
        plt.tight_layout()
        plt.show()
        #plt.savefig('cox_forest_plot.png', dpi=300)
        #print("Forest plot has been generated and saved as 'cox_forest_plot.png'.")
    else:
        print("Could not generate plot. No valid data to plot.")

    # ----------------------------------------------------
    # 7. COMBINE AND PRINT THE FINAL RESULTS TABLE
    # ----------------------------------------------------
    results_df['95% CI'] = results_df.apply(lambda row: f"({row['95% CI Lower']:.3f} - {row['95% CI Upper']:.3f})", axis=1)
    
    final_columns = ['Risk Factor', 'Descriptive Statistics', 'N', 'Events', 'Hazard Ratio (HR)', '95% CI', 'p-value']
    results_df = results_df[final_columns]
    
    print("\n" + "="*100)
    print("### Final Results Table ###")
    print("="*100)
    print(results_df.to_markdown(index=False))

print("\n" + "="*100)
print("Analysis complete.")

# ----------------------------------------------------
# 8. COMPARISON OF RISK FACTORS BY EVENT STATUS
# ----------------------------------------------------
print("\n" + "="*100)
print("### Comparison of Risk Factors: With vs. Without Events ###")
print("="*100)

comparison_results = []

# Get the two groups
df_no_event = df[df['event_for_comp_model'] == 0]
df_event = df[df['event_for_comp_model'] == 1]

n_no_event = len(df_no_event)
n_event = len(df_event)
print(f"Total patients without events: {n_no_event}")
print(f"Total patients with events: {n_event}")
print("-" * 50)

for variable in existing_variables:
    # Ensure there's data for comparison
    if df_no_event[variable].isnull().all() or df_event[variable].isnull().all():
        comparison_results.append({
            'Risk Factor': variable,
            'No Event (N)': 'N/A',
            'Event (N)': 'N/A',
            'Value (No Event)': 'Insufficient Data',
            'Value (Event)': 'Insufficient Data'
        })
        continue
    
    # Check if the variable is categorical or continuous
    is_categorical = (
        df[variable].dtype == 'object' or
        len(df[variable].dropna().unique()) <= 10
    )

    if is_categorical:
        # Categorical data: count and percentage
        no_event_counts = df_no_event[variable].value_counts(normalize=True).mul(100).round(1).sort_index()
        event_counts = df_event[variable].value_counts(normalize=True).mul(100).round(1).sort_index()
        
        no_event_str = "; ".join([f"{v}: {c:.1f}% ({df_no_event[variable].value_counts()[v]})" for v, c in no_event_counts.items()])
        event_str = "; ".join([f"{v}: {c:.1f}% ({df_event[variable].value_counts()[v]})" for v, c in event_counts.items()])
        
        comparison_results.append({
            'Risk Factor': variable,
            'Value (No Event)': no_event_str,
            'Value (Event)': event_str,
            'No Event (N)': len(df_no_event[variable].dropna()),
            'Event (N)': len(df_event[variable].dropna())
        })
    else:
        # Continuous data: mean ± SD and median (IQR)
        no_event_mean = df_no_event[variable].mean()
        no_event_std = df_no_event[variable].std()
        no_event_median = df_no_event[variable].median()
        no_event_q25, no_event_q75 = df_no_event[variable].quantile([0.25, 0.75])
        
        event_mean = df_event[variable].mean()
        event_std = df_event[variable].std()
        event_median = df_event[variable].median()
        event_q25, event_q75 = df_event[variable].quantile([0.25, 0.75])
        
        no_event_str = f"Mean: {no_event_mean:.2f} ± {no_event_std:.2f}; Med: {no_event_median:.2f} ({no_event_q25:.2f}-{no_event_q75:.2f})"
        event_str = f"Mean: {event_mean:.2f} ± {event_std:.2f}; Med: {event_median:.2f} ({event_q25:.2f}-{event_q75:.2f})"
        
        comparison_results.append({
            'Risk Factor': variable,
            'Value (No Event)': no_event_str,
            'Value (Event)': event_str,
            'No Event (N)': len(df_no_event[variable].dropna()),
            'Event (N)': len(df_event[variable].dropna())
        })

# Create the final comparison DataFrame
comparison_df = pd.DataFrame(comparison_results)

# Print the final comparison table in Markdown format
print(comparison_df.to_markdown(index=False))

print("\n" + "="*100)
print("Comparison table created successfully.")