import pandas as pd
import numpy as np
from lifelines import CoxPHFitter

# ----------------------------------------------------
# 1. DATA LOADING AND PREPROCESSING
# ----------------------------------------------------
try:
    df = pd.read_csv('playground/Micra/LPBaseline.csv')
except FileNotFoundError:
    print("Error: The file 'playground/Micra/LP_events.csv' was not found.")
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
        len(var_data.unique()) <= 10  # Assuming ≤10 unique values means categorical
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
    'Age',           # Example continuous variable
    'BSA',           # Example continuous variable
    'lowBSA',        # Example binary categorical variable
    'Sex',           # Example binary categorical variable
    'CKD',         # Example multi-level categorical variable
    'CCI','CCISev','HTN','MI','CHF','AF','Dementia','PAD','Position'
]

print(f"\nVariables to analyze: {variables_to_analyze}")

# Check which variables exist in the dataset
existing_variables = [var for var in variables_to_analyze if var in df.columns]
missing_variables = [var for var in variables_to_analyze if var not in df.columns]

print(f"Variables found in dataset: {existing_variables}")
if missing_variables:
    print(f"Variables NOT found in dataset: {missing_variables}")

# ----------------------------------------------------
# 5. UNIVARIATE CAUSE-SPECIFIC ANALYSIS LOOP
# ----------------------------------------------------
# Create an empty list to store the results from each model
all_results = []
failed_analyses = []

print("\n" + "="*80)
print("## Running Univariate Cause-Specific Models for Complications ##")
print("="*80)

for variable in existing_variables:  # Only analyze existing variables
    print(f"\n--- Analyzing Risk Factor: {variable} ---")
    
    # Check for missing data
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

    # Get descriptive statistics for this variable
    descriptive_stats = get_descriptive_stats(analysis_df, variable)
    print(f"Descriptive stats: {descriptive_stats}")
    
    try:
        cph = CoxPHFitter()
        cph.fit(analysis_df,
                duration_col='duration',
                event_col='event_for_comp_model',
                formula=variable)

        print(f"Model fitted successfully. Summary shape: {cph.summary.shape}")
        
        # For multi-level categorical variables, lifelines creates multiple rows.
        # This code will add a result for each level compared to the baseline.
        for idx, row in cph.summary.iterrows():
            result = {
                'Risk Factor': idx,
                'Descriptive Statistics': descriptive_stats,
                'N': len(analysis_df),
                'Events': analysis_df['event_for_comp_model'].sum(),
                'Hazard Ratio (HR)': row['exp(coef)'],
                # CORRECTED: Changed '0.95' to '95%' to match the new lifelines version
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

# Report failed analyses
if failed_analyses:
    print(f"\nFailed analyses: {len(failed_analyses)}")
    for failure in failed_analyses:
        print(f"  - {failure['variable']}: {failure['reason']}")

# Debug: Print all collected results before processing
print(f"\nDEBUG: All collected results:")
for i, result in enumerate(all_results):
    print(f"Result {i+1}: {result['Risk Factor']}")

# ----------------------------------------------------
# 6. COMBINE AND PRINT THE FINAL RESULTS TABLE
# ----------------------------------------------------
print("\n" + "="*100)
print("### Consolidated Summary of Univariate Cause-Specific Analyses with Descriptive Statistics ###")
print("="*100)

if not all_results:
    print("No results were generated.")
    print("DEBUG: Possible reasons:")
    print("1. All variables missing from dataset")
    print("2. All variables have no events (complications)")
    print("3. All models failed to converge")
    print("4. Data preprocessing removed all observations")
else:
    print(f"Processing {len(all_results)} results...")
    
    # Convert the list of dictionaries to a pandas DataFrame
    results_df = pd.DataFrame(all_results)
    
    print(f"Results DataFrame shape: {results_df.shape}")
    print(f"Results DataFrame columns: {results_df.columns.tolist()}")
    
    # Check for any missing values in key columns
    print("Checking for missing values in results:")
    print(results_df.isnull().sum())
    
    try:
        # Format the DataFrame for better readability
        results_df['Hazard Ratio (HR)'] = pd.to_numeric(results_df['Hazard Ratio (HR)'], errors='coerce').round(3)
        results_df['95% CI Lower'] = pd.to_numeric(results_df['95% CI Lower'], errors='coerce')
        results_df['95% CI Upper'] = pd.to_numeric(results_df['95% CI Upper'], errors='coerce')
        results_df['95% CI'] = results_df.apply(lambda row: f"({row['95% CI Lower']:.3f} - {row['95% CI Upper']:.3f})" 
                                              if pd.notna(row['95% CI Lower']) and pd.notna(row['95% CI Upper']) 
                                              else "N/A", axis=1)
        results_df['p-value'] = pd.to_numeric(results_df['p-value'], errors='coerce').round(4)
        
        # Drop the now redundant lower and upper CI columns
        results_df.drop(columns=['95% CI Lower', '95% CI Upper'], inplace=True)
        
        # Reorder columns for better presentation
        available_columns = results_df.columns.tolist()
        desired_order = ['Risk Factor', 'Descriptive Statistics', 'N', 'Events', 'Hazard Ratio (HR)', '95% CI', 'p-value']
        column_order = [col for col in desired_order if col in available_columns]
        results_df = results_df[column_order]
        
        print(f"Final results DataFrame shape: {results_df.shape}")
        
        # Print the table with better formatting
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 80)
        pd.set_option('display.max_rows', None)

        # Select and reorder columns for the final table
        #final_columns = ['Risk Factor', 'Descriptive Statistics', 'N', 'Events', 'Hazard Ratio (HR)', '95% CI', 'p-value']
        final_columns = ['Risk Factor', 'Hazard Ratio (HR)', '95% CI', 'p-value']
        results_df = results_df[final_columns]

        # --- NEW CODE FOR MARKDOWN OUTPUT ---
        # Use the .to_markdown() method to print a clean table
        print(results_df.to_markdown(index=False))
            
        # Option 1: Print without setting index (to see all rows clearly)
        #print("\n### RESULTS TABLE (All Rows) ###")
        #print(results_df.to_string(index=False))
        
        # Option 2: Also print with index for comparison
        #print("\n### RESULTS TABLE (With Risk Factor as Index) ###")
        #results_df_indexed = results_df.set_index('Risk Factor')
        #print(results_df_indexed.to_string())
        
    except Exception as e:
        print(f"ERROR in formatting results: {e}")
        print("Raw results DataFrame:")
        print(results_df)

# ----------------------------------------------------
# 7. ADDITIONAL SUMMARY STATISTICS
# ----------------------------------------------------
print("\n" + "="*100)
print("### Overall Dataset Summary ###")
print("="*100)

total_patients = len(df)
total_events = df['event_for_comp_model'].sum()
event_rate = (total_events / total_patients) * 100

print(f"Total patients in analysis: {total_patients}")
print(f"Total complications (events): {total_events}")
print(f"Complication rate: {event_rate:.1f}%")

if 'duration' in df.columns:
    median_followup = df['duration'].median()
    iqr_followup = f"{df['duration'].quantile(0.25):.1f}-{df['duration'].quantile(0.75):.1f}"
    print(f"Median follow-up time: {median_followup:.1f} (IQR: {iqr_followup})")

print("\n" + "="*100)
print("Analysis complete.")