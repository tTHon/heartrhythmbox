import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency
import numpy as np

def analyze_groups(file_path, group_variable, continuous_vars, categorical_vars):
    """
    Reads a CSV file, performs statistical tests to compare two groups,
    and prints a summary table.

    Args:
        file_path (str): The path to the CSV file.
        group_variable (str): The name of the column that defines the two groups.
        continuous_vars (list): A list of names of the continuous variables.
        categorical_vars (list): A list of names of the categorical variables.
    """
    try:
        # 1. Read the CSV file
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    # --- ADDED CODE STARTS HERE ---
    
    # Remove rows where the grouping variable itself is missing (NaN)
    initial_rows = len(df)
    df.dropna(subset=[group_variable], inplace=True)
    
    # Optionally, inform the user if rows were dropped
    if len(df) < initial_rows:
        print(f"Note: Removed {initial_rows - len(df)} rows due to missing values in the '{group_variable}' column.")

    # --- ADDED CODE ENDS HERE ---

    # Ensure the grouping variable has only two unique values
    if df[group_variable].nunique() != 2:
        print(f"Error: The grouping variable '{group_variable}' must have exactly two unique groups.")
        print(f"Found groups: {df[group_variable].unique()}")
        return
        
    # Get the names of the two groups (e.g., 0 and 1)
    group1_name, group2_name = df[group_variable].unique()

    # Separate data into two groups
    group1 = df[df[group_variable] == group1_name]
    group2 = df[df[group_variable] == group2_name]

    results = []

    # 2. Run statistical tests
    # --- Process Continuous Variables ---
    print(f"Comparing Group '{group1_name}' (n={len(group1)}) vs. Group '{group2_name}' (n={len(group2)})")
    print("-" * 50)
    
    for var in continuous_vars:
        # Extract data for each group, dropping missing values
        g1_data = group1[var].dropna()
        g2_data = group2[var].dropna()

        # Calculate median and IQR for summary
        median1 = g1_data.median()
        q1_1, q3_1 = g1_data.quantile(0.25), g1_data.quantile(0.75)
        summary1 = f"{median1:.2f} ({q1_1:.2f} - {q3_1:.2f})"

        median2 = g2_data.median()
        q1_2, q3_2 = g2_data.quantile(0.25), g2_data.quantile(0.75)
        summary2 = f"{median2:.2f} ({q1_2:.2f} - {q3_2:.2f})"

        # Perform Mann-Whitney U test
        stat, p_value = mannwhitneyu(g1_data, g2_data)
        
        results.append({
            'Characteristic': var,
            f'Group {group1_name}': summary1,
            f'Group {group2_name}': summary2,
            'P-Value': f"{p_value:.8f}",
            'Test Used': 'Mann-Whitney U'
        })

    # --- Process Categorical Variables ---
    for var in categorical_vars:
        # Create a contingency table
        contingency_table = pd.crosstab(df[group_variable], df[var])
        
        # Perform Chi-squared test
        chi2, p_value, _, _ = chi2_contingency(contingency_table)

        # Get summaries (count and percentage) for each category in the variable
        for category in contingency_table.columns:
            # Group 1 summary
            count1 = contingency_table.loc[group1_name, category]
            percent1 = (count1 / len(group1)) * 100
            summary1 = f"{count1} ({percent1:.1f}%)"

            # Group 2 summary
            count2 = contingency_table.loc[group2_name, category]
            percent2 = (count2 / len(group2)) * 100
            summary2 = f"{count2} ({percent2:.1f}%)"

            results.append({
                'Characteristic': f"{var} ({category})",
                f'Group {group1_name}': summary1,
                f'Group {group2_name}': summary2,
                'P-Value': f"{p_value:.6f}",
                'Test Used': 'Chi-Squared'
            })

    # 3. Print the results table
    if results:
        results_df = pd.DataFrame(results)
        print(results_df.to_markdown(index=False))
    else:
        print("No variables were analyzed.")


# Define your file path and variables
# In a real scenario, you would use a file path like: file_path = 'my_data.csv'
file_path = "playground/Micra/LP_events.csv" 
group_variable = 'lowBSA'
continuous_vars = []
categorical_vars = ['MajCom30D','Death','ChronicCom']

# Run the analysis
analyze_groups(file_path, group_variable, continuous_vars, categorical_vars)