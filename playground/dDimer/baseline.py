import pandas as pd
import numpy as np

def generate_baseline_table(csv_file_path):
    # 1. Load the dataset
    df = pd.read_csv('playground/dDimer/dDimer.csv')
    
    # 2. Define your variable types
    # You can adjust these lists based on your specific needs
    categorical_cols = [
        'Gender', 'procedure', 'Thrombus', 'typeAF', 'OAC', 
        'HTN', 'IHD', 'CVA', 'HF', 'DM', 'Shape', 'Complication'
    ]
    
    continuous_cols = [
        'age', 'dDimer', 'CHADVA', 'HASBLED', 'BMI', 
        'creatinine', 'GFR', 'LAVI', 'LAdiameter', 'LVEF', 'PW'
    ]
    
    results = []

    # 3. Process Continuous Variables
    for col in continuous_cols:
        if col in df.columns:
            # Drop missing values for accurate stats
            valid_data = df[col].dropna()
            
            mean_val = valid_data.mean()
            std_val = valid_data.std()
            median_val = valid_data.median()
            q1 = valid_data.quantile(0.25)
            q3 = valid_data.quantile(0.75)
            
            # Format: Mean ± SD
            results.append({
                'Variable': col,
                'Category': '',
                'Statistic': f"{mean_val:.2f} ± {std_val:.2f}",
                'Type': 'Mean ± SD'
            })
            # Format: Median (IQR) - often better for skewed data like dDimer
            results.append({
                'Variable': col,
                'Category': '',
                'Statistic': f"{median_val:.2f} ({q1:.2f}-{q3:.2f})",
                'Type': 'Median (IQR)'
            })

    # 4. Process Categorical Variables
    for col in categorical_cols:
        if col in df.columns:
            # Get counts for each category
            counts = df[col].value_counts().sort_index()
            total = len(df[col].dropna())
            
            for category, count in counts.items():
                perc = (count / total) * 100
                results.append({
                    'Variable': col,
                    'Category': str(category),
                    'Statistic': f"{count} ({perc:.1f}%)",
                    'Type': 'n (%)'
                })

    # 5. Create DataFrame and display
    results_df = pd.DataFrame(results)
    return results_df

# Execute the function
# Replace 'dDimer.csv' with your actual file path if different
table_one = generate_baseline_table('dDimer.csv')

# Print the table
print(table_one.to_string())

# Optional: Save to CSV
# table_one.to_csv('baseline_characteristics_output.csv', index=False)