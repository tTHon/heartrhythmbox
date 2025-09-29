import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load your CSV file
# NOTE: Make sure your file path is correct.
try:
    df = pd.read_csv("playground/Micra/LPBaseline.csv")
except FileNotFoundError:
    print("Error: The file 'LPBaseline.csv' was not found. Please check the file path.")
    exit()

def calculate_or_with_ci(df, outcome_var, predictor_var):
    """
    Calculates the Odds Ratio (OR), its 95% Confidence Interval (CI), and p-value
    for a given binary outcome and binary predictor using logistic regression.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        outcome_var (str): The name of the binary outcome variable (e.g., 'AcuteCom').
        predictor_var (str): The name of the binary predictor variable (e.g., 'Sex').

    Returns:
        dict: A dictionary containing the OR, lower CI, upper CI, and p-value.
    """
    data = df.dropna(subset=[outcome_var, predictor_var])
    formula = f"{outcome_var} ~ {predictor_var}"
    model = smf.logit(formula=formula, data=data).fit(disp=False)

    # Use .iloc[1] to access by position, avoiding the future warning
    or_val = np.exp(model.params.iloc[1])
    ci = np.exp(model.conf_int().iloc[1])
    p_value = model.pvalues.iloc[1]
    
    return {
        'Variable': predictor_var,
        'OR': or_val,
        'Lower CI': ci[0],
        'Upper CI': ci[1],
        'P-value': p_value
    }

# --- Main Analysis ---
# List of predictors to analyze
predictors_to_analyze = ["Sex","CCISev","lowBSA", "MI","PCI/CABG","CHF","PAD","CVA","Dementia","COPD","CNT","PU","Liver","DM","CKD","Malignancy","TV","SigValve","AF","HTN","CCISev","Antiplatelet","OAC","Access","Position","Model","Hemostasis"]
outcome_var = "AcuteCom"  # Binary outcome variable
results_list = []

for predictor in predictors_to_analyze:
    try:
        # Calculate the OR, CI, and p-value for each predictor
        result = calculate_or_with_ci(df, outcome_var, predictor)
        results_list.append(result)
    except Exception as e:
        print(f"Skipping predictor '{predictor}' due to error: {e}")

# Create the final DataFrame from the list of results
final_table = pd.DataFrame(results_list)

# Format the output for better readability
final_table = final_table.set_index('Variable').round(4)

print("Univariate Odds Ratios and 95% CI Table:")
print(final_table.to_markdown())