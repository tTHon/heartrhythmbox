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

def create_or_table(df, outcome_var, predictor_vars):
    """
    Calculates and returns a DataFrame of Odds Ratios, 95% CIs, and p-values
    for a list of predictors against a binary outcome.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        outcome_var (str): The name of the binary outcome variable (e.g., 'AcuteCom').
        predictor_vars (list): A list of predictor variable names.

    Returns:
        pd.DataFrame: A DataFrame with the OR, 95% CI, and p-value for each predictor.
    """
    results = []
    
    for predictor in predictor_vars:
        # Check if the predictor exists in the DataFrame
        if predictor not in df.columns:
            results.append({
                'Variable': predictor,
                'OR': np.nan,
                'Lower CI': np.nan,
                'Upper CI': np.nan,
                'P-value': np.nan
            })
            continue

        # Drop rows with missing values for the current analysis
        data = df.dropna(subset=[outcome_var, predictor])

        # Define the logistic regression formula
        formula = f"{outcome_var} ~ {predictor}"

        # Fit the logistic regression model
        try:
            model = smf.logit(formula=formula, data=data).fit(disp=False)
            
            # Extract results for the predictor (excluding the Intercept)
            or_val = np.exp(model.params[predictor])
            ci = np.exp(model.conf_int().loc[predictor])
            p_value = model.pvalues[predictor]
            
            results.append({
                'Variable': predictor,
                'OR': or_val,
                'Lower CI': ci[0],
                'Upper CI': ci[1],
                'P-value': p_value
            })
        except Exception as e:
            results.append({
                'Variable': predictor,
                'OR': np.nan,
                'Lower CI': np.nan,
                'Upper CI': np.nan,
                'P-value': np.nan
            })

    return pd.DataFrame(results).round(4)

# --- Example Usage ---
# Define your outcome and a list of all your predictor variables
outcome_var = "composite"
# Include a mix of categorical and continuous variables as needed
predictor_vars = ["Age", "CKD", "CCI", "BSAYu"] 

# Create the final table
final_table = create_or_table(df, outcome_var, predictor_vars)

# Print the table in markdown format for a clean output
print("Univariate Odds Ratios and 95% CI Table:")
print(final_table.to_markdown(index=False))