import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load your CSV file
# NOTE: Replace "playground/Micra/LPBaseline.csv" with the correct path to your file.
try:
    df = pd.read_csv("playground/Micra/LPBaseline.csv")
except FileNotFoundError:
    print("Error: The file 'LPBaseline.csv' was not found. Please check the file path.")
    # Exiting the script as we can't proceed without the data.
    exit()

def calculate_or_with_ci(df, outcome_var, predictor_var):
    """
    Calculates the Odds Ratio (OR) and its 95% Confidence Interval (CI)
    for a given binary outcome and binary predictor using logistic regression.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        outcome_var (str): The name of the binary outcome variable (e.g., 'AcuteCom').
        predictor_var (str): The name of the binary predictor variable (e.g., 'Sex').

    Returns:
        tuple: A tuple containing the OR, lower CI, and upper CI.
    """
    # 1. Prepare the data: Drop rows with missing values in the relevant columns.
    data = df.dropna(subset=[outcome_var, predictor_var])

    # 2. Define the logistic regression formula
    formula = f"{outcome_var} ~ {predictor_var}"

    # 3. Fit the logistic regression model
    model = smf.logit(formula=formula, data=data).fit()

    # 4. Extract the results for the predictor variable
    # We are interested in the coefficient (log-odds) and its confidence interval.
    # The [1] index is for the predictor variable, as [0] is the intercept.
    
    # Get the exponentiated coefficient, which is the Odds Ratio
    or_val = np.exp(model.params[1])

    # Get the 95% confidence intervals and exponentiate them to get the OR's CI
    ci = np.exp(model.conf_int().iloc[1])
    
    return or_val, ci[0], ci[1]

# --- Example Usage ---
# Assume 'AcuteCom' and 'Sex' are binary variables (0 or 1).
# If your variables are not 0/1, you might need to convert them first.
# For example: df['Male_dummy'] = (df['Sex'] == 'Male').astype(int)

# Set the outcome and the predictor you want to analyze
outcome_var = "AcuteCom"
predictor_var = "Sex" # Change this to any other variable you want to analyze

# Calculate the OR and 95% CI
or_value, lower_ci, upper_ci = calculate_or_with_ci(df, outcome_var, predictor_var)

# Print the results in a formatted string
print(f"Odds Ratio for {predictor_var} and {outcome_var}:")
print(f"OR = {or_value:.2f} (95% CI: {lower_ci:.2f} - {upper_ci:.2f})")