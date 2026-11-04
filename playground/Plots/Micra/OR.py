import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

# Load the data
# NOTE: Please replace 'your_data_file.csv' with the actual path to your CSV file
try:
    df = pd.read_csv('playground/Micra/LPBaseline.csv')
except FileNotFoundError:
    print("File not found. Please provide the correct file path.")
    exit()

# --- Data Preparation ---
# Define the outcome and predictor variables
outcome_var = 'AcuteCom'
predictor_vars = ['Age', 'Sex', 'CCI','BSA','lowBSA','HTN','BMI','CKD','AF','CVA','MI','CCISev','Position']

# Ensure the outcome variable is in the correct format
df['AcuteCom'] = df['AcuteCom'].astype(int)

# --- Loop through each predictor and run a separate regression ---
results_list = []

for predictor in predictor_vars:
    current_predictor_col = predictor

    # Handle all categorical predictors by converting them to numeric codes
    # NOTE: For nominal variables like 'Sex', this approach is not
    # statistically ideal compared to using pd.get_dummies, as it
    # treats categories as an ordered sequence (e.g., F=0, M=1).
    if df[predictor].dtype == 'object':
        df[predictor] = df[predictor].astype('category').cat.codes
    
    # Drop rows with any missing values
    df_clean = df.dropna(subset=[outcome_var, predictor])
    y = df_clean[outcome_var]
    X = df_clean[predictor]

    # Check for variance in predictor
    if X.nunique() > 1:
        X = sm.add_constant(X, prepend=False)
        try:
            model = sm.Logit(y, X).fit(disp=False)
            odds_ratio = np.exp(model.params[current_predictor_col])
            conf_int = np.exp(model.conf_int().loc[current_predictor_col])
            p_value = model.pvalues[current_predictor_col]
            
            results_list.append({
                'Predictor': predictor,
                'OR': odds_ratio,
                'CI_Lower': conf_int[0],
                'CI_Upper': conf_int[1],
                'p-value': p_value
            })
        except Exception:
            results_list.append({
                'Predictor': predictor,
                'OR': np.nan, 'CI_Lower': np.nan, 'CI_Upper': np.nan, 'p-value': np.nan
            })
    else:
        results_list.append({
            'Predictor': predictor,
            'OR': np.nan, 'CI_Lower': np.nan, 'CI_Upper': np.nan, 'p-value': np.nan
        })

# Create a DataFrame from the results list
results_df = pd.DataFrame(results_list)

# --- Create Professional Forest Plot ---
results_df_sorted = results_df.dropna(subset=['OR']).sort_values(by='OR', ascending=True)

# Filter out rows with NaN values for plotting
results_df_plot = results_df_sorted.dropna(subset=['OR', 'CI_Lower', 'CI_Upper'])

if not results_df_plot.empty:
    fig, ax = plt.subplots(figsize=(10, 8))
    
    line_color = '#1f77b4'
    marker_color = '#ff7f0e'
    ref_line_color = 'gray'
    font_size_labels = 12
    font_size_ticks = 10
    font_size_text = 9
    
    ax.hlines(y=results_df_plot['Predictor'], xmin=results_df_plot['CI_Lower'], xmax=results_df_plot['CI_Upper'],
              color=line_color, linestyle='-', linewidth=2, alpha=0.8, label='95% CI')
    
    ax.scatter(results_df_plot['OR'], results_df_plot['Predictor'], color=marker_color, zorder=3,
               s=100, marker='o', edgecolors='black', linewidth=0.8, label='Odds Ratio')
    
    ax.axvline(x=1, color=ref_line_color, linestyle='--', linewidth=1.5, alpha=0.7, label='No Effect (OR=1)')
    
    ax.set_xscale('log')
    
    min_or = results_df_plot['CI_Lower'].min() if not results_df_plot['CI_Lower'].empty else 0.1
    max_or = results_df_plot['CI_Upper'].max() if not results_df_plot['CI_Upper'].empty else 10
    
    x_ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
    x_ticks = [t for t in x_ticks if min_or * 0.5 < t < max_or * 2]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(t) for t in x_ticks])
    ax.tick_params(axis='x', labelsize=font_size_ticks)
    
    ax.set_xlabel('Odds Ratio (Log Scale)', fontsize=font_size_labels, fontweight='bold')
    ax.set_ylabel('Predictor Variable', fontsize=font_size_labels, fontweight='bold')
    ax.set_title('Forest Plot of Odds Ratios for Acute Complications', fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(axis='x', linestyle=':', alpha=0.6)
    
    for i, row in results_df_plot.iterrows():
        or_text = f"OR: {row['OR']:.2f}"
        p_text = f"p: {row['p-value']:.3f}"
        
        text_x_pos = row['CI_Upper'] * 1.02
        
        ax.text(text_x_pos, row['Predictor'], f"{or_text}, {p_text}",
                va='center', ha='left', fontsize=font_size_text, color='darkgreen')
    
    ax.set_ylim(-0.5, len(results_df_plot) - 0.5)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1)
    
    plt.show()
else:
    print("No valid results to plot. The results DataFrame is empty.")

print("\nResults of Multiple Univariate Logistic Regressions:")
print(results_df.to_markdown(index=False, floatfmt=".3f"))