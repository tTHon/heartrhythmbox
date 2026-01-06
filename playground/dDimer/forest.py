import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def generate_forest_plot(csv_file):
    df = pd.read_csv(csv_file)
    
    # Preprocessing
    # Log-transform dDimer (adding small constant if 0, though unlikely for dDimer)
    df['Log_dDimer'] = np.log(df['dDimer'])
    
    # Define predictors (using Log_dDimer instead of raw dDimer)
    # We can rename them for better display on the plot
    predictors = {
        'age': 'Age',
        'Gender': 'Gender (Female)', # Assuming 2 is female based on typical coding, but label just "Gender"
        'Log_dDimer': 'dDimer (Log)',
        'CHADVA': 'CHA2DS2-VASc',
        'HASBLED': 'HAS-BLED',
        'LAVI': 'LAVI',
        'LAdiameter': 'LA Diameter',
        'LVEF': 'LVEF',
        'PW': 'PW Velocity'
    }
    
    results = []
    
    for var, label in predictors.items():
        # Drop NaNs for the specific pair
        temp_df = df[[var, 'Thrombus']].dropna()
        X = temp_df[var]
        y = temp_df['Thrombus']
        X = sm.add_constant(X)
        
        try:
            model = sm.Logit(y, X).fit(disp=0)
            params = model.params
            conf = model.conf_int()
            p_val = model.pvalues[var]
            
            or_val = np.exp(params[var])
            lower_ci = np.exp(conf.loc[var][0])
            upper_ci = np.exp(conf.loc[var][1])
            
            results.append({
                'Variable': label,
                'OR': or_val,
                'Lower_CI': lower_ci,
                'Upper_CI': upper_ci,
                'P_value': p_val
            })
        except:
            continue
            
    # Create DataFrame and reverse order for plotting (top to bottom)
    res_df = pd.DataFrame(results).iloc[::-1]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Main Forest Plot
    # Error bars: xerr needs to be shape (2, N) -> [[left_diff], [right_diff]]
    y_pos = np.arange(len(res_df))
    x_err = [
        res_df['OR'] - res_df['Lower_CI'], 
        res_df['Upper_CI'] - res_df['OR']
    ]
    
    plt.errorbar(res_df['OR'], y_pos, xerr=x_err, fmt='s', color='black', 
                 ecolor='gray', capsize=5, markersize=6)
    
    # Reference line
    plt.axvline(x=1, color='red', linestyle='--', linewidth=1)
    
    # Formatting
    plt.yticks(y_pos, res_df['Variable'])
    plt.xlabel('Odds Ratio (95% CI)')
    plt.title('Forest Plot of Univariate Odds Ratios for Thrombus')
    
    # Add text annotations for values
    # Adjust x-axis limits to make room for text if needed, or put text on the right
    max_val = res_df['Upper_CI'].max()
    plt.xlim(0, max_val * 1.3) # Add space on right
    
    for i, row in res_df.iterrows():
        # Determine color based on significance
        color = 'red' if row['P_value'] < 0.05 else 'black'
        label_text = f"{row['OR']:.2f} ({row['Lower_CI']:.2f}-{row['Upper_CI']:.2f})"
        p_text = f"p={row['P_value']:.3f}" if row['P_value'] >= 0.001 else "p<0.001"
        
        # Position text
        text_x = max_val * 1.05
        # Use i (which is index from original df) ? No, i corresponds to y_pos in the reversed df
        # But wait, enumerate on reversed df gives 0, 1... which matches y_pos
        plt.text(text_x, i, f"{label_text}, {p_text}", va='center', fontsize=9, color=color)

    plt.tight_layout()
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.savefig('forest_plot.png')

generate_forest_plot('playground/dDimer/dDimer.csv')