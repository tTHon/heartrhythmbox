import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def check_normality(csv_file):
    # Load data
    df = pd.read_csv(csv_file)
    
    # Continuous variables to check
    continuous_cols = [
        'age', 'dDimer', 'CHADVA', 'HASBLED', 'BMI', 
        'creatinine', 'GFR', 'LAVI', 'LAdiameter', 'LVEF', 'PW'
    ]
    
    results = []
    
    # 1. Statistical Testing (Shapiro-Wilk)
    print(f"{'Variable':<15} {'Statistic':<10} {'P-value':<12} {'Conclusion'}")
    print("-" * 50)
    
    for col in continuous_cols:
        if col in df.columns:
            # Drop missing values for the test
            data = df[col].dropna()
            
            # Perform Shapiro-Wilk test
            stat, p_value = stats.shapiro(data)
            
            # Determine normality (alpha = 0.05)
            conclusion = "Normal" if p_value > 0.05 else "Not Normal"
            
            print(f"{col:<15} {stat:.3f}      {p_value:.3e}    {conclusion}")
            
            results.append({
                'Variable': col,
                'Statistic': stat,
                'P-value': p_value,
                'Distribution': conclusion
            })
            
    # 2. Visual Inspection (Optional: Save Plots)
    # This creates a grid of Histograms and Q-Q plots
    n_vars = len(continuous_cols)
    fig, axes = plt.subplots(n_vars, 2, figsize=(12, 4 * n_vars))
    
    for i, col in enumerate(continuous_cols):
        if col in df.columns:
            data = df[col].dropna()
            
            # Histogram with KDE
            sns.histplot(data, kde=True, ax=axes[i, 0])
            axes[i, 0].set_title(f'Histogram: {col}')
            
            # Q-Q Plot
            stats.probplot(data, dist="norm", plot=axes[i, 1])
            axes[i, 1].set_title(f'Q-Q Plot: {col}')
            
    plt.tight_layout()
    plt.show()


# Execute
check_normality('playground/dDimer/dDimer.csv')