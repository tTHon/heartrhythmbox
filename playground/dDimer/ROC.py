import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import roc_curve, roc_auc_score

def plot_roc_curves(csv_file):
    # 1. Load Data
    df = pd.read_csv(csv_file)
    
    # Pre-process: Log transform dDimer (to handle skewness)
    df['Log_dDimer'] = np.log(df['dDimer'])
    
    # Ensure we use the exact same dataset (complete cases) for fair comparison
    data = df[['Thrombus', 'CHADVA', 'Log_dDimer']].dropna()
    y = data['Thrombus']
    
    # --- Model 1: CHADVA Only (Baseline) ---
    X1 = sm.add_constant(data[['CHADVA']])
    model1 = sm.Logit(y, X1).fit(disp=0)
    y_pred1 = model1.predict(X1)
    
    # Calculate ROC metrics
    fpr1, tpr1, _ = roc_curve(y, y_pred1)
    auc1 = roc_auc_score(y, y_pred1)
    
    # --- Model 2: CHADVA + dDimer (Augmented) ---
    X2 = sm.add_constant(data[['CHADVA', 'Log_dDimer']])
    model2 = sm.Logit(y, X2).fit(disp=0)
    y_pred2 = model2.predict(X2)
    
    # Calculate ROC metrics
    fpr2, tpr2, _ = roc_curve(y, y_pred2)
    auc2 = roc_auc_score(y, y_pred2)
    
    # --- Plotting ---
    plt.figure(figsize=(9, 7))
    
    # Plot curves
    plt.plot(fpr1, tpr1, color='blue', lw=2, linestyle='--',
             label=f'Model 1: CHADS-VASc (AUC = {auc1:.3f})')
    
    plt.plot(fpr2, tpr2, color='red', lw=2, 
             label=f'Model 2: CHADS-VASc + dDimer (AUC = {auc2:.3f})')
    
    # Add diagonal reference line (random guess)
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':')
    
    # Formatting
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve Comparison', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# Execute
plot_roc_curves('playground/dDimer/dDimer.csv')