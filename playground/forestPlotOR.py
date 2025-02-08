import seaborn as sns
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# Sample data
data = {
    'Study': ['Study 1', 'Study 2', 'Study 3', 'Study 4', 'Study 5'],
    'OR': [0.8, 1.2, 1.5, 0.9, 1.1],
    'Lower CI': [0.6, 1.0, 1.2, 0.7, 0.9],
    'Upper CI': [1.0, 1.4, 1.8, 1.1, 1.3]
}

df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(8, 6))
sns.set(style="whitegrid")

# Create the forest plot
for i in range(df.shape[0]):
    plt.plot([df['Lower CI'][i], df['Upper CI'][i]], [i, i], color='black')
    plt.plot(df['OR'][i], i, 'ro')

# Customize the plot
plt.yticks(range(df.shape[0]), df['Study'])
plt.axvline(x=1, linestyle='--', color='grey')
plt.xlabel('Odds Ratio (OR)')
plt.title('Forest Plot with 95% CI')

plt.show()