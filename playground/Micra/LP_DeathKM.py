import pandas as pd
import io
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import matplotlib.ticker as ticker

# Load the data
df = pd.read_csv('playground/Micra/LP_events.csv')

# Calculate the total number of deaths and the total follow-up time in days
total_deaths = df['Death'].sum()
total_t2fu_days = df['T2FU'].sum()
df ['T2FU_years'] = df['T2FU'] / 365.25

# Convert the total follow-up time from days to years
total_patient_years = total_t2fu_days / 365.25

# Calculate the death rate per 100 patient-years
death_rate_per_100_py = (total_deaths / total_patient_years) * 100

print(f"Total number of patients: {len(df)}")
print(f"Mean (SD) follow-up time (years): {df['T2FU_years'].mean():.2f} ({df['T2FU_years'].std():.2f}) Range: {df['T2FU_years'].min():.2f}-{df['T2FU_years'].max():.2f}")
print(f"Median (IQR) follow-up time (years): {df['T2FU_years'].median():.1f} ({df['T2FU_years'].quantile(0.25):.1f}-{df['T2FU_years'].quantile(0.75):.1f})")
print(f"Total number of deaths: {total_deaths}")
print(f"Total patient-years: {total_patient_years:.2f}")
print(f"Death rate per 100 patient-years: {death_rate_per_100_py:.2f}")

# --- Kaplan-Meier Curve Section ---

# Set font family for the plot to 'Inter'
# Note: The 'Inter' font must be installed on your system for this to work.
plt.rc('font', family='Inter')

# Convert T2FU from days to months
df['T2FU_months'] = df['T2FU'] / 30.44

# Create a KaplanMeierFitter object
kmf = KaplanMeierFitter()

# Fit the data to the model using the new 'T2FU_months' column
kmf.fit(df['T2FU_months'], event_observed=df['Death'])
km_incidence = 1 - kmf.survival_function_
print(f"\nKaplan-Meier cumulative incidence of deaths (Whole Cohort): {km_incidence.iloc[-1,0]:.4f}")


# Plot the Kaplan-Meier survival curve with custom color and linestyle
kmf.plot_survival_function(
    color='purple',
    linestyle='-',
    linewidth=2,
    ci_show=True,
    show_censors=True,
    at_risk_counts=True,  # This is the key change
    label='Overall Survival'
)

# Customize plot appearance
plt.title('Kaplan-Meier Survival Curve', fontsize=16, fontweight='bold')
plt.xlabel('Months', fontsize=12)
plt.ylabel('Survival Probability', fontsize=12)

# Set the x-axis to show integer ticks for months
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
