import sys
import csv
import statistics

print(f"Python version: {sys.version}")
print(f"csv module version: {getattr(csv, '__version__', 'builtin')}")
print(f"statistics module version: {getattr(statistics, '__version__', 'builtin')}")
print("Modules used: csv, statistics")

def mean_and_sd(data):
    """
    Calculate mean and standard deviation of a list of numbers.
    Returns (mean, standard deviation).
    """
    if not data:
        raise ValueError("Data list is empty.")
    mean = statistics.mean(data)
    sd = statistics.stdev(data) if len(data) > 1 else 0.0
    return mean, sd

if __name__ == "__main__":
    # Read BSA values from BSA.csv
    bsa_values = []
    with open("playground/Micra/BSA.csv", newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            try:
                value = float(row[0])
                bsa_values.append(value)
            except (ValueError, IndexError):
                continue
    mean, sd = mean_and_sd(bsa_values)
    print(f"Mean BSA: {mean:.4f}, SD: {sd:.4f}")
    print(f"Min BSA: {min(bsa_values):.4f}, Max BSA: {max(bsa_values):.4f}")
    print(f"Count: {len(bsa_values)}")

    # Draw distribution curve
    # try:
        #import matplotlib.pyplot as plt
        #import numpy as np
        #from scipy import stats

        #plt.figure(figsize=(8, 5))
        # Histogram
        #plt.hist(bsa_values, bins=20, density=True, alpha=0.6, color='g', label='BSA Histogram')
        # Normal distribution curve
        #x = np.linspace(min(bsa_values), max(bsa_values), 100)
        #y = (1/(sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean)/sd)**2) if sd > 0 else np.zeros_like(x)
        #plt.plot(x, y, 'r-', label='Normal Curve')
        #plt.title("Distribution of BSA Values")
        #plt.xlabel("BSA")
        #plt.ylabel("Density")
        #plt.legend()
        #plt.tight_layout()
        #plt.show()

        # QQ plot
        #plt.figure(figsize=(6, 6))
        #stats.probplot(bsa_values, dist="norm", plot=plt)
        #plt.title("QQ Plot of BSA Values")
        #plt.tight_layout()
        #plt.show()
    #except ImportError:
        #print("matplotlib, numpy, or scipy not installed. Skipping plot.")