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
    print(f"Count: {len(bsa_values)}")