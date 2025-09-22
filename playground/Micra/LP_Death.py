import pandas as pd
import io

# The data provided by the user
data = """Death	T2FU
0	1944
0	2956
0	3072
0	1072
1	2636
0	7
0	7
0	581
0	775
0	808
0	7
0	7
0	240
0	7
1	302
1	498
0	7
1	100
1	82
0	2248
1	2175
1	408
0	2229
0	2225
0	12
1	1577
0	314
0	2128
0	2135
1	158
0	47
0	1556
0	7
1	114
0	820
0	1405
0	1419
0	1374
0	968
0	957
0	734
0	753
0	727
0	743
0	733
1	27
0	656
0	552
0	481
0	43
0	419
0	276
0	368
1	170
0	13
0	188
0	33
1	21
0	46
"""

# Read the text data into a pandas DataFrame
df = pd.read_csv(io.StringIO(data), sep='\t')

# Calculate the total number of deaths and the total follow-up time in days
total_deaths = df['Death'].sum()
total_t2fu_days = df['T2FU'].sum()

# Convert the total follow-up time from days to years
total_patient_years = total_t2fu_days / 365.25

# Calculate the death rate per 100 patient-years
death_rate_per_100_py = (total_deaths / total_patient_years) * 100

print(f"Total number of deaths: {total_deaths}")
print(f"Total patient-years: {total_patient_years:.2f}")
print(f"Death rate per 100 patient-years: {death_rate_per_100_py:.2f}")