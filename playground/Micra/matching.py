import pandas as pd
import os
import sys
from scipy.stats import ttest_ind, chi2_contingency

print("Python version:", sys.version)

df_cases = pd.read_csv('playground/Micra/Micra4Match.csv')
df_controls = pd.read_csv('playground/Micra/TV4Match.csv')

# Preprocess 'Gender' column
df_cases['Gender'] = df_cases['Gender'].replace({'1. female': 1, '0. male': 0})
df_controls['Gender'] = df_controls['Gender'].replace({'1. female': 1, '0. male': 0})

# Create a DataFrame to store the final matched results
df_final_matched = df_cases.copy()
for i in range(1, 4):
    df_final_matched[f'Match#{i}'] = None

# Keep track of used control IDs
used_control_ids = set()

# List of cases (IDs) to process, ordered to prioritize those that might find more matches
# This simple sort might help, but for true optimality, more complex algorithms are needed
df_cases_sorted = df_cases.sort_values(by='BSA') # Sort cases to potentially get better initial distribution

# --- Pass 1: Match using all 3 criteria (BSA, Gender, Age) ---
for index, case in df_cases_sorted.iterrows():
    case_id = case['ID']
    current_case_bsa = case['BSA']
    current_case_gender = case['Gender']
    current_case_age = case['Age']

    # Filter controls based on all 3 criteria and exclude already used controls
    #caliper = 0.0486, age 3
    eligible_controls_all_criteria = df_controls[
        (abs(df_controls['BSA'] - current_case_bsa) <= 0.00486) &
        (df_controls['Gender'] == current_case_gender) &
        (abs(df_controls['Age'] - current_case_age) <= 3) &
        (~df_controls['ID'].isin(used_control_ids)) # Exclude already used controls
    ]

    # Sort by BSA to prioritize closest matches
    sorted_eligible_controls = eligible_controls_all_criteria.sort_values(by='BSA')

    # Select up to 3 unique IDs
    new_matches = sorted_eligible_controls['ID'].head(3).tolist()

    # Assign matches and add to used_control_ids
    for i in range(1, 4):
        if i <= len(new_matches):
            df_final_matched.loc[df_final_matched['ID'] == case_id, f'Match#{i}'] = new_matches[i-1]
            used_control_ids.add(new_matches[i-1])
        else:
            df_final_matched.loc[df_final_matched['ID'] == case_id, f'Match#{i}'] = None

# --- Pass 2: Re-match unfilled slots using 2 criteria (BSA, Gender) and remaining controls ---
# Iterate through the cases again to fill any remaining empty match slots
for index, case in df_final_matched.iterrows(): # Iterate over the partially filled df_final_matched
    case_id = case['ID']
    current_case_bsa = case['BSA']
    current_case_gender = case['Gender']

    # Check if this case still needs more matches
    needs_more_matches = False
    current_matches = [case[f'Match#{i}'] for i in range(1, 4) if pd.notna(case[f'Match#{i}'])]
    if len(current_matches) < 3:
        needs_more_matches = True

    if needs_more_matches:
        # Filter controls based on BSA and Gender, and exclude already used controls
        eligible_controls_two_criteria = df_controls[
            (abs(df_controls['BSA'] - current_case_bsa) <= 0.0486) &
            (df_controls['Gender'] == current_case_gender) &
            (~df_controls['ID'].isin(used_control_ids)) # Exclude already used controls
        ]

        # Sort by BSA
        sorted_eligible_controls = eligible_controls_two_criteria.sort_values(by='BSA')

        # Fill the remaining slots
        matches_to_add = sorted_eligible_controls['ID'].head(3 - len(current_matches)).tolist()

        current_match_idx = len(current_matches) + 1 # Start filling from the next available Match# slot
        for match_id in matches_to_add:
            df_final_matched.loc[df_final_matched['ID'] == case_id, f'Match#{current_match_idx}'] = match_id
            used_control_ids.add(match_id)
            current_match_idx += 1

# Save the final adjusted DataFrame
df_final_matched.to_csv('playground/micra/unique_adjusted_matched_micra.csv', index=False)

print("Matching completed with unique controls. Results saved in 'unique_adjusted_matched_micra.csv'.")

# Draw summary table
def get_stats(df, id_list):
    sub = df[df['ID'].isin(id_list)]
    avg_age = sub['Age'].mean() if not sub.empty else float('nan')
    pct_female = 100 * (sub['Gender'] == 1).mean() if not sub.empty else float('nan')
    avg_bsa = sub['BSA'].mean() if not sub.empty else float('nan')
    return avg_age, pct_female, avg_bsa, sub

# Prepare lists for each group
cases_ids = df_final_matched['ID'].tolist()
match1_ids = df_final_matched['Match#1'].dropna().tolist()
match2_ids = df_final_matched['Match#2'].dropna().tolist()
match3_ids = df_final_matched['Match#3'].dropna().tolist()

# Use original cases and controls DataFrames for stats
stats_cases = get_stats(df_cases, cases_ids)
stats_match1 = get_stats(df_controls, match1_ids)
stats_match2 = get_stats(df_controls, match2_ids)
stats_match3 = get_stats(df_controls, match3_ids)

# Statistical tests
def ttest(a, b):
    if len(a) > 1 and len(b) > 1:
        stat, p = ttest_ind(a, b, nan_policy='omit')
        return p
    return float('nan')

def chi2test(a, b):
    # a and b are DataFrames with 'Gender' column (1=female, 0=male)
    count_a = [(a['Gender'] == 1).sum(), (a['Gender'] == 0).sum()]
    count_b = [(b['Gender'] == 1).sum(), (b['Gender'] == 0).sum()]
    table = [count_a, count_b]
    try:
        chi2, p, _, _ = chi2_contingency(table)
        return p
    except:
        return float('nan')

# Calculate p-values (cases vs each match group)
p_age_1 = ttest(stats_cases[3]['Age'], stats_match1[3]['Age'])
p_age_2 = ttest(stats_cases[3]['Age'], stats_match2[3]['Age'])
p_age_3 = ttest(stats_cases[3]['Age'], stats_match3[3]['Age'])

p_bsa_1 = ttest(stats_cases[3]['BSA'], stats_match1[3]['BSA'])
p_bsa_2 = ttest(stats_cases[3]['BSA'], stats_match2[3]['BSA'])
p_bsa_3 = ttest(stats_cases[3]['BSA'], stats_match3[3]['BSA'])

p_female_1 = chi2test(stats_cases[3], stats_match1[3])
p_female_2 = chi2test(stats_cases[3], stats_match2[3])
p_female_3 = chi2test(stats_cases[3], stats_match3[3])

# Print table
print("\nSummary Table (with p-values):")
print("{:<18} {:>10} {:>10} {:>10} {:>10}".format("", "Cases", "Match#1", "Match#2", "Match#3"))
print("{:<18} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f}".format("Avg Age", stats_cases[0], stats_match1[0], stats_match2[0], stats_match3[0]))
print("{:<18} {:>10} {:>10.3f} {:>10.3f} {:>10.3f}".format("P (Age)", "", p_age_1, p_age_2, p_age_3))
print("{:<18} {:>10.1f} {:>10.1f} {:>10.1f} {:>10.1f}".format("% Female", stats_cases[1], stats_match1[1], stats_match2[1], stats_match3[1]))
print("{:<18} {:>10} {:>10.3f} {:>10.3f} {:>10.3f}".format("P (%Female)", "", p_female_1, p_female_2, p_female_3))
print("{:<18} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format("Avg BSA", stats_cases[2], stats_match1[2], stats_match2[2], stats_match3[2]))
print("{:<18} {:>10} {:>10.3f} {:>10.3f} {:>10.3f}".format("P (BSA)", "", p_bsa_1, p_bsa_2, p_bsa_3))