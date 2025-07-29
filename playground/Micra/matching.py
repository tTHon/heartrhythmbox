import pandas as pd
import os
import sys

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
    eligible_controls_all_criteria = df_controls[
        (abs(df_controls['BSA'] - current_case_bsa) <= 0.05) &
        (df_controls['Gender'] == current_case_gender) &
        (abs(df_controls['Age'] - current_case_age) <= 5) &
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
            (abs(df_controls['BSA'] - current_case_bsa) <= 0.05) &
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