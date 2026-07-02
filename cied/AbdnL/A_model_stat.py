"""
Demographic Analysis — Overall + FP vs TN comparison
Dataset: All cases excluding FinalTest=1
FP vs TN comparison: Test set only (A_Test=1, GT-negative cases)

Input: A_modelCSV.csv
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, fisher_exact, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# ─── Load & exclude FinalTest=1 ───────────────────────────────────────────────
df       = pd.read_csv("cied/AbdnL/A_modelCSV.csv")   
df_clean = df[df["FinalTest"] != 1].copy()

# ─── Derived variables ────────────────────────────────────────────────────────
for d in [df_clean]:
    d["BMI"]        = d["Kg"] / ((d["cm"] / 100) ** 2)
    d["gender_bin"] = d["Gender"].str.extract(r"^(\d)").astype(float)  # 0=male,1=female
    d["BMI_cat"]    = pd.cut(d["BMI"], bins=[0,18.5,25,30,100],
                             labels=["<18.5","18.5–24.9","25–29.9","≥30"])
    d["Age_cat"]    = pd.cut(d["Age"], bins=[0,65,75,85,120],
                             labels=["≤65","66–75","76–85",">85"])

# Helper function for categorical p-values safely
def get_categorical_p(df_subset, var, group_var):
    contingency_table = pd.crosstab(df_subset[var], df_subset[group_var])
    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        return "N/A"
    try:
        if contingency_table.size == 4:
            _, p = fisher_exact(contingency_table)
            test_name = "Fisher"
        else:
            _, p, _, _ = chi2_contingency(contingency_table)
            test_name = "Chi-sq"
        return f"{p:.3f} ({test_name})" if p >= 0.001 else "<0.001"
    except:
        return "N/A"

# Helper function to print categorical sections in comparisons
def print_categorical_comparison(df_subset, var, label, group_var, group1_val, group0_val):
    print(f"\n{label}")
    p_val = get_categorical_p(df_subset, var, group_var)
    
    cats = sorted(df_subset[var].dropna().unique())
    g1_df = df_subset[df_subset[group_var] == group1_val]
    g0_df = df_subset[df_subset[group_var] == group0_val]
    
    for c in cats:
        n1 = len(g1_df[g1_df[var] == c])
        p1 = (n1 / len(g1_df) * 100) if len(g1_df) > 0 else 0
        
        n0 = len(g0_df[g0_df[var] == c])
        p0 = (n0 / len(g0_df) * 100) if len(g0_df) > 0 else 0
        
        str_1 = f"{n1} ({p1:.1f}%)"
        str_0 = f"{n0} ({p0:.1f}%)"
        
        print(f"  {str(c):<25s}  {str_1:>27s}  {str_0:>27s}  {p_val:>7s}")
        p_val = ""

# ─── 1. OVERALL BASELINE CHARACTERISTICS (All vs A_Test=0 vs A_Test=1) ─────────
all_pats = df_clean.copy()
t0       = df_clean[df_clean["A_Test"] == 0].copy()
t1       = df_clean[df_clean["A_Test"] == 1].copy()

print("=" * 95)
print("1. OVERALL BASELINE CHARACTERISTICS")
print(f"   All Patients (n={len(all_pats)}), A_Test=0 (n={len(t0)}), A_Test=1 (n={len(t1)})")
print("=" * 95)
print(f"{'Variable':<25s}  {'All Patients':>18s}  {'A_Test = 0':>18s}  {'A_Test = 1':>18s}  {'p-value':>8s}")
print("-" * 95)

for var, label in [("Age","Age (yr)"), ("Kg","Weight (kg)"), ("cm","Height (cm)"), ("BMI","BMI (kg/m²)")]:
    g_all = all_pats[var].dropna()
    g0 = t0[var].dropna()
    g1 = t1[var].dropna()
    
    med_all = f"{g_all.median():.1f} [{g_all.quantile(.25):.1f}–{g_all.quantile(.75):.1f}]"
    med_0   = f"{g0.median():.1f} [{g0.quantile(.25):.1f}–{g0.quantile(.75):.1f}]"
    med_1   = f"{g1.median():.1f} [{g1.quantile(.25):.1f}–{g1.quantile(.75):.1f}]"
    
    _, p = mannwhitneyu(g0, g1, alternative="two-sided")
    p_str = f"{p:.3f}" if p >= 0.001 else "<0.001"
    print(f"{label:<25s}  {med_all:>18s}  {med_0:>18s}  {med_1:>18s}  {p_str:>8s}")

categorical_vars = [
    ("Gender", "Gender"),
    ("Type", "Device Type"),
    ("noActiveL", "No. of Active Leads"),
    ("noAbdnL", "No. of Abandoned Leads")
]

for var, label in categorical_vars:
    print(f"\n{label}")
    p_val = get_categorical_p(df_clean, var, "A_Test")
    cats = sorted(df_clean[var].dropna().unique())
    for c in cats:
        n_all = len(all_pats[all_pats[var] == c])
        p_all = (n_all / len(all_pats)) * 100
        n_0 = len(t0[t0[var] == c])
        p_0 = (n_0 / len(t0)) * 100 if len(t0) > 0 else 0
        n_1 = len(t1[t1[var] == c])
        p_1 = (n_1 / len(t1)) * 100 if len(t1) > 0 else 0
        
        str_all = f"{n_all} ({p_all:.1f}%)"
        str_0   = f"{n_0} ({p_0:.1f}%)"
        str_1   = f"{n_1} ({p_1:.1f}%)"
        print(f"  {str(c):<23s}  {str_all:>18s}  {str_0:>18s}  {str_1:>18s}  {p_val:>8s}")
        p_val = ""

# ─── 1b. DEVICE TYPE BY ISABDN MATRIX (Stratified by A_Test 0 vs A_Test 1) ───
print("\n" + "=" * 105)
print("1b. DEVICE TYPE BY ISABDN STRATIFIED BY A_TEST COHORTS")
print("=" * 105)
print(f"{'Device Type':<16s} | {'A_Test = 0, isAbdn=0':^20s} {'A_Test = 0, isAbdn=1':^20s} | {'A_Test = 1, isAbdn=0':^20s} {'A_Test = 1, isAbdn=1':^20s}")
print("-" * 105)

all_types = sorted(df_clean['Type'].dropna().unique())
t0_abd0 = t0[t0['isAbdn'] == 0]
t0_abd1 = t0[t0['isAbdn'] == 1]
t1_abd0 = t1[t1['isAbdn'] == 0]
t1_abd1 = t1[t1['isAbdn'] == 1]

for t in all_types:
    c0_0 = len(t0_abd0[t0_abd0['Type'] == t])
    p0_0 = (c0_0 / len(t0_abd0) * 100) if len(t0_abd0) > 0 else 0
    
    c0_1 = len(t0_abd1[t0_abd1['Type'] == t])
    p0_1 = (c0_1 / len(t0_abd1) * 100) if len(t0_abd1) > 0 else 0
    
    c1_0 = len(t1_abd0[t1_abd0['Type'] == t])
    p1_0 = (c1_0 / len(t1_abd0) * 100) if len(t1_abd0) > 0 else 0
    
    c1_1 = len(t1_abd1[t1_abd1['Type'] == t])
    p1_1 = (c1_1 / len(t1_abd1) * 100) if len(t1_abd1) > 0 else 0
    print(f"{t:<16s} | {c0_0:>5d} ({p0_0:>4.1f}%)     {c0_1:>5d} ({p0_1:>4.1f}%)     | {c1_0:>5d} ({p1_0:>4.1f}%)     {c1_1:>5d} ({p1_1:>4.1f}%)")

print("-" * 105)
p_t0 = get_categorical_p(t0, 'Type', 'isAbdn')
p_t1 = get_categorical_p(t1, 'Type', 'isAbdn')
print(f"{'p-value (within split)':<16s} | {p_t0:^46s} | {p_t1:^46s}")


# ─── 2. FALSE POSITIVE (FP) vs TRUE NEGATIVE (TN) ────────────────────────────
test = df_clean[(df_clean["A_Test"] == 1) & (df_clean["isAbdn"] == 0)].copy()
test["FP_flag"] = test["FP"].astype(int)

fp = test[test["FP_flag"] == 1]
tn = test[test["FP_flag"] == 0]

print("\n" + "=" * 82)
print("2. FALSE POSITIVE (FP) vs TRUE NEGATIVE (TN)  (Test set, GT-negative)")
print(f"   FP n={len(fp)}, TN n={len(tn)}")
print("=" * 82)
print(f"{'Variable':<27s}  {'FP (n='+str(len(fp))+')':^27s}  {'TN (n='+str(len(tn))+')':^27s}  {'p':>7s}")
print("-" * 95)

# --- Continuous Demographics ---
for var, label in [("Age","Age (yr)"),("Kg","Weight (kg)"),("cm","Height (cm)"),("BMI","BMI (kg/m²)")]:
    g1 = fp[var].dropna()
    g0 = tn[var].dropna()
    if len(g1)<2 or len(g0)<2: continue
    _, p = mannwhitneyu(g1, g0, alternative="two-sided")
    m1   = f"{g1.median():.1f} [{g1.quantile(.25):.1f}–{g1.quantile(.75):.1f}]"
    m0   = f"{g0.median():.1f} [{g0.quantile(.25):.1f}–{g0.quantile(.75):.1f}]"
    p_str = f"{p:.3f}" if p >= 0.001 else "<0.001"
    print(f"{label:<27s}  {m1:>27s}  {m0:>27s}  {p_str:>7s}")

# --- Requested Categorical Variables ---
comp_cats = [
    ("Type", "Device Type"),
    ("noActiveL", "No. of Active Leads"),
    ("noAbdnL", "No. of Abandoned Leads"),
    ("hasIntra", "Has Intracardiac Artifacts"),
    ("hasExtra", "Has Extracardiac Artifacts")
]

for var, label in comp_cats:
    print_categorical_comparison(test, var, label, "FP_flag", 1, 0)


# ─── 3. ANY ERROR (FP+FN) vs CORRECT ─────────────────────────────────────────
test_all = df_clean[df_clean["A_Test"] == 1].copy()
test_all["error"] = ((test_all["FP"].astype(int)==1) | (test_all["FN"].astype(int)==1)).astype(int)

err  = test_all[test_all["error"] == 1]
corr = test_all[test_all["error"] == 0]

print("\n" + "=" * 82)
print("3. ANY ERROR (FP+FN) vs CORRECT  (Test set evaluation)")
print(f"   Error n={len(err)}, Correct n={len(corr)}")
print("=" * 82)
print(f"{'Variable':<27s}  {'Error (n='+str(len(err))+')':^27s}  {'Correct (n='+str(len(corr))+')':^27s}  {'p':>7s}")
print("-" * 95)

# --- Continuous Demographics ---
for var, label in [("Age","Age (yr)"),("Kg","Weight (kg)"),("cm","Height (cm)"),("BMI","BMI (kg/m²)")]:
    g1 = err[var].dropna()
    g0 = corr[var].dropna()
    if len(g1)<2 or len(g0)<2: continue
    _, p = mannwhitneyu(g1, g0, alternative="two-sided")
    m1   = f"{g1.median():.1f} [{g1.quantile(.25):.1f}–{g1.quantile(.75):.1f}]"
    m0   = f"{g0.median():.1f} [{g0.quantile(.25):.1f}–{g0.quantile(.75):.1f}]"
    p_str = f"{p:.3f}" if p >= 0.001 else "<0.001"
    print(f"{label:<27s}  {m1:>27s}  {m0:>27s}  {p_str:>7s}")

# --- Requested Categorical Variables ---
for var, label in comp_cats:
    print_categorical_comparison(test_all, var, label, "error", 1, 0)