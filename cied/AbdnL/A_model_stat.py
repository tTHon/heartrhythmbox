"""
Abandoned Lead Detection — Combined Analysis (With FN vs TP Addon)
==================================================================
Merges two previously separate scripts into one pipeline:

  PART A (from Abdn_FPFN.py):
      Apply the operating threshold (prob > 0.80, pixel > 600) to
      out-of-fold per-image predictions, compute the case-level
      confusion matrix (TP/FP/FN/TN) with Wilson 95% CIs, and run
      FP/FN error analysis stratified by confounders and device type.

  PART B (from A_model_stat.py):
      Baseline demographic characteristics (overall + train/test split)
      and FP-vs-TN / Error-vs-Correct / FN-vs-TP demographic comparisons.

Inputs:
    per_image_results.csv       — out-of-fold per-image predictions
    A_modelCSV.csv              — per-image clinical metadata

Output: printed report (all sections below), no files written.
"""

import re
import warnings
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PART A: PIPELINE GENERATION & CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

print(">> Running Pipeline: Matching predictions with clinical metadata...")

# 1. Load Data
df_meta = pd.read_csv("c:/CIEDID_data/AbdnL/FPFN/A_modelCSV.csv")
df_meta.columns = df_meta.columns.str.strip()  # Clear hidden whitespaces

df_img = pd.read_csv("c:/CIEDID_data/AbdnL/FPFN/per_image_results.csv")
df_img.columns = df_img.columns.str.strip()

# 2. Extract Patient ID from Image Path
def extract_id(path):
    match = re.search(r'\\([^\\]+)\.png$', str(path))
    if match:
        return match.group(1)
    return None

df_img['ID'] = df_img['image'].apply(extract_id)

# 3. Apply Decision Boundary (Threshold Optimization)
# Standard Operating Criteria: prob_80 >= 600 pixels
df_img['pred_positive'] = df_img['prob_80'] >= 600

# Roll up to Patient Level (Case-level Evaluation)
case_preds = df_img.groupby('ID')['pred_positive'].any().astype(int).reset_index()
case_preds.columns = ['ID', 'pred_isAbdn']

# Harmonize types and merge
df_meta['ID'] = df_meta['ID'].astype(str)
case_preds['ID'] = case_preds['ID'].astype(str)
df_all = pd.merge(df_meta, case_preds, on='ID', how='inner')

# Clean criteria (Exclude FinalTest=1, prevent NaN typing errors)
df_clean = df_all[df_all["FinalTest"] != 1].copy()
df_clean["Type"] = df_clean["Type"].fillna("Unknown").astype(str)

# Extract dynamic operational metrics
df_clean["FP_flag"] = ((df_clean["pred_isAbdn"] == 1) & (df_clean["isAbdn"] == 0)).astype(int)
df_clean["FN_flag"] = ((df_clean["pred_isAbdn"] == 0) & (df_clean["isAbdn"] == 1)).astype(int)

# ─── PART B: STATISTICAL ANALYSIS START ───────────────────────────────────────

# Derived variables for demographics
for d in [df_clean]:
    d["BMI"]        = d["Kg"] / ((d["cm"] / 100) ** 2)
    d["gender_bin"] = d["Gender"].str.extract(r"^(\d)").astype(float)
    d["BMI_cat"]    = pd.cut(d["BMI"], bins=[0,18.5,25,30,100], labels=["<18.5","18.5–24.9","25–29.9","≥30"])
    d["Age_cat"]    = pd.cut(d["Age"], bins=[0,65,75,85,120], labels=["≤65","66–75","76–85",">85"])

# Helper function for categorical tests (Chi-sq / Fisher automated)
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
        p_str = f"{p:.3f}" if p >= 0.001 else "<0.001"
        return f"{p_str} ({test_name})"
    except:
        return "N/A"

# Helper function to render categorical sections
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
        
        print(f"  {str(c):<25s}  {f'{n1} ({p1:.1f}%)':>27s}  {f'{n0} ({p0:.1f}%)':>27s}  {p_val:>15s}")
        p_val = ""

# ─── B1. OVERALL BASELINE CHARACTERISTICS ──────────────────────────────────────
all_pats = df_clean.copy()
t0       = df_clean[df_clean["A_Test"] == 0].copy()
t1       = df_clean[df_clean["A_Test"] == 1].copy()

print("\n" + "=" * 95)
print("1. OVERALL BASELINE CHARACTERISTICS")
print(f"   All Patients (n={len(all_pats)}), A_Test=0 (n={len(t0)}), A_Test=1 (n={len(t1)})")
print("=" * 95)
print(f"{'Variable':<25s}  {'All Patients':>18s}  {'A_Test = 0':>18s}  {'A_Test = 1':>18s}  {'p-value':>15s}")
print("-" * 105)

for var, label in [("Age","Age (yr)"), ("Kg","Weight (kg)"), ("cm","Height (cm)"), ("BMI","BMI (kg/m²)")]:
    g_all = all_pats[var].dropna()
    g0 = t0[var].dropna()
    g1 = t1[var].dropna()
    
    med_all = f"{g_all.median():.1f} [{g_all.quantile(.25):.1f}–{g_all.quantile(.75):.1f}]"
    med_0   = f"{g0.median():.1f} [{g0.quantile(.25):.1f}–{g0.quantile(.75):.1f}]"
    med_1   = f"{g1.median():.1f} [{g1.quantile(.25):.1f}–{g1.quantile(.75):.1f}]"
    
    _, p = mannwhitneyu(g0, g1, alternative="two-sided")
    p_str = f"{p:.3f} (MWU)" if p >= 0.001 else "<0.001 (MWU)"
    print(f"{label:<25s}  {med_all:>18s}  {med_0:>18s}  {med_1:>18s}  {p_str:>15s}")

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
        
        print(f"  {str(c):<23s}  {f'{n_all} ({p_all:.1f}%)':>18s}  {f'{n_0} ({p_0:.1f}%)':>18s}  {f'{n_1} ({p_1:.1f}%)':>18s}  {p_val:>15s}")
        p_val = ""


# ─── B2. FALSE POSITIVE (FP) vs TRUE NEGATIVE (TN) ────────────────────────────
test_neg = df_clean[(df_clean["A_Test"] == 1) & (df_clean["isAbdn"] == 0)].copy()

fp = test_neg[test_neg["FP_flag"] == 1]
tn = test_neg[test_neg["FP_flag"] == 0]

print("\n" + "=" * 82)
print("2. FALSE POSITIVE (FP) vs TRUE NEGATIVE (TN)  (Test set, GT-negative)")
print(f"   FP n={len(fp)}, TN n={len(tn)}")
print("=" * 82)
print(f"{'Variable':<27s}  {'FP (n='+str(len(fp))+')':^27s}  {'TN (n='+str(len(tn))+')':^27s}  {'p':>15s}")
print("-" * 105)

for var, label in [("Age","Age (yr)"),("Kg","Weight (kg)"),("cm","Height (cm)"),("BMI","BMI (kg/m²)")]:
    g1 = fp[var].dropna()
    g0 = tn[var].dropna()
    if len(g1)<2 or len(g0)<2:
        print(f"{label:<27s}  {'N/A':>27s}  {'N/A':>27s}  {'N/A':>15s}")
        continue
    _, p = mannwhitneyu(g1, g0, alternative="two-sided")
    m1   = f"{g1.median():.1f} [{g1.quantile(.25):.1f}–{g1.quantile(.75):.1f}]"
    m0   = f"{g0.median():.1f} [{g0.quantile(.25):.1f}–{g0.quantile(.75):.1f}]"
    p_str = f"{p:.3f} (MWU)" if p >= 0.001 else "<0.001 (MWU)"
    print(f"{label:<27s}  {m1:>27s}  {m0:>27s}  {p_str:>15s}")

comp_cats = [
    ("Type", "Device Type"),
    ("noActiveL", "No. of Active Leads"),
    ("noAbdnL", "No. of Abandoned Leads"),
    ("hasIntra", "Has Intracardiac Artifacts"),
    ("hasExtra", "Has Extracardiac Artifacts")
]

for var, label in comp_cats:
    print_categorical_comparison(test_neg, var, label, "FP_flag", 1, 0)


# ─── B3. ANY ERROR (FP+FN) vs CORRECT ─────────────────────────────────────────
test_all = df_clean[df_clean["A_Test"] == 1].copy()
test_all["error"] = ((test_all["FP_flag"] == 1) | (test_all["FN_flag"] == 1)).astype(int)

err  = test_all[test_all["error"] == 1]
corr = test_all[test_all["error"] == 0]

print("\n" + "=" * 82)
print("3. ANY ERROR (FP+FN) vs CORRECT  (Test set evaluation)")
print(f"   Error n={len(err)}, Correct n={len(corr)}")
print("=" * 82)
print(f"{'Variable':<27s}  {'Error (n='+str(len(err))+')':^27s}  {'Correct (n='+str(len(corr))+')':^27s}  {'p':>15s}")
print("-" * 105)

for var, label in [("Age","Age (yr)"),("Kg","Weight (kg)"),("cm","Height (cm)"),("BMI","BMI (kg/m²)")]:
    g1 = err[var].dropna()
    g0 = corr[var].dropna()
    if len(g1)<2 or len(g0)<2:
        print(f"{label:<27s}  {'N/A':>27s}  {'N/A':>27s}  {'N/A':>15s}")
        continue
    _, p = mannwhitneyu(g1, g0, alternative="two-sided")
    m1   = f"{g1.median():.1f} [{g1.quantile(.25):.1f}–{g1.quantile(.75):.1f}]"
    m0   = f"{g0.median():.1f} [{g0.quantile(.25):.1f}–{g0.quantile(.75):.1f}]"
    p_str = f"{p:.3f} (MWU)" if p >= 0.001 else "<0.001 (MWU)"
    print(f"{label:<27s}  {m1:>27s}  {m0:>27s}  {p_str:>15s}")

for var, label in comp_cats:
    print_categorical_comparison(test_all, var, label, "error", 1, 0)


# ─── B4. FALSE NEGATIVE (FN) vs TRUE POSITIVE (TP) [ADDON] ────────────────────
test_pos = df_clean[(df_clean["A_Test"] == 1) & (df_clean["isAbdn"] == 1)].copy()

fn_cases = test_pos[test_pos["FN_flag"] == 1]
tp_cases = test_pos[test_pos["FN_flag"] == 0]

print("\n" + "=" * 82)
print("4. FALSE NEGATIVE (FN) vs TRUE POSITIVE (TP)  (Test set, GT-positive)")
print(f"   FN n={len(fn_cases)}, TP n={len(tp_cases)}")
print("=" * 82)
print(f"{'Variable':<27s}  {'FN (n='+str(len(fn_cases))+')':^27s}  {'TP (n='+str(len(tp_cases))+')':^27s}  {'p':>15s}")
print("-" * 105)

for var, label in [("Age","Age (yr)"),("Kg","Weight (kg)"),("cm","Height (cm)"),("BMI","BMI (kg/m²)")]:
    g1 = fn_cases[var].dropna()
    g0 = tp_cases[var].dropna()
    if len(g1)<2 or len(g0)<2:
        print(f"{label:<27s}  {'N/A':>27s}  {'N/A':>27s}  {'N/A':>15s}")
        continue
    _, p = mannwhitneyu(g1, g0, alternative="two-sided")
    m1   = f"{g1.median():.1f} [{g1.quantile(.25):.1f}–{g1.quantile(.75):.1f}]"
    m0   = f"{g0.median():.1f} [{g0.quantile(.25):.1f}–{g0.quantile(.75):.1f}]"
    p_str = f"{p:.3f} (MWU)" if p >= 0.001 else "<0.001 (MWU)"
    print(f"{label:<27s}  {m1:>27s}  {m0:>27s}  {p_str:>15s}")

for var, label in comp_cats:
    print_categorical_comparison(test_pos, var, label, "FN_flag", 1, 0)