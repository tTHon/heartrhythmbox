"""
Demographic Analysis — Overall + FP vs TN comparison
Dataset: All cases excluding FinalTest=1
FP vs TN comparison: Test set only (A_Test=1, GT-negative cases)

Input: A_modelCSV.csv
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, fisher_exact
import warnings
warnings.filterwarnings('ignore')

# ─── Load & exclude FinalTest=1 ───────────────────────────────────────────────
df       = pd.read_csv("CIED/AbdnL/A_modelCSV.csv")   # ← วางไฟล์ไว้ใน folder เดียวกับ script
df_clean = df[df["FinalTest"] != 1].copy()

# ─── Derived variables ────────────────────────────────────────────────────────
for d in [df_clean]:
    d["BMI"]        = d["Kg"] / ((d["cm"] / 100) ** 2)
    d["gender_bin"] = d["Gender"].str.extract(r"^(\d)").astype(float)  # 0=male,1=female
    d["BMI_cat"]    = pd.cut(d["BMI"], bins=[0,18.5,25,30,100],
                             labels=["<18.5","18.5–24.9","25–29.9","≥30"])
    d["Age_cat"]    = pd.cut(d["Age"], bins=[0,65,75,85,120],
                             labels=["<65","65–74","75–84","≥85"])

# ─── 1. Overall demographic summary (all, FinalTest≠1) ───────────────────────
print("=" * 70)
print("1. OVERALL DEMOGRAPHICS  (all cases, FinalTest≠1)")
print(f"   Total n={len(df_clean)}  |  isAbdn=1: {int((df_clean['isAbdn']==1).sum())}  |  isAbdn=0: {int((df_clean['isAbdn']==0).sum())}")
print(f"   Demographics available (Age non-null): n={df_clean['Age'].notna().sum()}")
print("=" * 70)

has_demo = df_clean[df_clean["Age"].notna()].copy()

print(f"\n{'Variable':<15s}  {'Median [IQR]':>25s}  {'Range':>15s}")
print("-" * 58)
for var, label in [("Age","Age (yr)"),("Kg","Weight (kg)"),("cm","Height (cm)"),("BMI","BMI (kg/m²)")]:
    g = has_demo[var].dropna()
    med = f"{g.median():.1f} [{g.quantile(.25):.1f}–{g.quantile(.75):.1f}]"
    rng = f"{g.min():.1f}–{g.max():.1f}"
    print(f"{label:<15s}  {med:>25s}  {rng:>15s}")

print(f"\nGender (n={has_demo['gender_bin'].notna().sum()}):")
gcount = has_demo["gender_bin"].value_counts().sort_index()
for k,v in gcount.items():
    label = "Male" if k==0 else "Female"
    pct = v/gcount.sum()*100
    print(f"  {label}: {v} ({pct:.1f}%)")

print(f"\nBMI Category:")
bmi_all = has_demo["BMI_cat"].value_counts(sort=False)
for cat, n in bmi_all.items():
    pct = n/bmi_all.sum()*100
    print(f"  {str(cat):<12s}: {n} ({pct:.1f}%)")

print(f"\nAge Category:")
age_all = has_demo["Age_cat"].value_counts(sort=False)
for cat, n in age_all.items():
    pct = n/age_all.sum()*100
    print(f"  {str(cat):<8s}: {n} ({pct:.1f}%)")

# ─── 2. FP vs TN comparison (test set, GT-negative only) ─────────────────────
test   = df_clean[df_clean["A_Test"] == 1].copy()
test["BMI"]        = test["Kg"] / ((test["cm"] / 100) ** 2)
test["gender_bin"] = test["Gender"].str.extract(r"^(\d)").astype(float)
test["BMI_cat"]    = pd.cut(test["BMI"], bins=[0,18.5,25,30,100],
                            labels=["<18.5","18.5–24.9","25–29.9","≥30"])
test["Age_cat"]    = pd.cut(test["Age"], bins=[0,65,75,85,120],
                            labels=["<65","65–74","75–84","≥85"])
test["FP_flag"]    = (test["FP"] == 1).astype(float)
test["FN_flag"]    = (test["FN"] == 1).astype(float)

gt_neg = test[test["isAbdn"] == 0].copy()
fp     = gt_neg[gt_neg["FP_flag"] == 1]
tn     = gt_neg[gt_neg["FP_flag"] == 0]
n_fp   = fp["Age"].notna().sum()
n_tn   = tn["Age"].notna().sum()

print("\n" + "=" * 70)
print("2. FP vs TN COMPARISON  (test set, GT-negative cases)")
print(f"   GT-negative n={len(gt_neg)}  |  FP n={len(fp)}  |  TN n={len(tn)}")
print(f"   Demographics available: FP n={n_fp}, TN n={n_tn}  (missing: {len(gt_neg)-n_fp-n_tn})")
print("=" * 70)

print(f"\n{'Variable':<15s}  {'FP (n='+str(n_fp)+') median [IQR]':>27s}  {'TN (n='+str(n_tn)+') median [IQR]':>27s}  {'p':>7s}")
print("-" * 82)
for var, label in [("Age","Age (yr)"),("Kg","Weight (kg)"),("cm","Height (cm)"),("BMI","BMI (kg/m²)")]:
    g1 = fp[var].dropna()
    g0 = tn[var].dropna()
    if len(g1)<2 or len(g0)<2:
        print(f"{label:<15s}  {'n too small':>27s}")
        continue
    _, p = mannwhitneyu(g1, g0, alternative="two-sided")
    m1   = f"{g1.median():.1f} [{g1.quantile(.25):.1f}–{g1.quantile(.75):.1f}]"
    m0   = f"{g0.median():.1f} [{g0.quantile(.25):.1f}–{g0.quantile(.75):.1f}]"
    sig  = " *" if p < 0.05 else ""
    print(f"{label:<15s}  {m1:>27s}  {m0:>27s}  {p:>6.3f}{sig}")

print(f"\nGender:")
ct = pd.crosstab(gt_neg["gender_bin"], gt_neg["FP_flag"])
ct.index   = ["Male","Female"]
ct.columns = ["TN","FP"]
ct["Total"] = ct.sum(axis=1)
print(ct)
_, p_g = fisher_exact(ct[["TN","FP"]].values)
print(f"Fisher's exact p = {p_g:.3f}")

print(f"\nBMI Category:")
bmi_tbl = gt_neg.groupby("BMI_cat", observed=False).apply(
    lambda g: pd.Series({"n":len(g),"FP":int(g["FP_flag"].sum()),
                         "TN":int((g["FP_flag"]==0).sum()),
                         "FP_rate":round(g["FP_flag"].mean(),3)}),
    include_groups=False).reset_index()
print(bmi_tbl.to_string(index=False))

print(f"\nAge Category:")
age_tbl = gt_neg.groupby("Age_cat", observed=False).apply(
    lambda g: pd.Series({"n":len(g),"FP":int(g["FP_flag"].sum()),
                         "TN":int((g["FP_flag"]==0).sum()),
                         "FP_rate":round(g["FP_flag"].mean(),3)}),
    include_groups=False).reset_index()
print(age_tbl.to_string(index=False))

# ─── 3. Any error (FP+FN) vs correct ─────────────────────────────────────────
test["error"] = ((test["FP_flag"]==1) | (test["FN_flag"]==1)).astype(int)
has_demo_test = test[test["Age"].notna()].copy()
err  = has_demo_test[has_demo_test["error"]==1]
corr = has_demo_test[has_demo_test["error"]==0]

print("\n" + "=" * 70)
print("3. ANY ERROR (FP+FN) vs CORRECT  (test set with demographics)")
print(f"   Error n={len(err)}, Correct n={len(corr)}")
print("=" * 70)
print(f"\n{'Variable':<15s}  {'Error (n='+str(len(err))+') median [IQR]':>27s}  {'Correct (n='+str(len(corr))+') median [IQR]':>27s}  {'p':>7s}")
print("-" * 82)
for var, label in [("Age","Age (yr)"),("Kg","Weight (kg)"),("cm","Height (cm)"),("BMI","BMI (kg/m²)")]:
    g1 = err[var].dropna()
    g0 = corr[var].dropna()
    if len(g1)<2 or len(g0)<2: continue
    _, p = mannwhitneyu(g1, g0, alternative="two-sided")
    m1   = f"{g1.median():.1f} [{g1.quantile(.25):.1f}–{g1.quantile(.75):.1f}]"
    m0   = f"{g0.median():.1f} [{g0.quantile(.25):.1f}–{g0.quantile(.75):.1f}]"
    sig  = " *" if p < 0.05 else ""
    print(f"{label:<15s}  {m1:>27s}  {m0:>27s}  {p:>6.3f}{sig}")

print(f"\nGender:")
ct2 = pd.crosstab(has_demo_test["gender_bin"], has_demo_test["error"])
ct2.index   = ["Male","Female"]
ct2.columns = ["Correct","Error"]
ct2["Total"] = ct2.sum(axis=1)
print(ct2)
_, p2 = fisher_exact(ct2[["Correct","Error"]].values)
print(f"Fisher's exact p = {p2:.3f}")

print("\n* p < 0.05")
print("Mann–Whitney U for continuous variables; Fisher's exact for gender.")