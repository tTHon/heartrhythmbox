"""
Abandoned Lead Detection — Combined Analysis
==============================================
Merges two previously separate scripts into one pipeline:

  PART A (from Abdn_FPFN.py):
      Apply the operating threshold (prob > 0.80, pixel > 600) to
      out-of-fold per-image predictions, compute the case-level
      confusion matrix (TP/FP/FN/TN) with Wilson 95% CIs, and run
      FP/FN error analysis stratified by confounders and device type.

  PART B (from A_model_stat.py):
      Baseline demographic characteristics (overall + train/test split)
      and FP-vs-TN / Error-vs-Correct demographic comparisons.

Combining these into one script means the FP/FN outcome flags used in
PART B are guaranteed to come from the SAME threshold application as
PART A, rather than risking a stale/inconsistent "FP"/"FN" column
baked into a separately-maintained metadata CSV.

Inputs:
    per_image_results.csv   — out-of-fold per-image predictions
    A_model_detail.csv          — per-image clinical metadata (ID, Type,
                                   noActiveL, hasIntra, hasExtra, etc.)

Output: printed report (all sections below), no files written.
"""

import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu

warnings.filterwarnings("ignore")

# ── EDIT THESE PATHS ────────────────────────────────────────────────
OOF_RESULTS_CSV = "C:/CIEDID_data/AbdnL/FPFN/per_image_results.csv"
META_CSV        = "C:/CIEDID_data/AbdnL/FPFN/A_modelCSV.csv"
PROB_THRESHOLD  = 0.80
PIXEL_THRESHOLD = 600
PROB_COL        = f"prob_{int(PROB_THRESHOLD*100):02d}"   # e.g. "prob_80"
# ─────────────────────────────────────────────────────────────────────


def wilson_ci(successes, n, z=1.96):
    """Wilson score interval for a binomial proportion. Returns (lower, upper)."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = successes / n
    denom = 1 + (z**2) / n
    center = phat + (z**2) / (2 * n)
    margin = z * np.sqrt((phat * (1 - phat)) / n + (z**2) / (4 * n**2))
    lower = (center - margin) / denom
    upper = (center + margin) / denom
    return max(0.0, lower), min(1.0, upper)


def get_categorical_p(df_subset, var, group_var):
    """will use Fisher's exact as N is small."""
    contingency_table = pd.crosstab(df_subset[var], df_subset[group_var])
    
    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        return "N/A"
    try:
        _, p = fisher_exact(contingency_table)
        test_name = "Fisher"
        return f"{p:.3f} ({test_name})" if p >= 0.001 else "<0.001"
    #    if contingency_table.size == 4:
    #        _, p = fisher_exact(contingency_table)
    #        test_name = "Fisher"
    #    else:
    #        _, p, _, _ = chi2_contingency(contingency_table)
    #        test_name = "Chi-sq"
    #    return f"{p:.3f} ({test_name})" if p >= 0.001 else "<0.001"
    except Exception:
        return "N/A"


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


# =====================================================================
# PART A — Threshold application, confusion matrix, FP/FN error analysis
# (formerly Abdn_FPFN.py)
# =====================================================================

res = pd.read_csv(OOF_RESULTS_CSV)
meta = pd.read_csv(META_CSV)

res["img_id"] = res["image"].str.extract(r"([^\\\/]+)\.(png|jpe?g)$", expand=False)[0]


def norm_id(s):
    s = str(s).strip()
    s = re.sub(r"^a_", "", s)  # strip a_ prefix
    return s


res["img_id_norm"] = res["img_id"].apply(norm_id)
meta_test = meta[meta["A_Test"] == 1].copy()
meta_test["img_id_norm"] = meta_test["ID"].astype(str).apply(norm_id)

# Apply the operating threshold
res["pred_pos"] = (res[PROB_COL] > PIXEL_THRESHOLD).astype(int)
res["gt_pos"] = res["has_abandoned"].astype(int)


def outcome(row):
    if row["gt_pos"] == 1 and row["pred_pos"] == 1:
        return "TP"
    if row["gt_pos"] == 0 and row["pred_pos"] == 1:
        return "FP"
    if row["gt_pos"] == 1 and row["pred_pos"] == 0:
        return "FN"
    return "TN"


res["outcome"] = res.apply(outcome, axis=1)

merged = res.merge(meta_test, on="img_id_norm", how="left")

merged["has_intracardiac"] = merged["hasIntra"].notna().astype(int)
merged["has_extracardiac"] = merged["hasExtra"].notna().astype(int)
merged["is_CRT"] = merged["Type"].isin(["CRTD", "CRTP"]).astype(int)
merged["has_any_confounder"] = (
    merged["has_intracardiac"] | merged["has_extracardiac"] | merged["is_CRT"]
).astype(int)

# ─── A1. Overall confusion matrix ────────────────────────────────────
print("=" * 65)
print(f"A1. OVERALL CONFUSION MATRIX  (Test set, threshold prob>{PROB_THRESHOLD} & pixel>{PIXEL_THRESHOLD})")
print("=" * 65)
TP = (merged["outcome"] == "TP").sum()
FP = (merged["outcome"] == "FP").sum()
FN = (merged["outcome"] == "FN").sum()
TN = (merged["outcome"] == "TN").sum()
pos, neg = merged["gt_pos"].sum(), (merged["gt_pos"] == 0).sum()

print(f"\n{'':22s}  {'Pred POS':>10s}  {'Pred NEG':>10s}  {'Total':>6s}")
print(f"{'GT POS (has abandoned)':22s}  {'TP='+str(TP):>10s}  {'FN='+str(FN):>10s}  {pos:>6d}")
print(f"{'GT NEG (no abandoned)':22s}  {'FP='+str(FP):>10s}  {'TN='+str(TN):>10s}  {neg:>6d}")
print(f"{'Total':22s}  {TP+FP:>10d}  {FN+TN:>10d}  {len(merged):>6d}")

sens = TP / pos
spec = TN / neg
ppv = TP / (TP + FP) if (TP + FP) > 0 else 0
npv = TN / (TN + FN) if (TN + FN) > 0 else 0
f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
sens_ci = wilson_ci(TP, pos)
spec_ci = wilson_ci(TN, neg)
ppv_ci = wilson_ci(TP, TP + FP)
npv_ci = wilson_ci(TN, TN + FN)

print(f"\nSensitivity  : {TP}/{pos}  = {sens:.3f} [{sens_ci[0]:.3f}, {sens_ci[1]:.3f}]")
print(f"Specificity  : {TN}/{neg}  = {spec:.3f} [{spec_ci[0]:.3f}, {spec_ci[1]:.3f}]")
print(f"PPV          : {TP}/{TP+FP}  = {ppv:.3f} [{ppv_ci[0]:.3f}, {ppv_ci[1]:.3f}]")
print(f"NPV          : {TN}/{TN+FN}  = {npv:.3f} [{npv_ci[0]:.3f}, {npv_ci[1]:.3f}]")
print(f"F1 score     :            {f1:.3f}")

# ─── A2. FN cases detail ─────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"A2. FALSE NEGATIVE CASES  (missed abandoned lead, n={FN})")
print("=" * 65)
fn_cases = merged[merged["outcome"] == "FN"].copy()
print(fn_cases[["img_id", "Type", "noActiveL", "noAbdnL", "targ_px", "pred_px", PROB_COL,
                 "Intracardiac device", "Extracardiac type"]].to_string(index=False))

# ─── A3. FP cases detail ─────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"A3. FALSE POSITIVE CASES  (false alarm, n={FP})")
print("=" * 65)
fp_cases = merged[merged["outcome"] == "FP"].copy()
print(fp_cases[["img_id", "Type", "noActiveL", "pred_px", PROB_COL,
                 "Intracardiac device", "Extracardiac type"]].to_string(index=False))

# ─── A4. FP stratified by confounder ─────────────────────────────────
print("\n" + "=" * 65)
print(f"A4. FP RATE BY CONFOUNDER  (GT-negative cases only, n={neg})")
print("=" * 65)
neg_cases = merged[merged["gt_pos"] == 0].copy()
confounders = {
    "CRT (CRTD/CRTP)": "is_CRT",
    "hasIntra": "has_intracardiac",
    "hasExtra": "has_extracardiac",
    "Any confounder": "has_any_confounder",
}
print(f"\n{'Subgroup':<25s}  {'With confounder':>16s}  {'Without confounder':>19s}  {'Fisher p':>9s}")
print(f"{'':25s}  {'n / FP / rate':>16s}  {'n / FP / rate':>19s}  {'':>9s}")
print("-" * 78)
for label, col in confounders.items():
    g1 = neg_cases[neg_cases[col] == 1]
    g0 = neg_cases[neg_cases[col] == 0]
    fp1, fp0 = (g1["outcome"] == "FP").sum(), (g0["outcome"] == "FP").sum()
    n1, n0 = len(g1), len(g0)
    r1 = f"{fp1}/{n1} ({fp1/n1:.0%})" if n1 > 0 else "—"
    r0 = f"{fp0}/{n0} ({fp0/n0:.0%})" if n0 > 0 else "—"
    tbl = [[fp1, n1 - fp1], [fp0, n0 - fp0]]
    _, p = fisher_exact(tbl) if n1 > 0 and n0 > 0 else (None, float("nan"))
    ps = f"{p:.3f}" if not np.isnan(p) else "—"
    print(f"{label:<25s}  {r1:>16s}  {r0:>19s}  {ps:>9s}")

# ─── A5. FN stratified by confounder ─────────────────────────────────
print("\n" + "=" * 65)
print(f"A5. FN RATE BY CONFOUNDER  (GT-positive cases only, n={pos})")
print("=" * 65)
pos_cases = merged[merged["gt_pos"] == 1].copy()
print(f"\n{'Subgroup':<25s}  {'With confounder':>16s}  {'Without confounder':>19s}  {'Fisher p':>9s}")
print(f"{'':25s}  {'n / FN / rate':>16s}  {'n / FN / rate':>19s}  {'':>9s}")
print("-" * 78)
for label, col in confounders.items():
    g1 = pos_cases[pos_cases[col] == 1]
    g0 = pos_cases[pos_cases[col] == 0]
    fn1, fn0 = (g1["outcome"] == "FN").sum(), (g0["outcome"] == "FN").sum()
    n1, n0 = len(g1), len(g0)
    r1 = f"{fn1}/{n1} ({fn1/n1:.0%})" if n1 > 0 else "—"
    r0 = f"{fn0}/{n0} ({fn0/n0:.0%})" if n0 > 0 else "—"
    tbl = [[fn1, n1 - fn1], [fn0, n0 - fn0]]
    _, p = fisher_exact(tbl) if n1 > 0 and n0 > 0 else (None, float("nan"))
    ps = f"{p:.3f}" if not np.isnan(p) else "—"
    print(f"{label:<25s}  {r1:>16s}  {r0:>19s}  {ps:>9s}")

# ─── A6. FP breakdown by device type ─────────────────────────────────
print("\n" + "=" * 65)
print("A6. FP BREAKDOWN BY DEVICE TYPE  (GT-negative cases)")
print("=" * 65)
tbl = neg_cases.groupby("Type", as_index=False).agg(
    n=("outcome", "count"),
    FP=("outcome", lambda s: (s == "FP").sum()),
)
tbl["FP_rate"] = tbl["FP"] / tbl["n"]
tbl = tbl.sort_values("FP_rate", ascending=False)
print(tbl.to_string(index=False))

# ─── A7. FN breakdown by device type ─────────────────────────────────
print("\n" + "=" * 65)
print("A7. FN BREAKDOWN BY DEVICE TYPE  (GT-positive cases)")
print("=" * 65)
tbl2 = pos_cases.groupby("Type", dropna=False, as_index=False).agg(
    n=("outcome", "count"),
    FN=("outcome", lambda s: (s == "FN").sum()),
)
tbl2["FN_rate"] = tbl2["FN"] / tbl2["n"]
tbl2 = tbl2.sort_values("FN_rate", ascending=False)
print(tbl2.to_string(index=False))

# ─── A8. Pixel count analysis ────────────────────────────────────────
print("\n" + "=" * 65)
print(f"A8. PIXEL COUNT ANALYSIS  ({PROB_COL} = pixel count at prob>{PROB_THRESHOLD})")
print("=" * 65)
print(f"\n{'Outcome':<6s}  {'n':>3s}  {PROB_COL+' mean':>13s}  {PROB_COL+' median':>14s}  {'targ_px mean':>13s}")
print("-" * 58)
for oc in ["TP", "FP", "FN", "TN"]:
    g = merged[merged["outcome"] == oc]
    if len(g) == 0:
        continue
    print(f"{oc:<6s}  {len(g):>3d}  {g[PROB_COL].mean():>13.0f}  {g[PROB_COL].median():>14.0f}  {g['targ_px'].mean():>13.0f}")

print(f"\nFN cases — pixel detail (threshold = {PIXEL_THRESHOLD}):")
print(fn_cases[["img_id", "targ_px", "pred_px", PROB_COL]].to_string(index=False))


# =====================================================================
# PART B — Baseline demographics + FP/FN-based comparisons
# (formerly A_model_stat.py)
#
# Reuses the SAME threshold-derived outcome ("merged"/"res") from
# PART A above instead of relying on precomputed FP/FN columns in a
# separately-maintained metadata CSV — this keeps both halves of the
# analysis consistent with a single threshold application.
# =====================================================================

# Build df_clean from metadata, excluding the FinalTest=1 pipeline test set,
# then attach the FP/FN outcome flags derived in PART A.
df_clean = meta[meta["FinalTest"] != 1].copy()
df_clean["ID_norm"] = df_clean["ID"].astype(str).apply(norm_id)

outcome_map = res.set_index("img_id_norm")["outcome"]
df_clean["outcome"] = df_clean["ID_norm"].map(outcome_map)
df_clean["FP"] = (df_clean["outcome"] == "FP").astype(int)
df_clean["FN"] = (df_clean["outcome"] == "FN").astype(int)

for d in [df_clean]:
    d["BMI"] = d["Kg"] / ((d["cm"] / 100) ** 2)
    d["gender_bin"] = d["Gender"].str.extract(r"^(\d)").astype(float)  # 0=male,1=female
    d["BMI_cat"] = pd.cut(d["BMI"], bins=[0, 18.5, 25, 30, 100],
                          labels=["<18.5", "18.5–24.9", "25–29.9", "≥30"])
    d["Age_cat"] = pd.cut(d["Age"], bins=[0, 65, 75, 85, 120],
                          labels=["≤65", "66–75", "76–85", ">85"])

# ─── B1. OVERALL BASELINE CHARACTERISTICS ────────────────────────────
all_pats = df_clean.copy()
t0 = df_clean[df_clean["A_Test"] == 0].copy()
t1 = df_clean[df_clean["A_Test"] == 1].copy()

print("\n\n" + "=" * 95)
print("B1. OVERALL BASELINE CHARACTERISTICS")
print(f"    All Patients (n={len(all_pats)}), A_Test=0 (n={len(t0)}), A_Test=1 (n={len(t1)})")
print("=" * 95)
print(f"{'Variable':<25s}  {'All Patients':>18s}  {'A_Test = 0':>18s}  {'A_Test = 1':>18s}  {'p-value':>8s}")
print("-" * 95)

for var, label in [("Age", "Age (yr)"), ("Kg", "Weight (kg)"), ("cm", "Height (cm)"), ("BMI", "BMI (kg/m²)")]:
    g_all = all_pats[var].dropna()
    g0 = t0[var].dropna()
    g1 = t1[var].dropna()

    med_all = f"{g_all.median():.1f} [{g_all.quantile(.25):.1f}–{g_all.quantile(.75):.1f}]"
    med_0 = f"{g0.median():.1f} [{g0.quantile(.25):.1f}–{g0.quantile(.75):.1f}]"
    med_1 = f"{g1.median():.1f} [{g1.quantile(.25):.1f}–{g1.quantile(.75):.1f}]"

    _, p = mannwhitneyu(g0, g1, alternative="two-sided")
    p_str = f"{p:.3f}" if p >= 0.001 else "<0.001"
    print(f"{label:<25s}  {med_all:>18s}  {med_0:>18s}  {med_1:>18s}  {p_str:>8s}")

categorical_vars = [
    ("Gender", "Gender"),
    ("Type", "Device Type"),
    ("noActiveL", "No. of Active Leads"),
    ("noAbdnL", "No. of Abandoned Leads"),
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
        str_0 = f"{n_0} ({p_0:.1f}%)"
        str_1 = f"{n_1} ({p_1:.1f}%)"
        print(f"  {str(c):<23s}  {str_all:>18s}  {str_0:>18s}  {str_1:>18s}  {p_val:>8s}")
        p_val = ""

# ─── B1b. DEVICE TYPE BY ISABDN MATRIX ────────────────────────────────
print("\n" + "=" * 105)
print("B1b. DEVICE TYPE BY ISABDN STRATIFIED BY A_TEST COHORTS")
print("=" * 105)
print(f"{'Device Type':<16s} | {'A_Test = 0, isAbdn=0':^20s} {'A_Test = 0, isAbdn=1':^20s} | "
      f"{'A_Test = 1, isAbdn=0':^20s} {'A_Test = 1, isAbdn=1':^20s}")
print("-" * 105)

all_types = sorted(df_clean["Type"].dropna().unique())
t0_abd0 = t0[t0["isAbdn"] == 0]
t0_abd1 = t0[t0["isAbdn"] == 1]
t1_abd0 = t1[t1["isAbdn"] == 0]
t1_abd1 = t1[t1["isAbdn"] == 1]

for t in all_types:
    c0_0 = len(t0_abd0[t0_abd0["Type"] == t])
    p0_0 = (c0_0 / len(t0_abd0) * 100) if len(t0_abd0) > 0 else 0
    c0_1 = len(t0_abd1[t0_abd1["Type"] == t])
    p0_1 = (c0_1 / len(t0_abd1) * 100) if len(t0_abd1) > 0 else 0
    c1_0 = len(t1_abd0[t1_abd0["Type"] == t])
    p1_0 = (c1_0 / len(t1_abd0) * 100) if len(t1_abd0) > 0 else 0
    c1_1 = len(t1_abd1[t1_abd1["Type"] == t])
    p1_1 = (c1_1 / len(t1_abd1) * 100) if len(t1_abd1) > 0 else 0
    print(f"{t:<16s} | {c0_0:>5d} ({p0_0:>4.1f}%)     {c0_1:>5d} ({p0_1:>4.1f}%)     | "
          f"{c1_0:>5d} ({p1_0:>4.1f}%)     {c1_1:>5d} ({p1_1:>4.1f}%)")

print("-" * 105)
p_t0 = get_categorical_p(t0, "Type", "isAbdn")
p_t1 = get_categorical_p(t1, "Type", "isAbdn")
print(f"{'p-value (within split)':<16s} | {p_t0:^46s} | {p_t1:^46s}")

# ─── B2. FALSE POSITIVE (FP) vs TRUE NEGATIVE (TN) ────────────────────
test = df_clean[(df_clean["A_Test"] == 1) & (df_clean["isAbdn"] == 0)].copy()
test["FP_flag"] = test["FP"].astype(int)

fp = test[test["FP_flag"] == 1]
tn = test[test["FP_flag"] == 0]

print("\n" + "=" * 82)
print(f"B2. FALSE POSITIVE (FP) vs TRUE NEGATIVE (TN)  (Test set, GT-negative)")
print(f"    FP n={len(fp)}, TN n={len(tn)}")
print("=" * 82)
print(f"{'Variable':<27s}  {'FP (n='+str(len(fp))+')':^27s}  {'TN (n='+str(len(tn))+')':^27s}  {'p':>7s}")
print("-" * 95)

for var, label in [("Age", "Age (yr)"), ("Kg", "Weight (kg)"), ("cm", "Height (cm)"), ("BMI", "BMI (kg/m²)")]:
    g1 = fp[var].dropna()
    g0 = tn[var].dropna()
    if len(g1) < 2 or len(g0) < 2:
        continue
    _, p = mannwhitneyu(g1, g0, alternative="two-sided")
    m1 = f"{g1.median():.1f} [{g1.quantile(.25):.1f}–{g1.quantile(.75):.1f}]"
    m0 = f"{g0.median():.1f} [{g0.quantile(.25):.1f}–{g0.quantile(.75):.1f}]"
    p_str = f"{p:.3f}" if p >= 0.001 else "<0.001"
    print(f"{label:<27s}  {m1:>27s}  {m0:>27s}  {p_str:>7s}")

comp_cats = [
    ("Type", "Device Type"),
    ("noActiveL", "No. of Active Leads"),
    ("noAbdnL", "No. of Abandoned Leads"),
    ("hasIntra", "Has Intracardiac Artifacts"),
    ("hasExtra", "Has Extracardiac Artifacts"),
]

for var, label in comp_cats:
    print_categorical_comparison(test, var, label, "FP_flag", 1, 0)

# ─── B3. ANY ERROR (FP+FN) vs CORRECT ─────────────────────────────────
test_all = df_clean[df_clean["A_Test"] == 1].copy()
test_all["error"] = ((test_all["FP"].astype(int) == 1) | (test_all["FN"].astype(int) == 1)).astype(int)

err = test_all[test_all["error"] == 1]
corr = test_all[test_all["error"] == 0]

print("\n" + "=" * 82)
print(f"B3. ANY ERROR (FP+FN) vs CORRECT  (Test set evaluation)")
print(f"    Error n={len(err)}, Correct n={len(corr)}")
print("=" * 82)
print(f"{'Variable':<27s}  {'Error (n='+str(len(err))+')':^27s}  {'Correct (n='+str(len(corr))+')':^27s}  {'p':>7s}")
print("-" * 95)

for var, label in [("Age", "Age (yr)"), ("Kg", "Weight (kg)"), ("cm", "Height (cm)"), ("BMI", "BMI (kg/m²)")]:
    g1 = err[var].dropna()
    g0 = corr[var].dropna()
    if len(g1) < 2 or len(g0) < 2:
        continue
    _, p = mannwhitneyu(g1, g0, alternative="two-sided")
    m1 = f"{g1.median():.1f} [{g1.quantile(.25):.1f}–{g1.quantile(.75):.1f}]"
    m0 = f"{g0.median():.1f} [{g0.quantile(.25):.1f}–{g0.quantile(.75):.1f}]"
    p_str = f"{p:.3f}" if p >= 0.001 else "<0.001"
    print(f"{label:<27s}  {m1:>27s}  {m0:>27s}  {p_str:>7s}")

for var, label in comp_cats:
    print_categorical_comparison(test_all, var, label, "error", 1, 0)