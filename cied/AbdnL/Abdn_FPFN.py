"""
Abandoned Lead Detection — FP / FN Error Analysis
Threshold: prob > 0.8, pixel > 600
"""

import pandas as pd, numpy as np, re
from scipy.stats import fisher_exact

# ─── Load & prep ─────────────────────────────────────────────────────────────
res  = pd.read_csv("C:/CIEDID_data/AbdnL/FPFN/oof_per_image_results.csv")
meta = pd.read_csv("C:/CIEDID_data/AbdnL/FPFN/A_model_detail.csv")

# Extract image ID (handle both .png and .jpeg)
res["img_id"] = res["image"].str.extract(r"([^\\\/]+)\.(png|jpe?g)$", expand=False)[0]

def norm_id(s):
    s = str(s).strip()
    s = re.sub(r"^a_", "", s)   # strip a_ prefix
    return s

res["img_id_norm"]  = res["img_id"].apply(norm_id)
meta_test = meta[meta["A_Test"] == 1].copy()
meta_test["img_id_norm"] = meta_test["ID"].astype(str).apply(norm_id)

# Apply threshold: prob > 0.80 AND pixel_count > 600
res["pred_pos"] = (res["prob_80"] > 600).astype(int)
res["gt_pos"]   = res["has_abandoned"].astype(int)

def outcome(row):
    if row["gt_pos"]==1 and row["pred_pos"]==1: return "TP"
    if row["gt_pos"]==0 and row["pred_pos"]==1: return "FP"
    if row["gt_pos"]==1 and row["pred_pos"]==0: return "FN"
    return "TN"
res["outcome"] = res.apply(outcome, axis=1)

# Merge with metadata
merged = res.merge(meta_test, on="img_id_norm", how="left")

# Confounder flags
merged["has_intracardiac"]   = merged["hasIntra"].notna().astype(int)
merged["has_extracardiac"]   = merged["hasExtra"].notna().astype(int)
merged["is_CRT"]             = merged["Type"].isin(["CRTD","CRTP"]).astype(int)
merged["has_any_confounder"] = (
    merged["has_intracardiac"] | merged["has_extracardiac"] | merged["is_CRT"]
).astype(int)

# ─── 1. Overall confusion matrix ─────────────────────────────────────────────
print("=" * 65)
print("1. OVERALL CONFUSION MATRIX  (Test set n=50, threshold prob>0.80 & pixel>600)")
print("=" * 65)
TP = (merged["outcome"]=="TP").sum()
FP = (merged["outcome"]=="FP").sum()
FN = (merged["outcome"]=="FN").sum()
TN = (merged["outcome"]=="TN").sum()
pos, neg = merged["gt_pos"].sum(), (merged["gt_pos"]==0).sum()

print(f"\n{'':22s}  {'Pred POS':>10s}  {'Pred NEG':>10s}  {'Total':>6s}")
print(f"{'GT POS (has abandoned)':22s}  {'TP='+str(TP):>10s}  {'FN='+str(FN):>10s}  {pos:>6d}")
print(f"{'GT NEG (no abandoned)':22s}  {'FP='+str(FP):>10s}  {'TN='+str(TN):>10s}  {neg:>6d}")
print(f"{'Total':22s}  {TP+FP:>10d}  {FN+TN:>10d}  {len(merged):>6d}")

def wilson_ci(successes, n, z=1.96):
    """Wilson score interval for a binomial proportion. Returns (lower, upper)."""
    if n == 0:
        return (float('nan'), float('nan'))
    phat   = successes / n
    denom  = 1 + (z**2) / n
    center = phat + (z**2) / (2*n)
    margin = z * np.sqrt((phat*(1-phat))/n + (z**2)/(4*n**2))
    lower  = (center - margin) / denom
    upper  = (center + margin) / denom
    return max(0.0, lower), min(1.0, upper)

sens = TP/pos; spec = TN/neg
ppv  = TP/(TP+FP) if (TP+FP)>0 else 0
npv  = TN/(TN+FN) if (TN+FN)>0 else 0
f1   = 2*TP/(2*TP+FP+FN) if (2*TP+FP+FN)>0 else 0
sens_ci = wilson_ci(TP, pos)
spec_ci = wilson_ci(TN, neg)
ppv_ci  = wilson_ci(TP, TP+FP)
npv_ci  = wilson_ci(TN, TN+FN)
print(f"\nSensitivity  : {TP}/{pos}  = {sens:.3f} [{sens_ci[0]:.3f}, {sens_ci[1]:.3f}]")
print(f"Specificity  : {TN}/{neg}  = {spec:.3f} [{spec_ci[0]:.3f}, {spec_ci[1]:.3f}]")
print(f"PPV          : {TP}/{TP+FP}  = {ppv:.3f} [{ppv_ci[0]:.3f}, {ppv_ci[1]:.3f}]")
print(f"NPV          : {TN}/{TN+FN}  = {npv:.3f} [{npv_ci[0]:.3f}, {npv_ci[1]:.3f}]")
print(f"F1 score     :            {f1:.3f}")

# ─── 2. FN cases detail ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("2. FALSE NEGATIVE CASES  (missed abandoned lead, n=" + str(FN) + ")")
print("=" * 65)
fn_cases = merged[merged["outcome"]=="FN"].copy()
print(fn_cases[["img_id","Type","noActiveL","noAbdnL","targ_px","pred_px","prob_80",
                 "Intracardiac device","Extracardiac type"]].to_string(index=False))

# ─── 3. FP cases detail ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("3. FALSE POSITIVE CASES  (false alarm, n=" + str(FP) + ")")
print("=" * 65)
fp_cases = merged[merged["outcome"]=="FP"].copy()
print(fp_cases[["img_id","Type","noActiveL","pred_px","prob_80",
                 "Intracardiac device","Extracardiac type"]].to_string(index=False))

# ─── 4. FP stratified by confounder ─────────────────────────────────────────
print("\n" + "=" * 65)
print("4. FP RATE BY CONFOUNDER  (GT-negative cases only, n=" + str(neg) + ")")
print("=" * 65)
neg_cases = merged[merged["gt_pos"]==0].copy()
confounders = {
    "CRT (CRTD/CRTP)"      : "is_CRT",
    "hasIntra"  : "has_intracardiac",
    "hasExtra": "has_extracardiac",
    "Any confounder"       : "has_any_confounder",
}
print(f"\n{'Subgroup':<25s}  {'With confounder':>16s}  {'Without confounder':>19s}  {'Fisher p':>9s}")
print(f"{'':25s}  {'n / FP / rate':>16s}  {'n / FP / rate':>19s}  {'':>9s}")
print("-" * 78)
for label, col in confounders.items():
    g1 = neg_cases[neg_cases[col]==1]
    g0 = neg_cases[neg_cases[col]==0]
    fp1, fp0 = (g1["outcome"]=="FP").sum(), (g0["outcome"]=="FP").sum()
    n1, n0   = len(g1), len(g0)
    r1 = f"{fp1}/{n1} ({fp1/n1:.0%})" if n1>0 else "—"
    r0 = f"{fp0}/{n0} ({fp0/n0:.0%})" if n0>0 else "—"
    tbl = [[fp1, n1-fp1],[fp0, n0-fp0]]
    _, p = fisher_exact(tbl) if n1>0 and n0>0 else (None, float("nan"))
    ps = f"{p:.3f}" if not np.isnan(p) else "—"
    print(f"{label:<25s}  {r1:>16s}  {r0:>19s}  {ps:>9s}")

# ─── 5. FN stratified by confounder ─────────────────────────────────────────
print("\n" + "=" * 65)
print("5. FN RATE BY CONFOUNDER  (GT-positive cases only, n=" + str(pos) + ")")
print("=" * 65)
pos_cases = merged[merged["gt_pos"]==1].copy()
print(f"\n{'Subgroup':<25s}  {'With confounder':>16s}  {'Without confounder':>19s}  {'Fisher p':>9s}")
print(f"{'':25s}  {'n / FN / rate':>16s}  {'n / FN / rate':>19s}  {'':>9s}")
print("-" * 78)
for label, col in confounders.items():
    g1 = pos_cases[pos_cases[col]==1]
    g0 = pos_cases[pos_cases[col]==0]
    fn1, fn0 = (g1["outcome"]=="FN").sum(), (g0["outcome"]=="FN").sum()
    n1, n0   = len(g1), len(g0)
    r1 = f"{fn1}/{n1} ({fn1/n1:.0%})" if n1>0 else "—"
    r0 = f"{fn0}/{n0} ({fn0/n0:.0%})" if n0>0 else "—"
    tbl = [[fn1, n1-fn1],[fn0, n0-fn0]]
    _, p = fisher_exact(tbl) if n1>0 and n0>0 else (None, float("nan"))
    ps = f"{p:.3f}" if not np.isnan(p) else "—"
    print(f"{label:<25s}  {r1:>16s}  {r0:>19s}  {ps:>9s}")

# ─── 6. FP breakdown by device type ─────────────────────────────────────────
print("\n" + "=" * 65)
print("6. FP BREAKDOWN BY DEVICE TYPE  (GT-negative cases)")
print("=" * 65)
tbl = neg_cases.groupby("Type").apply(
    lambda g: pd.Series({"n": len(g), "FP": (g["outcome"]=="FP").sum()})
).reset_index()
tbl["FP_rate"] = tbl["FP"] / tbl["n"]
tbl = tbl.sort_values("FP_rate", ascending=False)
print(tbl.to_string(index=False))

# ─── 7. FN breakdown by device type ─────────────────────────────────────────
print("\n" + "=" * 65)
print("7. FN BREAKDOWN BY DEVICE TYPE  (GT-positive cases)")
print("=" * 65)
tbl2 = pos_cases.groupby("Type", dropna=False).apply(
    lambda g: pd.Series({"n": len(g), "FN": (g["outcome"]=="FN").sum()})
).reset_index()
tbl2["FN_rate"] = tbl2["FN"] / tbl2["n"]
tbl2 = tbl2.sort_values("FN_rate", ascending=False)
print(tbl2.to_string(index=False))

# ─── 8. Pixel count analysis ─────────────────────────────────────────────────
print("\n" + "=" * 65)
print("8. PIXEL COUNT ANALYSIS  (prob_80 = pixel count at prob>0.80)")
print("=" * 65)
print(f"\n{'Outcome':<6s}  {'n':>3s}  {'prob_80 mean':>13s}  {'prob_80 median':>14s}  {'targ_px mean':>13s}")
print("-" * 58)
for oc in ["TP","FP","FN","TN"]:
    g = merged[merged["outcome"]==oc]
    if len(g)==0: continue
    print(f"{oc:<6s}  {len(g):>3d}  {g['prob_80'].mean():>13.0f}  {g['prob_80'].median():>14.0f}  {g['targ_px'].mean():>13.0f}")

# FN: show individual prob_80 vs targ_px vs threshold
print(f"\nFN cases — pixel detail (threshold = 600):")
print(fn_cases[["img_id","targ_px","pred_px","prob_80"]].to_string(index=False))