#!/usr/bin/env python3
"""
score_heldout.py  --  Stage V scoring of the independent pipeline test set (FinalTest == 1)

This script does NOT run any model. It takes the frozen predictions that your inference
scripts wrote to one CSV and scores them against main_workbook.csv
(isAbandon / Manuf / ModelClass). Ground truth is only read here, at scoring time.

Predictions CSV (one row per test image, 324 rows):
    ID               workbook ID (also accepts "1612_crop" / "1612_crop.png")
    abdn_px          pixels with P(abandoned_lead) > 0.80 at the SAME resolution/preprocessing
                     used when the operating point was selected (prob>0.80, area>600)
    manuf_pred       BSX / Medtronic / Abbott / Biotronik
                     (pretrained vocab names "Boston Scientific" / "ST. Jude Medical" are remapped)
    modelclass_pred  e.g. MDT_Pacing, BSX_CRTD, Leadless, ... (or "Other" if classes were merged)
    crop_status      optional: "auto" or "manual" (manual = human-rescued crop)

Usage:
    python score_heldout.py --workbook main_workbook.csv --make_template preds_template.csv
    python score_heldout.py --workbook main_workbook.csv --pred preds.csv --out results_heldout \
        [--abdn_px_thr 600] [--other_classes ABT_HV,BSX_scICD]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Z = 1.959963984540054
MANUF_ALIAS = {"boston scientific": "BSX", "bsx": "BSX", "st. jude medical": "Abbott",
               "st jude medical": "Abbott", "abbott": "Abbott", "medtronic": "Medtronic",
               "mdt": "Medtronic", "biotronik": "Biotronik"}
PREFIX2MANUF = {"BSX": "BSX", "MDT": "Medtronic", "ABT": "Abbott", "BIO": "Biotronik"}
CRT_TYPES = {"CRTD", "CRTP"}
INVALID = "__invalid__"


# ───────────────────────── statistics helpers ─────────────────────────
def wilson(k, n, z=Z):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def prop(k, n):
    lo, hi = wilson(k, n)
    return {"k": int(k), "n": int(n), "est": (k / n if n else float("nan")), "lo": lo, "hi": hi}


def fmt(p):
    return f"{p['est']:.3f} ({p['k']}/{p['n']}; 95% CI {p['lo']:.3f}-{p['hi']:.3f})"


def binary_metrics(y, p):
    y = np.asarray(y).astype(bool)
    p = np.asarray(p).astype(bool)
    tp, fn = int((y & p).sum()), int((y & ~p).sum())
    fp, tn = int((~y & p).sum()), int((~y & ~p).sum())
    den = 2 * tp + fp + fn
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "sensitivity": prop(tp, tp + fn), "specificity": prop(tn, tn + fp),
            "PPV": prop(tp, tp + fp), "NPV": prop(tn, tn + fn),
            "F1": (2 * tp / den if den else float("nan")),
            "prevalence": prop(tp + fn, len(y)),
            "cases_reviewed_per_true_positive": ((tp + fp) / tp if tp else float("nan"))}


def auc_with_boot(y, s, n_boot=2000, seed=42):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)

    def auc(yy, ss):
        r = stats.rankdata(ss)
        n1 = yy.sum()
        n0 = len(yy) - n1
        return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

    if y.sum() == 0 or y.sum() == len(y):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        boots.append(auc(y[idx], s[idx]))
    return auc(y, s), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def group_p(groups, outcome, n_perm=10000, seed=42):
    """Fisher exact for 2 groups; Monte-Carlo permutation chi-square (exact-test style) for r x c."""
    g = pd.factorize(pd.Series(list(groups)))[0]
    o = np.asarray(outcome).astype(int)
    k = g.max() + 1
    if k < 2 or o.min() == o.max():
        return float("nan")
    if k == 2:
        tab = [[int(((g == i) & (o == j)).sum()) for j in (1, 0)] for i in (0, 1)]
        return float(stats.fisher_exact(tab)[1])
    n = len(o)
    gs = np.bincount(g, minlength=k).astype(float)

    def chi(oo):
        pos = np.bincount(g, weights=oo, minlength=k)
        tot = oo.sum()
        e_pos, e_neg = gs * tot / n, gs * (n - tot) / n
        return (((pos - e_pos) ** 2 / e_pos) + (((gs - pos) - e_neg) ** 2 / e_neg)).sum()

    obs = chi(o)
    rng = np.random.default_rng(seed)
    hits = sum(chi(rng.permutation(o)) >= obs - 1e-12 for _ in range(n_perm))
    return (hits + 1) / (n_perm + 1)


# ───────────────────────── multi-class report ─────────────────────────
def multiclass(y_true, y_pred, labels):
    labels = list(labels)
    yt = np.asarray(y_true, str)
    yp = np.asarray(y_pred, str)
    yp = np.where(np.isin(yp, labels), yp, INVALID)
    cols = labels + ([INVALID] if (yp == INVALID).any() else [])
    li = {l: i for i, l in enumerate(labels)}
    ci = {c: i for i, c in enumerate(cols)}
    cm = np.zeros((len(labels), len(cols)), int)
    for t, p in zip(yt, yp):
        cm[li[t], ci[p]] += 1
    n = cm.sum()
    rows = []
    for i, l in enumerate(labels):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = n - tp - fn - fp
        den = 2 * tp + fp + fn
        rows.append({"class": l, "support": int(tp + fn), "TP": int(tp), "FP": int(fp), "FN": int(fn),
                     "sensitivity": prop(tp, tp + fn), "specificity": prop(tn, tn + fp),
                     "PPV": prop(tp, tp + fp), "NPV": prop(tn, tn + fn),
                     "F1": (2 * tp / den if den else float("nan"))})
    present = [r for r in rows if r["support"] > 0]
    out = {"accuracy": prop(np.trace(cm[:, :len(labels)]), n),
           "balanced_accuracy": float(np.mean([r["sensitivity"]["est"] for r in present])),
           "macro_F1": float(np.mean([r["F1"] for r in present])),
           "n_invalid_predictions": int((yp == INVALID).sum())}
    pc = pd.DataFrame([{"class": r["class"], "support": r["support"], "TP": r["TP"], "FP": r["FP"],
                        "FN": r["FN"], "sens": r["sensitivity"]["est"], "sens_lo": r["sensitivity"]["lo"],
                        "sens_hi": r["sensitivity"]["hi"], "spec": r["specificity"]["est"],
                        "spec_lo": r["specificity"]["lo"], "spec_hi": r["specificity"]["hi"],
                        "PPV": r["PPV"]["est"], "NPV": r["NPV"]["est"], "F1": r["F1"]} for r in rows])
    return out, pc, pd.DataFrame(cm, index=labels, columns=cols)


def plot_cm(cm, title, path):
    norm = cm.values / np.maximum(cm.values.sum(axis=1, keepdims=True), 1)
    k = max(len(cm.index), len(cm.columns))
    fig, ax = plt.subplots(figsize=(max(4.5, 0.62 * k + 2), max(4, 0.55 * k + 1.5)))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cm.columns)))
    ax.set_xticklabels(cm.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cm.index)))
    ax.set_yticklabels(cm.index, fontsize=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm.values[i, j]
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reference (workbook)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ───────────────────────── data loading ─────────────────────────
def norm_id(s):
    s = re.sub(r"(_crop)?(\.\w+)?$", "", str(s).strip())
    m = re.search(r"(\d+)$", s)
    return m.group(1) if m else s


def load_truth(path):
    # the workbook carries ~16,000 empty "Unnamed" columns; only the first 14 are real
    df = pd.read_csv(path, usecols=range(14), low_memory=False)
    df["ID"] = df["ID"].map(norm_id)
    df["Manuf"] = df["Manuf"].str.strip()
    df["ModelClass"] = df["ModelClass"].str.strip()
    t = df[df["FinalTest"] == 1].copy().reset_index(drop=True)
    t["BMI"] = t["Kg"] / (t["cm"] / 100) ** 2
    t["Sex"] = np.where(t["Gender"].astype(str).str.contains("female", case=False), "female", "male")
    t["AgeGroup"] = pd.cut(t["Age"], [0, 65, 75, 85, 200], right=False,
                           labels=["<65", "65-74", "75-84", ">=85"]).astype(object)
    t["BMIGroup"] = pd.cut(t["BMI"], [0, 18.5, 25, 30, 100], right=False,
                           labels=["<18.5", "18.5-24.9", "25.0-29.9", ">=30"]).astype(object)
    t["IsCRT"] = t["Type"].isin(CRT_TYPES)
    return t


def derive_manuf(mc):
    return PREFIX2MANUF.get(str(mc).split("_")[0])


def leak_check(dirs, test_ids, all_ids):
    hits, matched = {}, 0
    for d in dirs:
        for f in Path(d).glob("*"):
            if not f.is_file():
                continue
            m = re.search(r"(\d+)", f.stem)
            if not m:
                continue
            i = m.group(1)
            if i in all_ids:
                matched += 1
                if i in test_ids:
                    hits.setdefault(str(d), []).append(f.name)
    return hits, matched


# ───────────────────────── main ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workbook", default="C:/CIEDID_data/main_workbook.csv")
    ap.add_argument("--pred", default="C:/CIEDID_data/finalTest/preds.csv")
    ap.add_argument("--out", default="C:/CIEDID_data/finalTest/results_heldout")
    ap.add_argument("--abdn_px_thr", type=int, default=600, help="pre-specified: pixels > 600 at prob > 0.80")
    ap.add_argument("--other_classes", default="", help="comma-separated ModelClass labels merged into 'Other' in training")
    ap.add_argument("--make_template", default="", help="write a CSV template with the test IDs and empty prediction columns")
    ap.add_argument("--leak_check_dirs", nargs="*", default=[],
                    help="training/validation image folders whose file stems contain the workbook ID")
    ap.add_argument("--n_perm", type=int, default=10000)
    a = ap.parse_args()

    truth = load_truth(a.workbook)
    if a.make_template:
        tmpl = pd.DataFrame({"ID": truth["ID"], "abdn_px": "", "manuf_pred": "",
                             "modelclass_pred": "", "crop_status": "auto"})
        tmpl.to_csv(a.make_template, index=False)
        print(f"template with {len(tmpl)} test IDs -> {a.make_template}")
        return
    if not a.pred:
        sys.exit("--pred is required")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = []

    def say(s=""):
        print(s)
        log.append(s)

    # ── integrity checks ─────────────────────────────────────────
    pred = pd.read_csv(a.pred)
    pred["ID"] = pred["ID"].map(norm_id)
    for c in ["abdn_px", "manuf_pred", "modelclass_pred"]:
        if c not in pred.columns:
            sys.exit(f"missing column in predictions: {c}")
    if pred["ID"].duplicated().any():
        sys.exit(f"duplicate IDs in predictions: {pred.loc[pred['ID'].duplicated(), 'ID'].tolist()}")
    extra = set(pred["ID"]) - set(truth["ID"])
    if extra:
        sys.exit(f"{len(extra)} predicted IDs are NOT in FinalTest==1 (test-set contamination?): {sorted(extra)[:10]}")
    missing = set(truth["ID"]) - set(pred["ID"])
    if missing:
        sys.exit(f"{len(missing)} test IDs have no prediction: {sorted(missing)[:10]}")
    if pred[["abdn_px", "manuf_pred", "modelclass_pred"]].isna().any().any():
        sys.exit("NaN/empty values in prediction columns")
    if "crop_status" not in pred.columns:
        pred["crop_status"] = "auto"
    pred["crop_status"] = pred["crop_status"].astype(str).str.strip().str.lower()

    if a.leak_check_dirs:
        hits, matched = leak_check(a.leak_check_dirs, set(truth["ID"]), set(load_truth(a.workbook)["ID"]) | set(truth["ID"]))
        say(f"[leak check] {matched} file stems matched a workbook ID")
        if hits:
            for d, fs in hits.items():
                say(f"  WARNING {d}: {len(fs)} files carry FinalTest==1 IDs, e.g. {fs[:5]}")
            sys.exit("held-out IDs found in training/validation folders; fix before scoring")
        say("  no FinalTest==1 IDs found in the listed folders")

    df = truth.merge(pred, on="ID", how="left", validate="one_to_one")
    df["manuf_pred"] = df["manuf_pred"].astype(str).str.strip().map(lambda s: MANUF_ALIAS.get(s.lower(), s))
    df["modelclass_pred"] = df["modelclass_pred"].astype(str).str.strip().replace({"Others": "Other"})
    df["abdn_px"] = pd.to_numeric(df["abdn_px"], errors="raise")

    other = [c for c in a.other_classes.split(",") if c]
    df["mc_true"] = df["ModelClass"].where(~df["ModelClass"].isin(other), "Other")

    say(f"Test set: n={len(df)}  |  abandoned lead (isAbandon=1): {int(df.isAbandon.sum())} "
        f"({df.isAbandon.mean():.1%})")
    say(f"Manuf: {df.Manuf.value_counts().to_dict()}")
    say(f"ModelClass (reference): {df.mc_true.value_counts().to_dict()}")
    miss = df[["Age", "Kg", "cm"]].isna().sum().to_dict()
    say(f"Missing covariates in test set: {miss}")
    n_manual = int((df.crop_status == "manual").sum())
    say(f"Crop status: auto={len(df) - n_manual}  manual(rescued)={n_manual}")
    auto_pass = prop(len(df) - n_manual, len(df))
    say(f"Automatic crop pass rate: {fmt(auto_pass)}")
    summary = {"n": len(df), "crop_auto_pass": auto_pass}

    # ── Stage I: abandoned lead, case level ──────────────────────
    say("\n=== Abandoned-lead detection (case level) ===")
    say(f"Operating point (pre-specified): abdn_px > {a.abdn_px_thr}")
    df["abdn_pred"] = (df["abdn_px"] > a.abdn_px_thr).astype(int)
    bm = binary_metrics(df["isAbandon"], df["abdn_pred"])
    auc, alo, ahi = auc_with_boot(df["isAbandon"], df["abdn_px"], seed=42)
    for k in ["sensitivity", "specificity", "PPV", "NPV"]:
        say(f"{k:>11}: {fmt(bm[k])}")
    say(f"{'F1':>11}: {bm['F1']:.3f}")
    say(f"Confusion: TP={bm['TP']} FN={bm['FN']} FP={bm['FP']} TN={bm['TN']}")
    say(f"Cases flagged per true positive: {bm['cases_reviewed_per_true_positive']:.1f}")
    say(f"AUC of abdn_px (descriptive; fixed prob>0.80): {auc:.3f} (bootstrap 95% CI {alo:.3f}-{ahi:.3f})")
    summary["abandoned_lead"] = {**bm, "AUC": auc, "AUC_lo": alo, "AUC_hi": ahi}
    err = df[df["isAbandon"] != df["abdn_pred"]][["ID", "isAbandon", "abdn_px", "Type", "Model", "ModelClass"]]
    err.assign(error=np.where(err.isAbandon == 1, "FN", "FP")).to_csv(out / "abandoned_errors.csv", index=False)

    # pre-specified follow-up of Stage I error analysis: FP rate in CRT vs non-CRT (reference-negative cases)
    neg = df[df.isAbandon == 0]
    a_crt = int(((neg.IsCRT) & (neg.abdn_pred == 1)).sum())
    n_crt = int(neg.IsCRT.sum())
    b_non = int(((~neg.IsCRT) & (neg.abdn_pred == 1)).sum())
    n_non = int((~neg.IsCRT).sum())
    p_crt = stats.fisher_exact([[a_crt, n_crt - a_crt], [b_non, n_non - b_non]])[1]
    say(f"FP rate, CRT (CRT-D/CRT-P) {fmt(prop(a_crt, n_crt))} vs non-CRT {fmt(prop(b_non, n_non))}; Fisher p={p_crt:.4f}")
    ft = neg.groupby("Type")["abdn_pred"].agg(["size", "sum"]).rename(columns={"size": "n_negatives", "sum": "FP"})
    ft["FP_rate"] = ft["FP"] / ft["n_negatives"]
    ft.to_csv(out / "abandoned_FP_by_type.csv")

    # ── Stage III: manufacturer ──────────────────────────────────
    say("\n=== Manufacturer (Stage III) ===")
    m_labels = sorted(df["Manuf"].unique())
    m_sum, m_pc, m_cm = multiclass(df["Manuf"], df["manuf_pred"], m_labels)
    say(f"Accuracy: {fmt(m_sum['accuracy'])}")
    say(f"Balanced accuracy: {m_sum['balanced_accuracy']:.3f}   Macro-F1: {m_sum['macro_F1']:.3f}   "
        f"invalid predictions (e.g. Vitatron): {m_sum['n_invalid_predictions']}")
    say(m_pc.round(3).to_string(index=False))
    m_pc.to_csv(out / "manuf_per_class.csv", index=False)
    m_cm.to_csv(out / "manuf_confusion.csv")
    plot_cm(m_cm, "Manufacturer - independent test set", out / "manuf_confusion.png")
    df[df.Manuf != df.manuf_pred][["ID", "Manuf", "manuf_pred", "ModelClass", "Model", "crop_status"]] \
        .to_csv(out / "manuf_errors.csv", index=False)
    summary["manufacturer"] = {**m_sum}

    # ── Stage IV: ModelClass ─────────────────────────────────────
    say("\n=== ModelClass (Stage IV; manufacturer x device category, NOT exact model) ===")
    c_labels = sorted(df["mc_true"].unique())
    c_sum, c_pc, c_cm = multiclass(df["mc_true"], df["modelclass_pred"], c_labels)
    say(f"Accuracy: {fmt(c_sum['accuracy'])}")
    say(f"Balanced accuracy: {c_sum['balanced_accuracy']:.3f}   Macro-F1: {c_sum['macro_F1']:.3f}   "
        f"invalid predictions: {c_sum['n_invalid_predictions']}")
    say(c_pc.round(3).to_string(index=False))
    c_pc.to_csv(out / "modelclass_per_class.csv", index=False)
    c_cm.to_csv(out / "modelclass_confusion.csv")
    plot_cm(c_cm, "ModelClass - independent test set", out / "modelclass_confusion.png")
    df[df.mc_true != df.modelclass_pred][["ID", "mc_true", "modelclass_pred", "Model", "Type", "crop_status"]] \
        .to_csv(out / "modelclass_errors.csv", index=False)
    summary["modelclass"] = {**c_sum}

    # consistency between the two classifiers (Leadless / Other have no unique manufacturer)
    der = df["modelclass_pred"].map(derive_manuf)
    ok = der.notna()
    conc = prop(int((der[ok] == df.loc[ok, "manuf_pred"]).sum()), int(ok.sum()))
    der_acc = prop(int((der[ok] == df.loc[ok, "Manuf"]).sum()), int(ok.sum()))
    say(f"\nStage III vs Stage IV manufacturer concordance (n={int(ok.sum())}, excl. Leadless/Other): {fmt(conc)}")
    say(f"Manufacturer derived from ModelClass prediction vs reference: {fmt(der_acc)}")
    summary["consistency"] = {"concordance": conc, "derived_manuf_accuracy": der_acc}

    # ── sensitivity analysis: exclude human-rescued crops ────────
    if n_manual:
        d2 = df[df.crop_status != "manual"]
        say(f"\n=== Sensitivity analysis: excluding {n_manual} human-rescued crop(s) (n={len(d2)}) ===")
        say(f"Manufacturer accuracy: {fmt(prop(int((d2.Manuf == d2.manuf_pred).sum()), len(d2)))}")
        say(f"ModelClass accuracy:   {fmt(prop(int((d2.mc_true == d2.modelclass_pred).sum()), len(d2)))}")

    # ── subgroup analysis (age / sex / BMI) ──────────────────────
    say("\n=== Subgroup analysis ===")
    say("Fisher exact (2 groups) or Monte-Carlo permutation chi-square (r x c). Sensitivity is not "
        "tested by subgroup: only %d reference-positive cases." % int(df.isAbandon.sum()))
    outcomes = {
        "abandoned_FP (reference-negative only)": (df[df.isAbandon == 0], lambda d: (d.abdn_pred == 1).astype(int)),
        "manufacturer_correct": (df, lambda d: (d.Manuf == d.manuf_pred).astype(int)),
        "modelclass_correct": (df, lambda d: (d.mc_true == d.modelclass_pred).astype(int)),
    }
    rows = []
    for oname, (sub, fn) in outcomes.items():
        for var in ["AgeGroup", "Sex", "BMIGroup"]:
            s = sub.dropna(subset=[var])
            y = fn(s)
            p = group_p(s[var].values, y.values, n_perm=a.n_perm)
            for g, idx in s.groupby(var).groups.items():
                k = int(y.loc[idx].sum())
                pr = prop(k, len(idx))
                rows.append({"outcome": oname, "variable": var, "group": g, "n": len(idx), "events": k,
                             "rate": pr["est"], "lo": pr["lo"], "hi": pr["hi"], "p_overall": p,
                             "n_excluded_missing": len(sub) - len(s)})
    sg = pd.DataFrame(rows)
    sg.to_csv(out / "subgroups.csv", index=False)
    say(sg.round(3).to_string(index=False))

    df.drop(columns=["BMI"]).to_csv(out / "scored_cases.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (out / "report.txt").write_text("\n".join(log), encoding="utf-8")
    print(f"\nAll outputs written to {out.resolve()}")


if __name__ == "__main__":
    main()