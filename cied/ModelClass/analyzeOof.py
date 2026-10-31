"""
analyze_oof.py
==============
Analyse pooled out-of-fold predictions from export_oof.py.

Produces (in --out_dir):
  1. summary_pooled.csv            pooled accuracy / balanced acc / macro-F1 with 95% CIs
  2. per_fold_metrics.csv          per-fold metrics + mean +/- SD
  3. per_class_metrics.csv         precision / recall (Wilson CI) / F1 / support
  4. confusion_matrix_pooled.csv/.png
  5. manufacturer_level.csv        collapse 13 classes -> manufacturer groups
  6. misclassified_cases.csv       every error (ID, fold, true -> pred, confidence)
  7. learning_curves_all_folds.png (if --models_dir given)
  8. patient leakage check         (if oof has a 'patient' column)
  9. McNemar test vs another OOF   (if --compare given; for ablations)

Usage:
  python analyze_oof.py --oof C:/CIEDID_data/ModelClass/oof_predictions.csv \
      --out_dir C:/CIEDID_data/ModelClass/analysis \
      --models_dir C:/CIEDID_data/ModelClass/models \
      [--compare C:/CIEDID_data/ModelClass/oof_imagenet.csv]
"""
import argparse
import io
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)


# ---------------------------------------------------------------- helpers
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return centre - half, centre + half


def manufacturer(label):
    """ABT_HV -> ABT, BSX_Pacing -> BSX, ... ; 'Leadless' spans manufacturers, kept separate."""
    return label if label == "Leadless" else label.split("_")[0]


def bootstrap_ci(y_true, y_pred, labels, fn, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(fn(y_true[idx], y_pred[idx], labels))
    return np.percentile(vals, [2.5, 97.5])


def _bal(yt, yp, labels): return balanced_accuracy_score(yt, yp)
def _f1(yt, yp, labels): return f1_score(yt, yp, labels=labels, average="macro", zero_division=0)


def metrics_row(df, labels):
    yt, yp = df["true"].values, df["pred"].values
    return {
        "n": len(df),
        "accuracy": accuracy_score(yt, yp),
        "balanced_accuracy": balanced_accuracy_score(yt, yp),
        "macro_f1": f1_score(yt, yp, labels=labels, average="macro", zero_division=0),
        "n_errors": int((yt != yp).sum()),
    }


# ---------------------------------------------------------------- analyses
def pooled_summary(oof, labels, out):
    yt, yp = oof["true"].values, oof["pred"].values
    k, n = int((yt == yp).sum()), len(oof)
    lo, hi = wilson(k, n)
    bal_lo, bal_hi = bootstrap_ci(yt, yp, labels, _bal)
    f1_lo, f1_hi = bootstrap_ci(yt, yp, labels, _f1)
    s = pd.DataFrame([
        ["accuracy", k / n, lo, hi, "Wilson"],
        ["balanced_accuracy", balanced_accuracy_score(yt, yp), bal_lo, bal_hi, "bootstrap(2000)"],
        ["macro_f1", _f1(yt, yp, labels), f1_lo, f1_hi, "bootstrap(2000)"],
    ], columns=["metric", "value", "ci_low", "ci_high", "ci_method"])
    s.to_csv(out / "summary_pooled.csv", index=False)
    print(f"\n== Pooled OOF (n={n}, errors={n - k}) ==")
    print(s.round(4).to_string(index=False))


def per_fold(oof, labels, out):
    rows = []
    for f, g in oof.groupby("fold"):
        rows.append({"fold": f, **metrics_row(g, labels)})
    t = pd.DataFrame(rows)
    mean = t[["accuracy", "balanced_accuracy", "macro_f1"]].mean()
    sd = t[["accuracy", "balanced_accuracy", "macro_f1"]].std(ddof=1)
    t.to_csv(out / "per_fold_metrics.csv", index=False)
    print("\n== Per fold ==")
    print(t.round(4).to_string(index=False))
    print("mean +/- SD: " + "  ".join(f"{c}={mean[c]:.4f}+/-{sd[c]:.4f}" for c in mean.index))


def per_class(oof, labels, out):
    yt, yp = oof["true"].values, oof["pred"].values
    p, r, f, sup = precision_recall_fscore_support(yt, yp, labels=labels, zero_division=0)
    rows = []
    for i, c in enumerate(labels):
        tp = int(((yt == c) & (yp == c)).sum())
        lo, hi = wilson(tp, int(sup[i]))
        rows.append({"class": c, "support": int(sup[i]), "tp": tp,
                     "precision": p[i], "recall": r[i], "recall_ci_low": lo,
                     "recall_ci_high": hi, "f1": f[i]})
    t = pd.DataFrame(rows)
    t.to_csv(out / "per_class_metrics.csv", index=False)
    print("\n== Per class (pooled) ==")
    print(t.round(3).to_string(index=False))


def confusion(oof, labels, out):
    cm = confusion_matrix(oof["true"], oof["pred"], labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out / "confusion_matrix_pooled.csv")
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] > 0:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    fig.colorbar(im); plt.tight_layout()
    plt.savefig(out / "confusion_matrix_pooled.png", dpi=150); plt.close(fig)


def manufacturer_level(oof, out):
    m = oof.copy()
    m["true_m"] = m["true"].map(manufacturer)
    m["pred_m"] = m["pred"].map(manufacturer)
    k = int((m["true_m"] == m["pred_m"]).sum()); n = len(m)
    lo, hi = wilson(k, n)
    cross = m[m["true_m"] != m["pred_m"]]
    pd.DataFrame([{"n": n, "correct": k, "accuracy": k / n, "ci_low": lo, "ci_high": hi,
                   "cross_manufacturer_errors": len(cross)}]).to_csv(out / "manufacturer_level.csv", index=False)
    print(f"\n== Manufacturer-level (Leadless kept as its own group) ==")
    print(f"accuracy {k}/{n} = {k/n:.4f} (Wilson 95% CI {lo:.4f}-{hi:.4f}); "
          f"cross-manufacturer errors: {len(cross)}")
    return m


def misclassified(oof, labels, out):
    err = oof[oof["true"] != oof["pred"]].copy()
    pcols = {c: f"p_{c}" for c in labels}
    err["p_pred"] = [r[pcols[r["pred"]]] for _, r in err.iterrows()]
    err["p_true"] = [r[pcols[r["true"]]] for _, r in err.iterrows()]
    err["cross_manufacturer"] = err["true"].map(manufacturer) != err["pred"].map(manufacturer)
    keep = [c for c in ["ID", "fold", "patient", "true", "pred", "p_true", "p_pred",
                        "cross_manufacturer"] if c in err.columns]
    err[keep].sort_values(["true", "pred"]).to_csv(out / "misclassified_cases.csv", index=False)
    print(f"\n== Misclassified: {len(err)} cases -> misclassified_cases.csv ==")
    print(err.groupby(["true", "pred"]).size().sort_values(ascending=False).head(15).to_string())


def patient_leakage(oof):
    if "patient" not in oof.columns:
        print("\n(no patient column in OOF file; leakage check skipped. "
              "Re-run export_oof.py with --patient_col to enable.)")
        return
    folds_per_pt = oof.groupby("patient")["fold"].nunique()
    multi = folds_per_pt[folds_per_pt > 1]
    n_dup = int((oof.groupby("patient").size() > 1).sum())
    print(f"\n== Patient leakage check ==\npatients with >1 image: {n_dup}; "
          f"patients whose images fall in >1 fold validation sets: {len(multi)}")
    if len(multi):
        print("These patients also have images in other folds' TRAINING sets -> "
              "consider patient-grouped CV (GroupKFold / StratifiedGroupKFold).")
        print(multi.index.tolist())


def mcnemar(oof, other_path):
    o = pd.read_csv(other_path)
    mm = oof.merge(o, on="ID", suffixes=("_a", "_b"))
    assert (mm["true_a"] == mm["true_b"]).all(), "true labels differ between files"
    a_ok = mm["pred_a"] == mm["true_a"]
    b_ok = mm["pred_b"] == mm["true_b"]
    b = int((a_ok & ~b_ok).sum()); c = int((~a_ok & b_ok).sum())
    p = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) > 0 else 1.0
    print(f"\n== McNemar (exact), A=--oof vs B=--compare, n={len(mm)} ==\n"
          f"A correct & B wrong: {b}   A wrong & B correct: {c}   p = {p:.4f}")


def learning_curves(models_dir, n_folds, out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for f in range(n_folds):
        p = pathlib.Path(models_dir) / f"fold_{f}" / "training_history.csv"
        if not p.exists():
            continue
        lines = [l for l in p.read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.startswith("#")]
        df = pd.read_csv(io.StringIO("\n".join(lines)), on_bad_lines="skip")
        df = df[df["epoch"] != "epoch"].reset_index(drop=True)      # repeated headers (one per phase)
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce") if c != "time" else df[c]
        for ax, col in zip(axes, ["valid_loss", "accuracy", "balanced_accuracy_score"]):
            if col in df.columns:
                ax.plot(df.index + 1, df[col], label=f"fold {f}")
        if f == 0:
            boundary = int((df["epoch"].diff() < 0).idxmax()) if (df["epoch"].diff() < 0).any() else None
    for ax, col in zip(axes, ["valid_loss", "accuracy", "balanced_accuracy"]):
        ax.set_title(col); ax.set_xlabel("epoch (phase 1 + phase 2)"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out / "learning_curves_all_folds.png", dpi=150); plt.close(fig)
    print("\nSaved learning_curves_all_folds.png")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="C:/CIEDID_data/ModelClass/oof_predictions.csv")
    ap.add_argument("--out_dir", default="C:/CIEDID_data/ModelClass/analysis")
    ap.add_argument("--models_dir", default="C:/CIEDID_data/ModelClass/models")
    ap.add_argument("--compare", default=None,
                help="second OOF csv for McNemar (e.g. ImageNet ablation)")
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    oof = pd.read_csv(args.oof)
    oof["ID"] = oof["ID"].astype(str)
    labels = sorted(set(oof["true"]) | set(oof["pred"]))
    print(f"OOF rows={len(oof)}  unique IDs={oof['ID'].nunique()}  folds={sorted(oof['fold'].unique())}")

    pooled_summary(oof, labels, out)
    per_fold(oof, labels, out)
    per_class(oof, labels, out)
    confusion(oof, labels, out)
    manufacturer_level(oof, out)
    misclassified(oof, labels, out)
    patient_leakage(oof)
    if args.models_dir:
        learning_curves(args.models_dir, int(oof["fold"].max()) + 1, out)
    if args.compare:
        mcnemar(oof, args.compare)
    print(f"\nAll outputs in {out}")


if __name__ == "__main__":
    main()