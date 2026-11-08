"""
run_manuf_ablation.py  (Stage III: manufacturer classifier)
===========================================================
Runs finetuneManufCV.py (the patched V6 of your script) for three starting points,
identical in every other respect (folds, seed, epochs, lrs, augmentation, loss,
5-class output with Vitatron masked):

  A pretrained_full   pretrained body + pretrained head   (your original method)
  B encoder_newhead   pretrained body + fresh head
  C imagenet_newhead  ImageNet body    + fresh head

  A vs B = value of keeping the pretrained head
  B vs C = value of the CIED-specific body (the encoder-transfer claim)
  A vs C = overall value of the pretrained model

PRIMARY analysis uses FINAL-epoch predictions (val_predictions_phase2_finalepoch.csv);
the checkpoint-selected predictions (val_predictions_phase2.csv) are reported as a
secondary analysis so the size of the selection bias is visible.
The held-out pipeline test set (FinalTest == 1) is never used.

Usage (3 conditions x 5 folds x len(seeds) runs; resumes where it stopped):
  python run_manuf_ablation.py --inits encoder_newhead imagenet_newhead --seeds 42   # B and C only
  python run_manuf_ablation.py --train_script finetuneManufCV.py \
      --root C:/CIEDID_data/Manuf --seeds 42
  python run_manuf_ablation.py --skip_training --root C:/CIEDID_data/Manuf     # analyse only
Optional: --stageiv_oof C:/CIEDID_data/ModelClass/oof_encoder_s42.csv
  compares condition A (final epoch) with the manufacturer derived by collapsing the
  Stage IV 13-class prediction (Leadless cannot be mapped to a manufacturer -> counted wrong).

Outputs under --root:
  models_<init>_s<seed>/fold_k/...               (written by finetuneManufCV.py)
  oof_<init>_s<seed>_finalepoch.csv / _selected.csv   pooled out-of-fold predictions
  manuf_ablation_summary.csv, manuf_ablation_per_fold.csv, manuf_ablation_per_class.csv,
  manuf_ablation_mcnemar.csv
"""
import argparse
import itertools
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score, f1_score

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")  # Vitatron predictions are expected

INITS = ["pretrained_full", "encoder_newhead", "imagenet_newhead"]
PRESENT = ["Biotronik", "BSX", "Medtronic", "Abbott"]          # classes present in the data
SHORT = {"pretrained_full": "A", "encoder_newhead": "B", "imagenet_newhead": "C"}
PRED_FILES = {"finalepoch": "val_predictions_phase2_finalepoch.csv",
              "selected":   "val_predictions_phase2.csv"}


# ---------------------------------------------------------------- training
def run_one(args, init, seed, fold):
    out_dir = Path(args.root) / f"models_{init}_s{seed}"
    done = out_dir / f"fold_{fold}" / PRED_FILES["finalepoch"]
    if done.exists() and not args.force:
        print(f"[skip] {init} seed={seed} fold={fold}")
        return
    # Resolve the training script: as given, else next to this runner (fixes "can't open file"
    # when the command is launched from a different working directory, e.g. the VS Code root).
    script = Path(args.train_script)
    if not script.is_absolute() and not script.exists():
        script = Path(__file__).resolve().parent / script
    if not script.exists():
        raise FileNotFoundError(f"Training script not found: {script}. Put finetuneManufCV.py next to "
                                f"this runner or pass its full path with --train_script.")
    cmd = [sys.executable, str(script), "--init", init, "--fold", str(fold),
           "--seed", str(seed), "--output_dir", str(out_dir),
           "--model", args.model, "--labels_csv", args.labels_csv, "--crop_dir", args.crop_dir]
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------- pooling / metrics
def pool(root, init, seed, n_folds, kind):
    frames = []
    for f in range(n_folds):
        p = Path(root) / f"models_{init}_s{seed}" / f"fold_{f}" / PRED_FILES[kind]
        d = pd.read_csv(p)
        d.insert(0, "fold", f)
        frames.append(d)
    oof = pd.concat(frames, ignore_index=True)
    oof["ID"] = oof["ID"].astype(str)
    assert oof["ID"].is_unique, "an ID appears in more than one fold's validation set"
    oof.to_csv(Path(root) / f"oof_{init}_s{seed}_{kind}.csv", index=False)
    return oof


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return c - h, c + h


def metrics(df):
    yt, yp = df["true"].values, df["pred"].values
    k, n = int((yt == yp).sum()), len(df)
    lo, hi = wilson(k, n)
    return {"n": n, "n_errors": n - k, "accuracy": k / n, "acc_ci_low": lo, "acc_ci_high": hi,
            "balanced_accuracy": balanced_accuracy_score(yt, yp),
            "macro_f1": f1_score(yt, yp, labels=PRESENT, average="macro", zero_division=0),
            "pred_Vitatron": int((df["pred"] == "Vitatron(absent)").sum())}


def per_class(df):
    rows = []
    for c in PRESENT:
        sub = df[df["true"] == c]
        k = int((sub["pred"] == c).sum())
        lo, hi = wilson(k, len(sub))
        rows.append({"class": c, "support": len(sub), "sensitivity": k / len(sub) if len(sub) else np.nan,
                     "sens_ci_low": lo, "sens_ci_high": hi})
    return pd.DataFrame(rows)


def mcnemar(a, b):
    m = a[["ID", "true", "pred"]].merge(b[["ID", "true", "pred"]], on="ID", suffixes=("_a", "_b"))
    assert (m["true_a"] == m["true_b"]).all(), "true labels differ between conditions"
    a_ok, b_ok = m["pred_a"] == m["true_a"], m["pred_b"] == m["true_b"]
    n01 = int((a_ok & ~b_ok).sum()); n10 = int((~a_ok & b_ok).sum())
    p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if (n01 + n10) else 1.0
    return len(m), n01, n10, p


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i]); adj[i] = min(1.0, run)
    return adj


STAGEIV_TO_MANUF = {"ABT": "Abbott", "BIO": "Biotronik", "BSX": "BSX", "MDT": "Medtronic"}


def stageiv_compare(a_oof, path):
    s4 = pd.read_csv(path)
    s4["ID"] = s4["ID"].astype(str)
    s4["pred"] = s4["pred"].map(lambda c: STAGEIV_TO_MANUF.get(str(c).split("_")[0], "NOT_DERIVABLE"))
    s4 = s4[["ID", "pred"]].merge(a_oof[["ID", "true"]], on="ID")          # true manufacturer from Stage III file
    a = a_oof[["ID", "true", "pred"]]
    n, n01, n10, p = mcnemar(a, s4.assign(true=s4["true"]))
    acc_a = (a_oof["pred"] == a_oof["true"]).mean()
    acc_s4 = (s4["pred"] == s4["true"]).mean()
    print(f"\n== Stage III (reference condition, final epoch) vs manufacturer derived from Stage IV, n={n} common IDs ==")
    print(f"accuracy  Stage III: {acc_a:.4f}   Stage IV-derived: {acc_s4:.4f}   "
          f"(Stage IV predictions of 'Leadless' count as wrong)")
    print(f"Stage III right & IV wrong: {n01}   Stage III wrong & IV right: {n10}   exact McNemar p = {p:.4g}")
    print("Note: the two pipelines used different fold splits, but every prediction is out-of-fold, "
          "so the paired comparison is valid.")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_script", default="finetuneManufCV.py")
    ap.add_argument("--root", default="C:/CIEDID_data/Manuf/20epochs_1e-4")
    ap.add_argument("--model", default="C:/CIEDID_data/pkl/classification_manuf.pkl")
    ap.add_argument("--labels_csv", default="C:/CIEDID_data/main_workbook.csv")
    ap.add_argument("--crop_dir", default="C:/CIEDID_data/Images_crop")
    ap.add_argument("--n_folds", type=int, default=5) #default 5
    ap.add_argument("--inits", nargs="+", choices=INITS, default=["encoder_newhead", "imagenet_newhead"],
                    help="conditions to run/analyse, e.g. INITS, encoder_newhead, imagenet_newhead (A, B, C)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--stageiv_oof", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip_training", action="store_true")
    args = ap.parse_args()

    if not args.skip_training:
        for seed in args.seeds:
            for init in args.inits:
                for fold in range(args.n_folds):
                    run_one(args, init, seed, fold)

    summ, pf, pc, mc = [], [], [], []
    for seed in args.seeds:
        for kind in ("finalepoch", "selected"):
            oofs = {i: pool(args.root, i, seed, args.n_folds, kind) for i in args.inits}
            for i, d in oofs.items():
                summ.append({"seed": seed, "analysis": kind, "condition": f"{SHORT[i]} {i}", **metrics(d)})
                for f, g in d.groupby("fold"):
                    pf.append({"seed": seed, "analysis": kind, "condition": SHORT[i], "fold": f, **metrics(g)})
                if kind == "finalepoch":
                    pc.append(per_class(d).assign(seed=seed, condition=f"{SHORT[i]} {i}"))
            pairs = [pr for pr in [("pretrained_full", "encoder_newhead"),
                                   ("encoder_newhead", "imagenet_newhead"),
                                   ("pretrained_full", "imagenet_newhead")]
                     if pr[0] in oofs and pr[1] in oofs]
            res = [(a, b, *mcnemar(oofs[a], oofs[b])) for a, b in pairs]
            adj = holm(np.array([r[5] for r in res]))
            for (a, b, n, n01, n10, p), pa in zip(res, adj):
                mc.append({"seed": seed, "analysis": kind, "comparison": f"{SHORT[a]} vs {SHORT[b]}",
                           "n": n, "first_right_second_wrong": n01, "first_wrong_second_right": n10,
                           "p_exact": p, "p_holm": pa})
            if kind == "finalepoch" and args.stageiv_oof and seed == args.seeds[0]:
                ref = next(i for i in INITS if i in oofs)          # A if present, else B, else C
                print(f"(Stage III reference for the Stage IV comparison: {SHORT[ref]} {ref})")
                stageiv_compare(oofs[ref], args.stageiv_oof)

    S, F, C, M = pd.DataFrame(summ), pd.DataFrame(pf), pd.concat(pc), pd.DataFrame(mc)
    # paired per-fold differences
    wide = F.pivot_table(index=["seed", "analysis", "fold"], columns="condition",
                         values=["accuracy", "balanced_accuracy"]).reset_index()
    wide.columns = ["_".join(c).strip("_") for c in wide.columns]
    for m in ("accuracy", "balanced_accuracy"):
        if {f"{m}_B", f"{m}_C"} <= set(wide.columns):
            wide[f"d_{m}_B_minus_C"] = wide[f"{m}_B"] - wide[f"{m}_C"]
        if {f"{m}_A", f"{m}_B"} <= set(wide.columns):
            wide[f"d_{m}_A_minus_B"] = wide[f"{m}_A"] - wide[f"{m}_B"]
    root = Path(args.root)
    S.to_csv(root / "manuf_ablation_summary.csv", index=False)
    wide.to_csv(root / "manuf_ablation_per_fold.csv", index=False)
    C.to_csv(root / "manuf_ablation_per_class.csv", index=False)
    M.to_csv(root / "manuf_ablation_mcnemar.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n== Pooled out-of-fold metrics ==")
    print(S.round(4).to_string(index=False))
    print("\n== McNemar (exact; Holm-adjusted across the comparisons run) ==")
    print(M.round(6).to_string(index=False))
    print("\n== Paired per-fold differences, FINAL epoch (mean +/- SD across folds) ==")
    fe = wide[wide["analysis"] == "finalepoch"]
    for col in [c for c in wide.columns if c.startswith("d_")]:
        print(f"  {col:34s} {fe.groupby('seed')[col].mean().mean():+.4f} +/- {fe.groupby('seed')[col].std().mean():.4f}")
    print("\n== Per-class sensitivity (final epoch) ==")
    print(C.round(3).to_string(index=False))
    print("\nReading guide: report FINAL-epoch results as primary. Biotronik/Abbott have few images, "
          "so their per-class CIs are wide. If B vs C is not significant, report that encoder "
          "transfer did not help this task; if C is under-trained (check training_history.csv), "
          "say so and strengthen the baseline before claiming a benefit.")


if __name__ == "__main__":
    main()