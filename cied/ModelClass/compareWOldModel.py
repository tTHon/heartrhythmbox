"""
run_ablation.py
===============
Ablation for the Stage IV classifier:

  A "encoder"  : backbone initialised from classification_model.pkl (encoder-only transfer)
  B "imagenet" : plain ImageNet-pretrained ResNet-50  (--pretrained_path "")

Everything else is IDENTICAL: folds (StratifiedKFold rs=42), epochs (8 + 20),
learning rates, augmentation, class weights, grouping, seed.
Primary comparison uses FINAL-epoch weights (oof_final_epoch.csv), pooled over
the 5 folds, with an exact McNemar test on paired per-image correctness.
The held-out set (FinalTest == 1) is never used.

Usage (trains 2 conditions x 5 folds x len(seeds) runs; resumes if interrupted):
  python run_ablation.py --train_script finetuneModelClass.py \
      --root C:/CIEDID_data/ModelClass --seeds 42 [43 44]  [--patient_col PatientID]

Outputs under --root:
  models_encoder_s42/fold_k/..., models_imagenet_s42/fold_k/...
  oof_encoder_s42.csv, oof_imagenet_s42.csv        (pooled final-epoch OOF)
  ablation_summary.csv                             (metrics + McNemar per seed)
  ablation_per_fold.csv                            (paired per-fold differences)
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score, f1_score

CONDITIONS = {
    "encoder":  None,   # filled from --encoder_pkl
    "imagenet": "",     # empty string -> plain ImageNet weights in the training script
}


def run_one(args, cond, seed, fold):
    out_dir = Path(args.root) / f"models_{cond}_s{seed}"
    done = out_dir / f"fold_{fold}" / "oof_final_epoch.csv"
    if done.exists() and not args.force:
        print(f"[skip] {cond} seed={seed} fold={fold} (already done)")
        return
    pkl = args.encoder_pkl if cond == "encoder" else CONDITIONS[cond]
    cmd = [sys.executable, args.train_script,
           "--labels_csv", args.labels_csv, "--crop_dir", args.crop_dir,
           "--output_dir", str(out_dir), "--pretrained_path", pkl,
           "--min_class_count", str(args.min_class_count),
           "--fold", str(fold), "--seed", str(seed)]
    if args.patient_col:
        cmd += ["--patient_col", args.patient_col]
    print("\n>>>", " ".join(f'"{c}"' if c == "" else c for c in cmd))
    subprocess.run(cmd, check=True)


def pool(root, cond, seed, n_folds):
    parts = []
    for f in range(n_folds):
        p = Path(root) / f"models_{cond}_s{seed}" / f"fold_{f}" / "oof_final_epoch.csv"
        parts.append(pd.read_csv(p))
    oof = pd.concat(parts, ignore_index=True)
    oof["ID"] = oof["ID"].astype(str)
    oof.to_csv(Path(root) / f"oof_{cond}_s{seed}.csv", index=False)
    return oof


def metrics(df):
    yt, yp = df["true"].values, df["pred"].values
    labels = sorted(set(yt))
    return {"n": len(df), "accuracy": (yt == yp).mean(),
            "balanced_accuracy": balanced_accuracy_score(yt, yp),
            "macro_f1": f1_score(yt, yp, labels=labels, average="macro", zero_division=0),
            "n_errors": int((yt != yp).sum())}


def mcnemar_exact(a, b):
    m = a.merge(b, on="ID", suffixes=("_a", "_b"))
    assert (m["true_a"] == m["true_b"]).all(), "true labels differ between conditions"
    a_ok, b_ok = m["pred_a"] == m["true_a"], m["pred_b"] == m["true_b"]
    n01 = int((a_ok & ~b_ok).sum())     # A right, B wrong
    n10 = int((~a_ok & b_ok).sum())     # A wrong, B right
    p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if (n01 + n10) else 1.0
    return n01, n10, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_script", default="finetuneModelClass.py")
    ap.add_argument("--root", default="C:/CIEDID_data/ModelClass")
    ap.add_argument("--labels_csv", default="C:/CIEDID_data/main_workbook.csv")
    ap.add_argument("--crop_dir", default="C:/CIEDID_data/Images_crop")
    ap.add_argument("--encoder_pkl", default="C:/CIEDID_data/pkl/classification_model.pkl")
    ap.add_argument("--min_class_count", type=int, default=1)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--patient_col", default=None)
    ap.add_argument("--force", action="store_true", help="retrain even if outputs exist")
    ap.add_argument("--skip_training", action="store_true", help="only pool + analyse existing runs")
    args = ap.parse_args()

    if not args.skip_training:
        for seed in args.seeds:
            for cond in CONDITIONS:
                for fold in range(args.n_folds):
                    run_one(args, cond, seed, fold)

    summary, per_fold = [], []
    for seed in args.seeds:
        enc = pool(args.root, "encoder", seed, args.n_folds)
        img = pool(args.root, "imagenet", seed, args.n_folds)
        for name, df in (("encoder", enc), ("imagenet", img)):
            summary.append({"seed": seed, "condition": name, **metrics(df)})
        n01, n10, p = mcnemar_exact(enc, img)
        summary.append({"seed": seed, "condition": "McNemar (encoder vs imagenet)",
                        "encoder_right_imagenet_wrong": n01,
                        "encoder_wrong_imagenet_right": n10, "p_exact": p})
        for f in range(args.n_folds):
            e, i = metrics(enc[enc.fold == f]), metrics(img[img.fold == f])
            per_fold.append({"seed": seed, "fold": f,
                             **{f"enc_{k}": e[k] for k in ("accuracy", "balanced_accuracy", "macro_f1")},
                             **{f"img_{k}": i[k] for k in ("accuracy", "balanced_accuracy", "macro_f1")},
                             "d_accuracy": e["accuracy"] - i["accuracy"],
                             "d_balanced_accuracy": e["balanced_accuracy"] - i["balanced_accuracy"]})

    S = pd.DataFrame(summary); F = pd.DataFrame(per_fold)
    S.to_csv(Path(args.root) / "ablation_summary.csv", index=False)
    F.to_csv(Path(args.root) / "ablation_per_fold.csv", index=False)
    print("\n== Pooled final-epoch OOF, per condition / seed ==")
    print(S.round(4).to_string(index=False))
    print("\n== Paired per-fold differences (encoder - imagenet) ==")
    print(F.round(4).to_string(index=False))
    d = F.groupby("seed")[["d_accuracy", "d_balanced_accuracy"]].agg(["mean", "std"])
    print("\nMean +/- SD of per-fold differences by seed:\n", d.round(4).to_string())
    if len(args.seeds) > 1:
        sd = S[S.condition.isin(["encoder", "imagenet"])].groupby("condition")[
            ["accuracy", "balanced_accuracy", "macro_f1"]].agg(["mean", "std"])
        print("\nAcross-seed mean +/- SD (pooled metrics):\n", sd.round(4).to_string())
    print("\nInterpretation guide: a non-significant McNemar p with overlapping per-fold "
          "differences means NO evidence that encoder transfer helps; report that as is.")


if __name__ == "__main__":
    main()