#!/usr/bin/env python
"""
pool_oof_stageIII.py
====================
Pools the per-fold validation predictions written by finetuneManufCV_dataeff.py
  <root>/<init>_frac<f>_seed<s>/fold_N/val_predictions_phase2_finalepoch.csv
into ONE file per run (columns: ID, fold, true, pred, p_<class>...), the same shape as the
Stage IV pooled files.   <init> = pretrained_full | encoder_newhead | imagenet_newhead | scratch.

No paths needed:   python pool_oof_stageIII.py
  - scans <root> (default C:/CIEDID_data/Manuf/models), INCLUDING sub-folders
  - also picks up a  zeroshot/  run (val_predictions_zeroshot.csv) if present
  - writes <root>/pooled_oof/oof_<arm>_[frac<f>_]s<seed>.csv
      arm names: encoder_newhead -> encoder, imagenet_newhead -> imagenet, pretrained_full -> pretrained
      e.g. oof_encoder_frac0.25_s42.csv, oof_imagenet_frac0.5_s42.csv

Options (rarely needed):
  --root D:/other/models     if the models moved (pointing at one run folder also works)
  --which selected           pool val_predictions_phase2.csv (checkpoint chosen on the fold itself)
                             instead of the final-epoch file (bias comparison only)
  --allow_partial            also write runs with fewer than 5 folds (off by default)
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUN_RE = re.compile(r"^(?P<init>.+)_frac(?P<frac>[0-9.]+)_seed(?P<seed>\d+)$")
ARM_NAME = {"encoder_newhead": "encoder", "imagenet_newhead": "imagenet", "pretrained_full": "pretrained"}
KEEP_NON_CLASS_P = {"p_pred", "p_true"}


def metrics(df):
    classes = sorted(df["true"].unique())
    rec, f1 = [], []
    for c in classes:
        tp = ((df.true == c) & (df.pred == c)).sum()
        fn = ((df.true == c) & (df.pred != c)).sum()
        fp = ((df.true != c) & (df.pred == c)).sum()
        rec.append(tp / (tp + fn) if tp + fn else np.nan)
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan)
    return (df.true == df.pred).mean(), np.nanmean(rec), np.nanmean(f1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="C:/CIEDID_data/Manuf/models/fractions")
    ap.add_argument("--which", choices=["final", "selected"], default="final")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--allow_partial", action="store_true")
    ap.add_argument("--out", default=None, help="default: <root>/pooled_oof")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"Root not found: {root}  (use --root)")

    def is_run(d):
        return (d.is_dir() and (RUN_RE.match(d.name) or d.name == "zeroshot")
                and any(x.name.startswith("fold_") for x in d.iterdir()))

    runs = ([root] if is_run(root) else []) + sorted(
        (d for d in root.rglob("*") if d != root and is_run(d)), key=lambda d: str(d))
    if not runs:
        sys.exit(f"No run folders like <init>_frac<f>_seed<s> (containing fold_N) found at or under {root}\n"
                 f"Hint: --root should be the folder that CONTAINS the run folders.")
    names = [d.name for d in runs]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        for n in dup:
            print(f"ERROR duplicate run folder name '{n}' at:")
            for d in runs:
                if d.name == n:
                    print(f"   {d}")
        sys.exit("Remove or rename the duplicates, or point --root at one location.")
    base = root.parent if is_run(root) else root
    out_dir = Path(args.out) if args.out else base / "pooled_oof"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Found runs:")
    for d in runs:
        print(f"   {d}")

    pooled, rows = {}, []
    for d in runs:
        zs = d.name == "zeroshot"
        fname = ("val_predictions_zeroshot.csv" if zs else
                 "val_predictions_phase2_finalepoch.csv" if args.which == "final" else "val_predictions_phase2.csv")
        parts, missing = [], []
        for f in args.folds:
            p = d / f"fold_{f}" / fname
            if p.exists():
                x = pd.read_csv(p)
                x["fold"] = f
                parts.append(x)
            else:
                missing.append(f)
        if missing and not args.allow_partial:
            print(f"SKIP  {d.name}: missing {fname} in fold(s) {missing}")
            continue
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        problems = []
        if df["ID"].duplicated().any():
            problems.append(f"{int(df['ID'].duplicated().sum())} duplicated ID(s) across folds")
        pcols = [c for c in df.columns if c.startswith("p_") and c not in KEEP_NON_CLASS_P]
        if not set(df["true"]).issubset({c[2:] for c in pcols}):
            problems.append("true labels not covered by p_<class> columns")
        if problems:
            print(f"SKIP  {d.name}: " + "; ".join(problems))
            continue
        df = df[["ID", "fold", "true", "pred"] + pcols].rename(columns={"p_Vitatron(absent)": "p_Vitatron"})
        df["ID"] = df["ID"].astype(str)
        if zs:
            init, frac, seed, name = "zeroshot", 1.0, -1, "oof_zeroshot.csv"
        else:
            m = RUN_RE.match(d.name)
            init, frac, seed = m["init"], float(m["frac"]), int(m["seed"])
            name = f"oof_{ARM_NAME.get(init, init)}_" + ("" if frac == 1.0 else f"frac{frac:g}_") + f"s{seed}.csv"
        df.to_csv(out_dir / name, index=False)
        pooled[(init, frac, seed)] = df
        acc, ba, mf1 = metrics(df)
        rows.append(dict(run=d.name, n=len(df), folds=df["fold"].nunique(), errors=int((df.true != df.pred).sum()),
                         acc=acc, bal_acc=ba, macro_f1=mf1, file=name))

    if not rows:
        sys.exit("Nothing pooled.")
    print("\n" + pd.DataFrame(rows).round(4).to_string(index=False))

    groups = {}
    for (init, frac, seed), df in pooled.items():
        if init == "zeroshot":
            continue
        groups.setdefault((frac, seed), {})[init] = df.set_index("ID")
    for (frac, seed), arms in sorted(groups.items()):
        if len(arms) < 2:
            continue
        nm = list(arms)
        ref = arms[nm[0]]
        ok = True
        for n in nm[1:]:
            o = arms[n]
            if set(o.index) != set(ref.index) or not (o.loc[ref.index, "true"] == ref["true"]).all() \
                    or not (o.loc[ref.index, "fold"] == ref["fold"]).all():
                ok = False
                print(f"WARNING frac={frac:g} seed={seed}: {n} differs from {nm[0]} in cases/labels/folds")
        if ok:
            print(f"paired OK: frac={frac:g} seed={seed}: {', '.join(nm)} share identical cases, labels and folds")
    print(f"\nWritten to: {out_dir}")


if __name__ == "__main__":
    main()