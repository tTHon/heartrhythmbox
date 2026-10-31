"""
export_oof.py
=============
Rebuild each CV fold's validation split exactly as finetuneModelClass.py did,
load that fold's saved weights, and write ONE pooled out-of-fold prediction
file (oof_predictions.csv).  The held-out set (FinalTest == 1) is never touched.

Usage:
  python export_oof.py \
      --labels_csv C:/CIEDID_data/main_workbook.csv \
      --crop_dir   C:/CIEDID_data/Images_crop \
      --models_dir C:/CIEDID_data/ModelClass/models \
      --out_csv    C:/CIEDID_data/ModelClass/oof_predictions.csv \
      [--patient_col PatientID]      # optional, enables leakage check later

Requirements for exact reproducibility of the split (must match training):
  --n_splits 5, StratifiedKFold(shuffle=True, random_state=42),
  stratified on ModelClass_grouped (grouping map read from fold_0).

NOTE: finetuneModelClass.py saves the BEST-checkpoint weights
(modelclass_classifier_weights.pth), selected on the validation fold's
balanced accuracy, so OOF metrics from these weights are slightly optimistic.
To use final-epoch weights instead, add before `learner.load(...)` in the
training script:
    torch.save(learner.model.state_dict(), out / "final_epoch_weights.pth")
and pass --weights_name final_epoch_weights.pth here.
"""
import argparse
import pathlib
import platform
import warnings

import numpy as np
import pandas as pd
import torch
from fastai.vision.all import *
from sklearn.model_selection import StratifiedKFold

if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

_orig_load = torch.load
def _patched_load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_load(*a, **k)
torch.load = _patched_load
warnings.filterwarnings("ignore")


def get_x(r): return r["image_path"]
def get_y(r): return r["ModelClass_grouped"]


def build_pool(args, mapping):
    cols = ["ID", "ModelClass", "FinalTest"] + ([args.patient_col] if args.patient_col else [])
    df = pd.read_csv(args.labels_csv, usecols=cols)
    df["ID"] = df["ID"].astype(str)
    df["image_path"] = df["ID"].apply(lambda i: str((args.crop_dir / f"{i}_crop.png").resolve()))
    df = df[df["image_path"].apply(lambda p: pathlib.Path(p).exists())]
    df = df[df["ModelClass"].notna()]
    pool = df[df["FinalTest"] == 0].reset_index(drop=True)      # same order as training
    pool["ModelClass_grouped"] = pool["ModelClass"].map(mapping)
    assert pool["ModelClass_grouped"].notna().all(), "Class missing from grouping map"
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", default="C:/CIEDID_data/main_workbook.csv")
    ap.add_argument("--crop_dir", default="C:/CIEDID_data/Images_crop")
    ap.add_argument("--models_dir", default="C:/CIEDID_data/ModelClass/models")
    ap.add_argument("--out_csv", default="C:/CIEDID_data/ModelClass/oof_predictions.csv")
    ap.add_argument("--weights_name", default="modelclass_classifier_weights.pth")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--patient_col", default=None)
    args = ap.parse_args()
    args.crop_dir = pathlib.Path(args.crop_dir)
    models_dir = pathlib.Path(args.models_dir)

    f0 = models_dir / "fold_0"
    mapping = pd.read_csv(f0 / "class_grouping_map.csv", index_col=0)["grouped_into"].to_dict()
    class_names = (f0 / "class_names.txt").read_text().splitlines()
    pool = build_pool(args, mapping)
    print(f"Pool n={len(pool)}  classes={len(class_names)}")

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    splits = list(skf.split(pool, pool["ModelClass_grouped"]))

    all_rows = []
    for fold, (_, val_idx) in enumerate(splits):
        fdir = models_dir / f"fold_{fold}"
        wpath = fdir / args.weights_name
        if not wpath.exists():
            print(f"!! fold {fold}: {wpath} not found, skipped")
            continue
        fold_classes = (fdir / "class_names.txt").read_text().splitlines()
        assert fold_classes == class_names, f"fold {fold}: class order differs"

        df = pool.copy()
        df["is_valid"] = False
        df.loc[df.index[val_idx], "is_valid"] = True

        dblock = DataBlock(
            blocks=(ImageBlock, CategoryBlock(vocab=class_names)),
            get_x=get_x, get_y=get_y,
            splitter=ColSplitter(col="is_valid"),
            item_tfms=Resize(args.img_size, method="pad", pad_mode="zeros"),
            batch_tfms=[Normalize.from_stats(*imagenet_stats)],
        )
        dls = dblock.dataloaders(df, bs=32, num_workers=0)
        assert list(dls.vocab) == class_names

        learn = vision_learner(dls, resnet50, pretrained=False)
        learn.model.load_state_dict(torch.load(wpath, map_location="cpu"))
        probs, targs = learn.get_preds(dl=dls.valid)

        val_df = df[df["is_valid"]].reset_index(drop=True)
        assert len(val_df) == len(probs), "validation ordering mismatch"
        out = pd.DataFrame({
            "ID": val_df["ID"],
            "fold": fold,
            "true": [class_names[i] for i in targs.numpy()],
            "pred": [class_names[i] for i in probs.argmax(1).numpy()],
        })
        # sanity: label from CSV must equal label from dataloader
        assert (out["true"].values == val_df["ModelClass_grouped"].values).all(), \
            "dataloader order != dataframe order"
        if args.patient_col:
            out["patient"] = val_df[args.patient_col].values
        for j, c in enumerate(class_names):
            out[f"p_{c}"] = probs[:, j].numpy()
        all_rows.append(out)
        acc = (out["true"] == out["pred"]).mean()
        print(f"fold {fold}: n={len(out)}  acc={acc:.4f}")

    oof = pd.concat(all_rows, ignore_index=True)
    oof.to_csv(args.out_csv, index=False)
    print(f"\nSaved {len(oof)} OOF predictions -> {args.out_csv}")
    print(f"Unique IDs: {oof['ID'].nunique()} (should equal pool size {len(pool)} if all folds present)")


if __name__ == "__main__":
    main()