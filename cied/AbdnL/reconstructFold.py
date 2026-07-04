"""
Reconstruct the exact train/validation image lists for each of the 5 CV folds,
WITHOUT retraining any model.

This works because finetuneCV.py's build_dataframe() uses a fixed
random_state=42 for StratifiedKFold — the split is fully deterministic
given the same image folder contents and iteration order.

IMPORTANT: This will only reproduce the original folds correctly if
args.new_imgs / args.new_masks have NOT been modified (no files added,
removed, or renamed) since the original training run. If you're unsure,
compare the resulting total image count against what you expect
(should match the segmentation-cohort / training-set counts on record).

Usage:
    python reconstruct_folds.py \
        --new_imgs "C:/CIEDID_data/AbdnL/data" \
        --new_masks "C:/CIEDID_data/AbdnL/mask" \
        --n_splits 5 \
        --out_dir "C:/CIEDID_data/AbdnL/fold_assignments"
"""

import argparse
import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from fastai.vision.all import PILImage


def build_dataframe_only(new_imgs, new_masks):
    """Mirrors finetuneCV.py's image/mask discovery logic exactly."""
    rows = []
    exts = {".jpg", ".png", ".jpeg", ".bmp"}
    for img in new_imgs.iterdir():
        if img.suffix.lower() not in exts:
            continue
        mask = new_masks / f"{img.stem}_mask.png"
        if not mask.exists():
            mask = new_masks / f"{img.stem}.png"
        if not mask.exists():
            continue
        arr = np.array(PILImage.create(mask))
        rows.append({
            "image":         str(img.resolve()),
            "mask":          str(mask.resolve()),
            "has_abandoned": 3 in np.unique(arr),
        })
    df = pd.DataFrame(rows).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new_imgs", default="C:/CIEDID_data/AbdnL/data")
    ap.add_argument("--new_masks", default="C:/CIEDID_data/AbdnL/mask")
    ap.add_argument("--out_dir", default="C:/CIEDID_data/AbdnL/fold_assignments")
    ap.add_argument("--n_splits", type=int, default=5)
    args = ap.parse_args()

    new_imgs = pathlib.Path(args.new_imgs)
    new_masks = pathlib.Path(args.new_masks)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataframe_only(new_imgs, new_masks)
    print(f"Total images discovered: {len(df)}")
    print(f"  with abandoned lead   : {df['has_abandoned'].sum()}")
    print(f"  without abandoned lead: {(~df['has_abandoned']).sum()}")
    print()
    print("⚠️  Sanity check: compare the total above against your recorded "
          "training-set count. If it doesn't match, the image folder has "
          "likely changed since the original training run, and this "
          "reconstruction will NOT match the original folds.")
    print()

    # Reproduce the exact same StratifiedKFold call as finetuneCV.py
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    splits = list(skf.split(df, df["has_abandoned"]))

    for fold_idx, (train_idx, valid_idx) in enumerate(splits):
        fold_df = df.copy()
        fold_df["is_valid"] = False
        fold_df.loc[fold_df.index[valid_idx], "is_valid"] = True
        fold_df["fold"] = fold_idx

        out_path = out_dir / f"fold_{fold_idx}_assignment.csv"
        fold_df.to_csv(out_path, index=False)

        n_train = (~fold_df["is_valid"]).sum()
        n_valid = fold_df["is_valid"].sum()
        n_valid_abdn = (fold_df["is_valid"] & fold_df["has_abandoned"]).sum()
        print(f"Fold {fold_idx}: train={n_train}, valid={n_valid} "
              f"(valid abandoned-lead cases={n_valid_abdn}) → saved to {out_path}")

    # Also save a combined out-of-fold map: for each image, which fold
    # was it held out in (i.e. which fold's model should predict it
    # out-of-sample).
    oof_map = df.copy()
    oof_map["oof_fold"] = -1
    for fold_idx, (train_idx, valid_idx) in enumerate(splits):
        oof_map.loc[oof_map.index[valid_idx], "oof_fold"] = fold_idx

    oof_path = out_dir / "oof_fold_map.csv"
    oof_map.to_csv(oof_path, index=False)
    print(f"\n✅ Combined out-of-fold map saved → {oof_path}")
    print("   Use this to look up, for each image, which fold's checkpoint "
          "(best_abdn.pth in fold_N/) should be used for out-of-fold inference.")


if __name__ == "__main__":
    main()
    