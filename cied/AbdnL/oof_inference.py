"""
oof_inference.py
=================
Run TRUE out-of-fold inference for the abandoned-lead segmentation model.

For each of the 5 CV folds, this loads that fold's best_abdn.pth checkpoint
and runs inference ONLY on the images that were held out as that fold's
validation set (i.e. images the model never saw during training).

Combining all 5 folds' results gives a leakage-free per-image results file
covering the entire training-set pool (n≈197), which can then be used for
threshold grid search without the memorization concerns raised earlier.

Prerequisite: run reconstruct_folds.py first to generate oof_fold_map.csv.

Usage:
    python oof_inference.py \
        --oof_map "C:/CIEDID_data/AbdnL/fold_assignments/oof_fold_map.csv" \
        --models_dir "C:/CIEDID_data/AbdnL/models" \
        --output_dir "C:/CIEDID_data/AbdnL/oof_results"

Assumes each fold's checkpoint lives at:
    {models_dir}/fold_{N}/best_abdn.pth
Adjust --fold_subdir_pattern if your directory layout differs.
"""

import argparse
import pathlib
import platform
import sys
import numpy as np
import pandas as pd
import torch
from fastai.vision.all import *
from tqdm import tqdm

# ── PyTorch 2.6+ fix ────────────────────────────────────────────────
_orig_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault('weights_only', False)
    return _orig_load(*a, **kw)
torch.load = _patched_load

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

CLASS_NAMES = ["background", "generator", "lead", "abandoned_lead"]
ABDN_CLASS  = 3
N_CLASSES   = len(CLASS_NAMES)
SMOOTH      = 1e-6

# Same normalization stats as test_abdn.py / eval_single.py — keep identical
# so out-of-fold predictions are directly comparable to prior single-model runs.
NORM_MEAN = [0.5150052309036255] * 3
NORM_STD  = [0.23788487911224365] * 3

PROB_THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.65, 0.70,
                   0.75, 0.80, 0.85, 0.90, 0.95]


def get_x(r): return r["image"]
def get_y(r): return r["mask"]


def build_dls_for_fold(fold_df, img_size, device):
    """fold_df: images to run inference on (all treated as 'valid')."""
    BS_DUMMY = 4
    dummy = pd.concat([fold_df.iloc[[0]]] * BS_DUMMY, ignore_index=True)
    dummy["is_valid"] = False
    eval_df = pd.concat([dummy, fold_df.assign(is_valid=True)], ignore_index=True)

    dblock = DataBlock(
        blocks     = (ImageBlock, MaskBlock(codes=CLASS_NAMES)),
        get_x      = get_x,
        get_y      = get_y,
        splitter   = ColSplitter(col="is_valid"),
        item_tfms  = Resize(img_size, method='pad', pad_mode='zeros'),
        batch_tfms = [Normalize.from_stats(NORM_MEAN, NORM_STD)],
    )
    return dblock.dataloaders(eval_df, bs=BS_DUMMY, num_workers=0, pin_memory=True)


def load_model_weights(weights_path, dls, device):
    learner = unet_learner(dls, resnet50, n_out=N_CLASSES).to_fp16()
    state = torch.load(weights_path, map_location=device)
    learner.model.load_state_dict(state)
    learner.model.to(device)
    learner.model.eval()
    return learner.model


def run_inference(model, dls, fold_df, device, fold_idx):
    records = []
    paths = fold_df["image"].tolist()
    path_idx = 0

    with torch.no_grad():
        pbar = tqdm(dls.valid, total=len(dls.valid),
                    desc=f"Fold {fold_idx} OOF inference", unit="batch")
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            with torch.amp.autocast('cuda'):
                probs = model(xb).softmax(dim=1)

            preds = probs.argmax(dim=1)
            abdn_prob = probs[:, ABDN_CLASS, :, :]

            batch_paths = paths[path_idx:path_idx + len(xb)]
            path_idx += len(xb)
            pbar.set_postfix({"images_done": path_idx})

            for i in range(len(xb)):
                row = {
                    "image":         batch_paths[i] if i < len(batch_paths) else "",
                    "oof_fold":      fold_idx,
                    "has_abandoned": (yb[i] == ABDN_CLASS).sum().item() > 0,
                    "targ_px":       (yb[i] == ABDN_CLASS).sum().item(),
                    "pred_px":       (preds[i] == ABDN_CLASS).sum().item(),
                }
                for pt in PROB_THRESHOLDS:
                    row[f"prob_{int(pt*100):02d}"] = (abdn_prob[i] > pt).sum().item()
                records.append(row)

    return pd.DataFrame(records)


def main():
    ap = argparse.ArgumentParser()
    # ↓↓↓ แก้ path ตรงนี้ ↓↓↓
    ap.add_argument("--oof_map", default="C:/CIEDID_data/AbdnL/fold_assignments/oof_fold_map.csv")
    ap.add_argument("--models_dir", default="C:/CIEDID_data/AbdnL/models/best")
    ap.add_argument("--output_dir", default="C:/CIEDID_data/AbdnL/oof_results")
    ap.add_argument("--fold_subdir_pattern", default="fold_{}/best_abdn.pth",
                     help="Relative path (under models_dir) to each fold's checkpoint, "
                          "with {} as the fold index placeholder.")
    # ↑↑↑ แก้ path ตรงนี้ ↑↑↑
    ap.add_argument("--img_size", type=int, default=640)
    ap.add_argument("--n_splits", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    oof_map = pd.read_csv(args.oof_map)
    print(f"Loaded oof_fold_map.csv: {len(oof_map)} images total")
    print(oof_map["oof_fold"].value_counts().sort_index())

    all_results = []

    for fold_idx in range(args.n_splits):
        fold_df = oof_map[oof_map["oof_fold"] == fold_idx].copy().reset_index(drop=True)
        if len(fold_df) == 0:
            print(f"⚠️  Fold {fold_idx}: no images found — skipping")
            continue

        weight_path = pathlib.Path(args.models_dir) / args.fold_subdir_pattern.format(fold_idx)
        if not weight_path.exists():
            print(f"⚠️  Fold {fold_idx}: checkpoint not found at {weight_path} — skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Fold {fold_idx}: {len(fold_df)} out-of-fold images, "
              f"checkpoint = {weight_path}")
        print(f"{'='*60}")

        dls = build_dls_for_fold(fold_df, args.img_size, device)
        model = load_model_weights(weight_path, dls, device)
        fold_results = run_inference(model, dls, fold_df, device, fold_idx)
        all_results.append(fold_results)

        del model
        torch.cuda.empty_cache()

    if not all_results:
        raise RuntimeError("No fold results were produced — check paths and try again.")

    combined = pd.concat(all_results, ignore_index=True)
    out_path = output_dir / "oof_per_image_results.csv"
    combined.to_csv(out_path, index=False)

    print(f"\n✅ Combined out-of-fold results: {len(combined)} images "
          f"({combined['has_abandoned'].sum()} with abandoned lead, "
          f"{(~combined['has_abandoned']).sum()} without)")
    print(f"💾 Saved → {out_path}")
    print("\nNext step: run oof_grid_search.py on this file to sweep "
          "prob/pixel thresholds without any train-test leakage.")


if __name__ == "__main__":
    main()