"""
Finetune Segmentation Model — New Data Only (5 Classes)
========================================================
Class mapping:
    0 = Background
    1 = Pacemaker / Defibrillator generator
    2 = Heart Monitor / ICM
    3 = Active Lead 
    4 = Abandoned Lead (เป้าหมายหลัก)
"""

import sys
import pathlib
import platform
import argparse
import os
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# แก้ไขปัญหา Font ในบางระบบ
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts=false"

# ==========================================================
# COMPATIBILITY PATCHES
# ==========================================================
if not hasattr(np, 'int'):
    np.int = int

import fastai.callback.fp16
if not hasattr(fastai.callback.fp16, 'AMPMode'):
    class AMPMode:
        def __init__(self, *args, **kwargs): pass
    fastai.callback.fp16.AMPMode = AMPMode
    sys.modules['fastai.callback.fp16'].AMPMode = AMPMode

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

import torch
import pandas as pd
from fastai.vision.all import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ==========================================================
# CONFIGURATION (Default Paths)
# ==========================================================
BASE_DIR        = pathlib.Path("cied")
MODEL_PATH      = BASE_DIR / "segmentation.pkl"
NEW_IMGS        = BASE_DIR / "AbdnL/data"
NEW_MASKS       = BASE_DIR / "AbdnL/mask"
OUTPUT_DIR      = BASE_DIR / "AbdnL/models"
EXP_NAME        = "seg_finetune_5cls_new_only"

N_OUT_NEW       = 5
CLASS_NAMES     = ["background", "generator", "monitor", "lead", "abandoned_lead"]
IMG_SIZE        = 1024
PATCH_SIZE      = 256
SEED            = 42

CLASS_COLORS = {
    0: (30,  30,  30),    # Background
    1: (255, 80,  80),    # Generator color (Red)
    2: (0, 200, 200),     # Monitor color (Cyan)
    3: (80,  180, 255),   # Lead 
    4: (80,  255, 130),   # Abandoned Lead (Green)
}

# ==========================================================
# STEP 1 — Build DataFrame (New Data Only)
# ==========================================================
def build_dataframe(args):
    rows = []
    if args.new_imgs.exists() and args.new_masks.exists():
        print(f"Loading data from: {args.new_imgs}")
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        img_files = sorted([f for f in args.new_imgs.iterdir() if f.suffix.lower() in exts])

        found_new, missing = 0, []
        for img_path in img_files:
            # ตรวจสอบว่ามี mask หรือไม่ (ใช้ชื่อไฟล์_mask.png)
            mask_path = args.new_masks / f"{img_path.stem}_mask.png"
            if not mask_path.exists():
                # ลองหาแบบไม่มี _mask เผื่อไว้
                mask_path = args.new_masks / f"{img_path.stem}.png"

            if mask_path.exists():
                mask_arr = np.array(PILImage.create(mask_path))
                has_abandoned = 4 in np.unique(mask_arr)
                rows.append({
                    "image":         str(img_path),
                    "mask":          str(mask_path),
                    "has_abandoned": has_abandoned,
                })
                found_new += 1
            else:
                missing.append(img_path.name)

        print(f"  Found: {found_new} image-mask pairs")
        if missing: print(f"  [WARN] {len(missing)} images skipped (no mask found)")
    else:
        print(f"[ERROR] Path missing: {args.new_imgs} or {args.new_masks}")
        sys.exit(1)

    df = pd.DataFrame(rows)
    
    # Train/Valid split
    np.random.seed(SEED)
    df["is_valid"] = False
    valid_idx = df.sample(frac=args.valid_split, random_state=SEED).index
    df.loc[valid_idx, "is_valid"] = True

    # Oversampling Abandoned Lead (Class 4)
    train_df = df[~df["is_valid"]]
    with_abl = train_df[train_df["has_abandoned"]]
    without_abl = train_df[~train_df["has_abandoned"]]

    if len(with_abl) > 0 and len(with_abl) < len(without_abl):
        ratio = min(len(without_abl) // len(with_abl), args.oversample_new)
        if ratio > 1:
            extra = pd.concat([with_abl] * (ratio - 1))
            df = pd.concat([df, extra]).reset_index(drop=True)
            print(f"  Oversampled Class 4 (Abandoned) x{ratio}")

    return df

# ==========================================================
# STEP 2 — Validate classes in masks
# ==========================================================
def validate_masks(df):
    print("\nValidating Classes in Masks...")
    unique_vals = set()
    for m in df["mask"].unique():
        mask = np.array(PILImage.create(m))
        unique_vals.update(np.unique(mask).tolist())
    
    for i in range(N_OUT_NEW):
        status = "✓ Found" if i in unique_vals else "✗ Missing"
        print(f"  Class {i} ({CLASS_NAMES[i]}): {status}")

# ==========================================================
# STEP 3 — Weight Transfer & Helper Functions
# ==========================================================
def load_pretrained_weights(learner_new, old_model_path):
    print(f"\nLoading weights from {old_model_path.name}...")
    old_learn = load_learner(old_model_path, cpu=True)
    old_state = old_learn.model.state_dict()
    new_state = learner_new.model.state_dict()

    for k, v in old_state.items():
        if k in new_state and new_state[k].shape == v.shape:
            new_state[k] = v
    
    learner_new.model.load_state_dict(new_state)
    del old_learn
    return learner_new

def to_color(mask_arr):
    if len(mask_arr.shape) == 3: mask_arr = mask_arr[:, :, 0]
    h, w = mask_arr.shape
    c = np.zeros((h, w, 3), dtype=np.uint8)
    for v in np.unique(mask_arr):
        c[mask_arr == v] = CLASS_COLORS.get(int(v), (200, 200, 200))
    return c

def dice_abandoned(inp, targ, eps=1e-6):
    """Dice score เฉพาะ class 4 (abandoned_lead)"""
    pred   = inp.argmax(dim=1)
    pred_4 = (pred == 4).float()
    targ_4 = (targ == 4).float()
    inter  = (pred_4 * targ_4).sum()
    union  = pred_4.sum() + targ_4.sum()
    return (2. * inter + eps) / (union + eps)

def preview_predictions(learner, df, n=4):
    valid_df = df[df["is_valid"]].drop_duplicates(subset=["image"])
    sample_rows = valid_df.head(n).to_dict('records')
    
    fig, axes = plt.subplots(len(sample_rows), 3, figsize=(12, 4*len(sample_rows)))
    if len(sample_rows) == 1: axes = [axes]

    for i, row in enumerate(sample_rows):
        res = learner.predict(row["image"])
        pred, gt, orig = np.array(res[1]), np.array(PILImage.create(row["mask"])), np.array(PILImage.create(row["image"]).convert("RGB"))

        axes[i][0].imshow(orig); axes[i][0].set_title("Original"); axes[i][0].axis("off")
        axes[i][1].imshow(to_color(gt)); axes[i][1].set_title("Ground Truth"); axes[i][1].axis("off")
        axes[i][2].imshow(to_color(pred)); axes[i][2].set_title("Prediction"); axes[i][2].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "preview.png")
    plt.show()

# ==========================================================
# MAIN EXECUTION
# ==========================================================
def finetune(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_dataframe(args)
    validate_masks(df)

    dls = ImageDataLoaders.from_df(
        df, fn_col="image", label_col="mask", valid_col="is_valid",
        item_tfms=Resize(args.img_size),
        batch_tfms=[*aug_transforms(size=args.patch_size, do_flip=True, max_rotate=15.0)],
        y_block=MaskBlock(codes=CLASS_NAMES), bs=args.batch_size, num_workers=0
    )

    learner = unet_learner(dls, resnet50, n_out=N_OUT_NEW, metrics=[DiceMulti(), dice_abandoned]).to_fp16()
    learner = load_pretrained_weights(learner, args.model_path)

    print("\nPhase 1: Training Head...")
    learner.freeze()
    learner.fit_one_cycle(args.epochs_head, lr_max=1e-3)

    print("\nPhase 2: Full Finetuning...")
    learner.unfreeze()
    learner.fit_one_cycle(args.epochs_full, lr_max=slice(1e-6, 1e-4))

    learner.export(OUTPUT_DIR / f"{EXP_NAME}.pkl")
    print(f"\nModel saved to {OUTPUT_DIR / EXP_NAME}.pkl")
    preview_predictions(learner, df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  default=str(MODEL_PATH))
    parser.add_argument("--new_imgs",    default=str(NEW_IMGS))
    parser.add_argument("--new_masks",   default=str(NEW_MASKS))
    parser.add_argument("--epochs_head", type=int, default=5)
    parser.add_argument("--epochs_full", type=int, default=20)
    parser.add_argument("--batch_size",  type=int, default=2)
    parser.add_argument("--img_size",    type=int, default=IMG_SIZE)
    parser.add_argument("--patch_size",  type=int, default=PATCH_SIZE)
    parser.add_argument("--valid_split", type=float, default=0.2)
    parser.add_argument("--oversample_new", type=int, default=10)

    args = parser.parse_args()
    args.model_path = pathlib.Path(args.model_path)
    args.new_imgs   = pathlib.Path(args.new_imgs)
    args.new_masks  = pathlib.Path(args.new_masks)

    finetune(args)