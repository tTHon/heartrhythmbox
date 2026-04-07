"""
Finetune Segmentation Model — New Data Only (4 Classes)
Updated with Focal + Dice Loss and Optimized Hyperparameters
ช้ามากกก
========================================================
Class mapping:
    0 = Background
    1 = Pacemaker / Defibrillator generator
    2 = Active Lead 
    3 = Abandoned Lead (เป้าหมายหลัก)
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
#from fastai.losses import CombinedLoss, DiceLoss, FocalLoss # Import เพิ่มเติม
import matplotlib.pyplot as plt

# ==========================================================
# CONFIGURATION
# ==========================================================
BASE_DIR        = pathlib.Path("cied")
MODEL_PATH      = BASE_DIR / "segmentation.pkl"
NEW_IMGS        = BASE_DIR / "AbdnL/data"
NEW_MASKS       = BASE_DIR / "AbdnL/mask"
OUTPUT_DIR      = BASE_DIR / "AbdnL/models"
EXP_NAME        = "AbdnL_finetune"

N_OUT_NEW       = 4
CLASS_NAMES     = ["background", "generator", "lead", "abandoned_lead"]
IMG_SIZE        = 1024
PATCH_SIZE      = 512  # ปรับเพิ่มจาก 256 เพื่อให้เห็นความต่อเนื่องของเส้น
SEED            = 42

CLASS_COLORS = {
    0: (30,  30,  30),    # Background
    1: (255, 80,  80),    # Generator (Red)
    2: (80,  180, 255),   # Lead (Blue)
    3: (80,  255, 130),   # Abandoned Lead (Green)
}

# (ฟังก์ชัน build_dataframe, validate_masks, load_pretrained_weights, to_color คงเดิมตาม finetune5.py)
# [ส่วนที่ข้ามไป...]

def build_dataframe(args):
    rows = []
    if args.new_imgs.exists() and args.new_masks.exists():
        print(f"Loading data from: {args.new_imgs}")
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        img_files = sorted([f for f in args.new_imgs.iterdir() if f.suffix.lower() in exts])
        found_new, missing = 0, []
        for img_path in img_files:
            mask_path = args.new_masks / f"{img_path.stem}_mask.png"
            if not mask_path.exists(): mask_path = args.new_masks / f"{img_path.stem}.png"
            if mask_path.exists():
                mask_arr = np.array(PILImage.create(mask_path))
                has_abandoned = 3 in np.unique(mask_arr)
                rows.append({"image": str(img_path), "mask": str(mask_path), "has_abandoned": has_abandoned})
                found_new += 1
            else: missing.append(img_path.name)
        print(f"  Found: {found_new} image-mask pairs")
    df = pd.DataFrame(rows)
    np.random.seed(SEED)
    df["is_valid"] = False
    valid_idx = df.sample(frac=args.valid_split, random_state=SEED).index
    df.loc[valid_idx, "is_valid"] = True
    return df

def dice_abandoned(inp, targ, eps=1e-6):
    pred   = inp.argmax(dim=1)
    pred_3 = (pred == 3).float()
    targ_3 = (targ == 3).float()
    inter  = (pred_3 * targ_3).sum()
    union  = pred_3.sum() + targ_3.sum()
    return (2. * inter + eps) / (union + eps)

def load_pretrained_weights(learner_new, old_model_path):
    old_learn = load_learner(old_model_path, cpu=True)
    old_state = old_learn.model.state_dict()
    new_state = learner_new.model.state_dict()
    for k, v in old_state.items():
        if k in new_state and new_state[k].shape == v.shape: new_state[k] = v
    learner_new.model.load_state_dict(new_state)
    return learner_new

class CombinedLossCustom(Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5):
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
    def forward(self, pred, targ):
        return self.dice_weight * self.dice(pred, targ) + self.focal_weight * self.focal(pred, targ)

    def activation(self, x): return torch.softmax(x, dim=1)
    def decodes(self, x):    return x.argmax(dim=1)

def finetune(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_dataframe(args)

    dls = ImageDataLoaders.from_df(
        df, fn_col="image", label_col="mask", valid_col="is_valid",
        item_tfms=Resize(args.img_size),
        batch_tfms=[*aug_transforms(size=args.patch_size, do_flip=True, max_rotate=15.0)],
        y_block=MaskBlock(codes=CLASS_NAMES), bs=args.batch_size, num_workers=0
    )

    # ปรับปรุง 1: ใช้ CombinedLoss (Dice + Focal) เพื่อแก้ปัญหา Class Imbalance
    loss_func = CombinedLossCustom(dice_weight=0.5, focal_weight=0.5)
   
    learner = unet_learner(dls, resnet50, n_out=N_OUT_NEW, 
                           loss_func=loss_func, 
                           metrics=[DiceMulti(), dice_abandoned]).to_fp16()
    
    learner = load_pretrained_weights(learner, args.model_path)

    print("\nPhase 1: Training Head...")
    learner.freeze()
    learner.fit_one_cycle(args.epochs_head, lr_max=1e-3)

    print("\nPhase 2: Full Finetuning...")
    learner.unfreeze()
    # ปรับปรุง 2: เพิ่ม Learning Rate ให้สูงขึ้นเพื่อให้โมเดลปรับตัวกับข้อมูลเส้นบางๆ ได้ดีขึ้น
    learner.fit_one_cycle(args.epochs_full, lr_max=slice(1e-5, 8e-4))

    learner.export(OUTPUT_DIR / f"{EXP_NAME}.pkl")
    print(f"\nModel saved to {OUTPUT_DIR / EXP_NAME}.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs_head", type=int, default=5)
    parser.add_argument("--epochs_full", type=int, default=20) # เพิ่ม Epoch เป็น 40 เพื่อการเรียนรู้ที่ลึกขึ้น
    parser.add_argument("--batch_size",  type=int, default=1)
    parser.add_argument("--img_size",    type=int, default=1024) # เพิ่มส่วนที่ขาดหายไป
    parser.add_argument("--patch_size",  type=int, default=512) # ใช้ Patch Size ที่ใหญ่ขึ้น
    parser.add_argument("--valid_split", type=float, default=0.2)
    
    args = parser.parse_args()
    args.new_imgs = NEW_IMGS
    args.new_masks = NEW_MASKS
    args.model_path = MODEL_PATH
    
    finetune(args)