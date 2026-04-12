# Finetune โมเดล UNet จาก ResNet50 เพื่อเพิ่มคลาส "Abandoned Lead" โดยใช้ FastAI
# Yes/No Report for Abandoned Lead และ Dice เฉพาะ Class 1 (Generator) เพื่อวัดประสิทธิภาพของโมเดลในแต่ละคลาส

import sys
import pathlib
import platform
import argparse
import os
import numpy as np
import pandas as pd
import torch
import warnings
from fastai.vision.all import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# ==============================
# 1. GLOBAL FUNCTIONS (Must be outside for Pickle/export)
# ==============================
CLASS_NAMES = ["background", "generator", "lead", "abandoned_lead"]

def get_x(r): return r["image"]
def get_y(r): return r["mask"]

def dice_generator(inp, targ, eps=1e-6):
    """ รายงาน Dice เฉพาะ Class 1 (Generator) """
    pred = inp.argmax(dim=1)
    p = (pred == 1).float()
    t = (targ == 1).float()
    inter = (p * t).sum()
    union = p.sum() + t.sum()
    return (2. * inter + eps) / (union + eps)

def abdn_lead_sensitivity(inp, targ):
    """ รายงานว่าตรวจเจอ Abandoned Lead หรือไม่ (Yes/No) """
    threshold_pixels=20
    pred = inp.argmax(dim=1)
    # ทายว่ามี (Yes) ถ้าพิกเซล Class 3 > threshold
    pred_yes = (pred == 3).sum(dim=(1,2)) > threshold_pixels
    # ของจริงคือมี ถ้ามีพิกเซล Class 3 ใน Mask
    targ_yes = (targ == 3).sum(dim=(1,2)) > 0
    
    actual_pos = targ_yes.sum()
    if actual_pos == 0: return torch.tensor(0.0)
    
    tp = (pred_yes & targ_yes).sum()
    return tp.float() / actual_pos.float()

# ==============================
# DATAFRAME BUILDER
# ==============================
def build_dataframe(args):
    rows = []
    exts = {".jpg", ".png", ".jpeg", ".bmp"}
    for img in args.new_imgs.iterdir():
        if img.suffix.lower() not in exts: continue
        mask = args.new_masks / f"{img.stem}_mask.png"
        if not mask.exists(): mask = args.new_masks / f"{img.stem}.png"
        if not mask.exists(): continue
        
        arr = np.array(PILImage.create(mask))
        rows.append({
            "image": str(img.resolve()),
            "mask": str(mask.resolve()),
            "has_abandoned": 3 in np.unique(arr)
        })
    
    df = pd.DataFrame(rows)
    train_idx, valid_idx = train_test_split(
        df.index, test_size=args.valid_split, 
        stratify=df["has_abandoned"], random_state=42
    )
    df["is_valid"] = False
    df.loc[valid_idx, "is_valid"] = True
    
    # Oversampling สำหรับ Train set
    train_df = df[~df["is_valid"]]
    with_abl = train_df[train_df["has_abandoned"]]
    if len(with_abl) > 0:
        extra = pd.concat([with_abl] * (args.oversample_new - 1))
        df = pd.concat([df, extra]).reset_index(drop=True)
        print(f"🔥 Oversampled Abandoned Lead x{args.oversample_new}")
    return df

# ==============================
# LOAD PRETRAINED
# ==============================
def load_pretrained_weights(learner, path):
    device = next(learner.model.parameters()).device
    old = load_learner(path, cpu=True)
    old_state = old.model.state_dict()
    new_state = learner.model.state_dict()
    for k, v in old_state.items():
        if k in new_state and new_state[k].shape == v.shape:
            new_state[k] = v.to(device)
    learner.model.load_state_dict(new_state)
    return learner

# ==============================
# MAIN TRAINING FUNCTION
# ==============================
def finetune(args):
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    df = build_dataframe(args)
    
    # DataBlock - ใช้กลยุทธ์ Zhukov (Resize 1024)
    dblock = DataBlock(
        blocks=(ImageBlock, MaskBlock(codes=CLASS_NAMES)),
        get_x=get_x, get_y=get_y,
        splitter=ColSplitter(col='is_valid'),
        item_tfms=Resize(1024), 
        batch_tfms=[*aug_transforms(size=args.patch_size, max_warp=0)] # ปิด warp เพื่อรักษาเส้นสายไฟ
    )

    dls = dblock.dataloaders(df, bs=args.batch_size, num_workers=0)
    
    # Weights สำหรับ Loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = torch.tensor([1.0, 10.0, 10.0, 50.0]).to(device) # เน้นหนักที่ Abandoned Lead
    loss_func = FocalLossFlat(axis=1, weight=weights)

    learner = unet_learner(
        dls, resnet50, n_out=4, 
        loss_func=loss_func, 
        metrics=[dice_generator, abdn_lead_sensitivity] # รายงานผลตามโจทย์
    ).to_fp16()

    learner = load_pretrained_weights(learner, args.model_path)

    print("--- Phase 1: Training Head ---")
    learner.freeze()
    learner.fit_one_cycle(args.epochs_head, 3e-4)

    print("--- Phase 2: Full Fine-Tuning ---")
    learner.unfreeze()
    learner.fit_one_cycle(args.epochs_full, lr_max=slice(1e-6, 1e-4))

    # Export Model
    model_export_path = pathlib.Path(args.output_dir) / "seg_abdnL_final.pkl"
    learner.export(model_export_path)
    print(f"✅ Model saved to: {model_export_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="C:/CIEDID_data/pkl/segmentation.pkl")
    parser.add_argument("--new_imgs", default="C:/CIEDID_data/AbdnL/data")
    parser.add_argument("--new_masks", default="C:/CIEDID_data/AbdnL/mask")
    parser.add_argument("--output_dir", default="C:/CIEDID_data/AbdnL/models")
    parser.add_argument("--epochs_head", type=int, default=5)
    parser.add_argument("--epochs_full", type=int, default=20) # เพิ่มรอบเพื่อให้เรียนรู้คลาสใหม่
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=384) # ใช้ Patch ขนาดใหญ่เพื่อให้เห็นสายยาวๆ -- ดีกว่าเพื่ม oversampling
    parser.add_argument("--valid_split", type=float, default=0.2)
    parser.add_argument("--oversample_new", type=int, default=5) # เพิ่มการเห็น Abandoned Lead ซ้ำๆ

    args = parser.parse_args()
    args.new_imgs = pathlib.Path(args.new_imgs)
    args.new_masks = pathlib.Path(args.new_masks)
    args.model_path = pathlib.Path(args.model_path)
    
    finetune(args)