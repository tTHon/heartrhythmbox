# Finetuning Script for AbdnL Dataset
# 4 classes: Background, Generator, Active Lead, Abandoned Lead
# image resize first to 512x512
# Key Improvements:
# 1. Class Weighting: เพิ่มน้ำหนักให้ Abandoned Lead เพื่อช่วยให้โมเดลเรียนรู้ได้ดีขึ้น
# 2. FP16 Training: เปิดใช้งานการฝึกแบบ Half Precision เพื่อประหยัดแรมและเพิ่ม Batch Size ได้
# 3. Gradient Accumulation: ใช้เทคนิคนี้เพื่อจำลอง Batch Size ใหญ่ขึ้นโดยไม่ต้องใช้แรมมาก
# 4. Learning Rate Scheduling: ปรับ Learning Rate ให้แคบลงในช่วง Full Finetuning เพื่อความเสถียร

import sys
import pathlib
import platform
import argparse
import os
import numpy as np
import torch
import warnings
from fastai.vision.all import *

warnings.filterwarnings("ignore")

# Fix for Windows/Linux path compatibility
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# ==============================
# CONFIG & PATHS
# ==============================
BASE_DIR = pathlib.Path("cied")
MODEL_PATH = BASE_DIR / "segmentation.pkl"
NEW_IMGS = BASE_DIR / "AbdnL/data"
NEW_MASKS = BASE_DIR / "AbdnL/mask"
OUTPUT_DIR = BASE_DIR / "AbdnL/models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# CUSTOM METRICS (เพิ่มตรงนี้)
# ==============================
def dice_multi(input, target):
    # คำนวณ Dice Score เฉลี่ยของทุก Class (0-3)
    return Dice(axis=1)(input, target)

def dice_abandoned(input, target):
    # คำนวณ Dice Score เฉพาะ Class 3 (Abandoned Lead)
    return Dice(axis=1, i=3)(input, target)

def get_msk(fn):
    # ดึงชื่อไฟล์ mask ให้ตรงกับรูปภาพ
    return NEW_MASKS / f"{fn.stem}_mask.png"

# ==============================
# TRAINING FUNCTION
# ==============================
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=str(MODEL_PATH))
    parser.add_argument("--batch_size", type=int, default=2) # สำหรับ 8GB ใช้ 2-4
    parser.add_argument("--size", type=int, default=512)    # ปรับเป็น 512 เพื่อความละเอียด
    parser.add_argument("--epochs_head", type=int, default=5)
    parser.add_argument("--epochs_full", type=int, default=15) # เพิ่มรอบเพื่อให้ Dice นิ่ง
    args = parser.parse_args()

    print(f"Using Device: {default_device()}")
    
    fnames = get_image_files(NEW_IMGS)
    codes = np.array(['background', 'generator', 'lead', 'abandoned_lead'])

    # DataBlock พร้อมการจัดการ Resize ที่ดีขึ้น
    dls = SegmentationDataLoaders.from_label_func(
        path=NEW_IMGS,
        fnames=fnames,
        label_func=get_msk,
        codes=codes,
        bs=args.batch_size,
        item_tfms=[Resize(args.size, method='pad', pad_mode='zeros')], # ป้องกันรูปเบี้ยว
        batch_tfms=[IntToFloatTensor(), Normalize.from_stats(*imagenet_stats)]
    )

    # 1. เพิ่ม Weight ให้ Class 3 (Abandoned Lead) 10 เท่า
    weights = torch.tensor([1.0, 1.0, 1.0, 10.0]).to(default_device())
    loss_func = CrossEntropyLossFlat(axis=1, weight=weights)

    # Load Model
    print("Loading pretrained model...")
    learn = load_learner(args.model_path, cpu=False)
    
    # สร้าง Learner ใหม่โดยใช้โครงสร้างเดิมแต่ปรับ Loss และ Dls
    learner = unet_learner(dls, resnet34, metrics=[dice_multi, Dice(axis=1, i=3)], 
                           loss_func=loss_func).to_fp16() # 2. เปิด FP16 ประหยัดแรม
    
    learner.model = learn.model # Copy weights

    # Phase 1: Train Head
    print("Phase 1: Training Head...")
    learner.freeze()
    learner.fit_one_cycle(args.epochs_head, 1e-3)

    # Phase 2: Full Finetuning
    print("Phase 2: Full Finetuning with Gradient Accumulation...")
    learner.unfreeze()
    
    # 3. ใช้ Gradient Accumulation (n_acc=4) เพื่อจำลอง Batch Size ใหญ่
    # 4. ปรับ Learning Rate ให้แคบลงเพื่อความเสถียร
    learner.fit_one_cycle(args.epochs_full, 
                          lr_max=slice(5e-6, 5e-5), 
                          cbs=[GradientAccumulation(n_acc=4)])

    # Save Results
    model_path = OUTPUT_DIR / "AbnL_finetune.pkl"
    learner.export(model_path)
    
    # Save Learning Curve
    learner.recorder.plot_loss()
    plt.savefig(OUTPUT_DIR / "learning_curve_v2.png")
    
    print(f"DONE! Model saved to {model_path}")

if __name__ == "__main__":
    train()