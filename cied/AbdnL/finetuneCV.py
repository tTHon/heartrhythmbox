# finetuning script for abandoned lead segmentation (5-Fold CV - One fold at a time)
# how to use: python finetune_5Fold.py --fold_idx 0

import pathlib
import platform
import argparse
import numpy as np
import pandas as pd
import torch
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from fastai.vision.all import *

# 1. นำเข้า StratifiedKFold แทน train_test_split
from sklearn.model_selection import StratifiedKFold

# PyTorch 2.6+ fix: monkey-patch torch.load to use weights_only=False for fastai compatibility
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

warnings.filterwarnings("ignore")

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath


def parse_lr_arg(value):
    value = value.strip()
    if value.startswith("slice(") and value.endswith(")"):
        inner = value[len("slice("):-1]
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        return slice(float(parts[0]), float(parts[1]))
    return float(value)

# ==============================
# 1. GLOBAL FUNCTIONS 
# ==============================
CLASS_NAMES  = ["background", "generator", "lead", "abandoned_lead"]
CLASS_COLORS = [
    (0.15, 0.15, 0.15),
    (0.20, 0.60, 1.00),
    (0.20, 0.85, 0.45),
    (1.00, 0.25, 0.25),
]

def get_x(r): return r["image"]
def get_y(r): return r["mask"]

def dice_generator(inp, targ, eps=1e-6):
    pred = inp.argmax(dim=1)
    p = (pred == 1).float()
    t = (targ == 1).float()
    inter = (p * t).sum()
    union = p.sum() + t.sum()
    return (2. * inter + eps) / (union + eps)

class AbdnLeadSensitivity(Metric):
    def reset(self):
        self.tp  = 0
        self.pos = 0

    def accumulate(self, learn):
        scale = (args.patch_size / args.img_size) ** 2
        threshold_px = int(args.abdn_min_512 * scale)
        inp  = learn.pred
        targ = learn.yb[0]
        pred = inp.argmax(dim=1)
        pred_yes = (pred == 3).sum(dim=(1, 2)) > threshold_px
        targ_yes = (targ == 3).sum(dim=(1, 2)) > 0 
        self.tp  += (pred_yes & targ_yes).sum().item()
        self.pos += targ_yes.sum().item()

    @property
    def value(self):
        return self.tp / self.pos if self.pos > 0 else float('nan')

    @property
    def name(self):
        return "abdn_sensitivity"
    
def dice_abdn(inp, targ, eps=1e-6):
    pred  = inp.argmax(dim=1)
    p     = (pred == 3).float()
    t     = (targ == 3).float()
    inter = (p * t).sum()
    union = p.sum() + t.sum()
    return (2. * inter + eps) / (union + eps)

def dice_lead(inp, targ, eps=1e-6):
    pred  = inp.argmax(dim=1)
    p     = (pred == 2).float()
    t     = (targ == 2).float()
    inter = (p * t).sum()
    union = p.sum() + t.sum()
    return (2. * inter + eps) / (union + eps)

class MultiScaleCrop(ItemTransform):
    split_idx = 0   # train only
    def __init__(self, img_size, scales=(0.6, 0.75, 0.875, 1.0)):
        self.img_size = img_size
        self.scales   = scales

    def encodes(self, xy: tuple):
        img, mask = xy
        scale     = float(np.random.choice(self.scales))
        crop_px   = int(self.img_size * scale)
        w, h = img.size if hasattr(img, 'size') else (img.shape[-1], img.shape[-2])
        max_x = max(w - crop_px, 0)
        max_y = max(h - crop_px, 0)
        x0    = np.random.randint(0, max_x + 1)
        y0    = np.random.randint(0, max_y + 1)
        img  = img.crop( (x0, y0, x0 + crop_px, y0 + crop_px))
        mask = mask.crop((x0, y0, x0 + crop_px, y0 + crop_px))
        return img, mask

@torch.no_grad()
def get_segmentation_stats(dls):
    print("  📊 Calculating custom dataset statistics (mean/std)...")
    device = dls.train.device
    sum_ = torch.zeros(3, device=device)
    sum_sq = torch.zeros(3, device=device)
    count = 0
    for xb, yb in dls.train:
        b, c, h, w = xb.shape
        num_pixels = b * h * w
        sum_ += torch.sum(xb, dim=[0, 2, 3])
        sum_sq += torch.sum(xb**2, dim=[0, 2, 3])
        count += num_pixels
    mean = sum_ / count
    std = torch.sqrt((sum_sq / count) - (mean**2))
    print(f"  📊 Calculated Mean: {mean.tolist()}, Std: {std.tolist()}")
    return mean.cpu(), std.cpu()

# ==============================
# 2. DATASET SUMMARY
# ==============================
def summarize_dataset(df):
    print("\n========== DATASET SUMMARY ==========")
    total = len(df)
    train = len(df[~df["is_valid"]])
    valid = len(df[df["is_valid"]])
    print(f"Total images : {total}")
    print(f"Train images : {train}")
    print(f"Valid images : {valid}")

    print("\n--- Image-level class presence ---")
    class_counts = {i: 0 for i in range(len(CLASS_NAMES))}
    for _, row in df.iterrows():
        mask = np.array(PILImage.create(row["mask"]))
        for c in np.unique(mask):
            class_counts[int(c)] += 1
    for i, name in enumerate(CLASS_NAMES):
        print(f"Class {i} ({name}): present in {class_counts[i]} images")
    print("====================================\n")

# ==============================
# 3. BATCH INSPECTION VISUALISER (ย่อไว้เพื่อประหยัดบรรทัด)
# ==============================
def _mask_to_rgb(mask_arr):
    h, w = mask_arr.shape
    rgb = np.zeros((h, w, 3), dtype=float)
    for cls_idx, color in enumerate(CLASS_COLORS):
        where = mask_arr == cls_idx
        for ch, v in enumerate(color):
            rgb[:, :, ch][where] = v
    return rgb

def show_batch_inspection(dls, n=4, save_path=None):
    # (ใช้ Logic เดิมของคุณทั้งหมด)
    pass # ข้ามส่วนนี้ไปเพื่อความกระชับ คุณสามารถก๊อปโค้ด show_batch_inspection เดิมมาแปะทับได้เลยครับถ้าต้องการใช้งาน

# ==============================
# 4. DATAFRAME BUILDER (🔥 ปรับแก้ให้ใช้ StratifiedKFold)
# ==============================
def build_dataframe(args):
    rows = []
    exts = {".jpg", ".png", ".jpeg", ".bmp"}
    for img in args.new_imgs.iterdir():
        if img.suffix.lower() not in exts:
            continue
        mask = args.new_masks / f"{img.stem}_mask.png"
        if not mask.exists():
            mask = args.new_masks / f"{img.stem}.png"
        if not mask.exists():
            continue
        arr = np.array(PILImage.create(mask))
        rows.append({
            "image":         str(img.resolve()),
            "mask":          str(mask.resolve()),
            "has_abandoned": 3 in np.unique(arr),
        })

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # NEW: Cross-Validation Logic
    # ------------------------------------------------------------------
    print(f"\n🔀 Running Cross-Validation: Fold {args.fold_idx+1} of {args.n_splits}")
    
    # ล็อก random_state ไว้ที่ 42 เสมอ เพื่อให้การแบ่ง split เหมือนเดิมทุกครั้งที่รัน
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    splits = list(skf.split(df.index, df["has_abandoned"]))
    
    # ดึง index ของ Fold ที่เลือกรันในรอบนี้
    train_idx, valid_idx = splits[args.fold_idx]

    df["is_valid"] = False
    df.loc[valid_idx, "is_valid"] = True

    # ------------------------------------------------------------------
    # Oversampling Logic (Bug 1 fixed)
    # ------------------------------------------------------------------
    train_df    = df[~df["is_valid"]].copy()
    valid_df    = df[df["is_valid"]].copy()

    with_abl = train_df[train_df["has_abandoned"]]
    if len(with_abl) > 0 and args.oversample_new > 1:
        extra    = pd.concat([with_abl] * (args.oversample_new - 1), ignore_index=True)
        train_df = pd.concat([train_df, extra], ignore_index=True)
        print(f"🔥 Oversampled abandoned lead ×{args.oversample_new} "
              f"({len(with_abl)} → {len(train_df[train_df['has_abandoned']])} train rows)")

    # Re-merge: valid set is guaranteed untouched
    df = pd.concat([train_df, valid_df], ignore_index=True)
    return df


# ==============================
# 5. LOAD PRETRAINED WEIGHTS
# ==============================
ENCODER_PREFIXES = ("layers.0.",)
def load_pretrained_weights(learner, path):
    device    = next(learner.model.parameters()).device
    old       = load_learner(path, cpu=True)
    old_state = old.model.state_dict()
    new_state = learner.model.state_dict()

    loaded = skipped_mismatch = skipped_decoder = 0
    for k, v in old_state.items():
        is_encoder = any(k.startswith(p) for p in ENCODER_PREFIXES)
        if not is_encoder:
            skipped_decoder += 1
            continue
        if k in new_state and new_state[k].shape == v.shape:
            new_state[k] = v.to(device)
            loaded += 1
        else:
            skipped_mismatch += 1

    learner.model.load_state_dict(new_state)
    print(f"✅ Encoder weights loaded: {loaded} layers transferred")
    return learner

# ==============================
# 6. MAIN TRAINING FUNCTION
# ==============================
class FocalDiceLoss(nn.Module):
    def __init__(self, weights: torch.Tensor, focal_w: float = 0.5, dice_w:  float = 0.5, gamma: float = 2.0):
        super().__init__()
        self.focal_w = focal_w
        self.dice_w  = dice_w
        self.focal   = FocalLossFlat(axis=1, weight=weights, gamma=gamma)

    def forward(self, inp: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        focal = self.focal(inp, targ)
        n_classes = inp.shape[1]
        probs  = inp.softmax(dim=1)
        targ_oh = torch.zeros_like(probs)
        targ_oh.scatter_(1, targ.unsqueeze(1).long(), 1.0)
        inter = (probs * targ_oh).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + targ_oh.sum(dim=(0, 2, 3))
        dice_per_class = (2.0 * inter + 1.0) / (union + 1.0)
        dice_loss = 1.0 - dice_per_class.mean()
        return self.focal_w * focal + self.dice_w * dice_loss

    def activation(self, out):  return out.softmax(dim=1)
    def decodes(self, out):     return out.argmax(dim=1)

def plot_learning_curves(csv_path, output_path):
    # (ใช้ Logic เดิมของคุณทั้งหมด)
    pass # ย่อไว้เพื่อความกระชับ

def finetune(args):
    # 🔥 เปลี่ยน Output Directory ให้เป็นของ Fold นั้นๆ
    out = pathlib.Path(args.output_dir) / f"fold_{args.fold_idx}"
    out.mkdir(parents=True, exist_ok=True)

    import csv
    from datetime import datetime
    csv_path = out / "training_history.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["# TRAINING CONFIG", datetime.now().strftime("%Y-%m-%d %H:%M")])
        writer.writerow(["# Fold", f"{args.fold_idx+1} of {args.n_splits}"])
        # เขียน config อื่นๆ ตามเดิม...
        writer.writerow([]) 

    df = build_dataframe(args)
    
    n_valid_abdn = df[df["is_valid"] & df["has_abandoned"]].shape[0]
    n_train_abdn = df[~df["is_valid"] & df["has_abandoned"]].shape[0]
    print(f"🔍 Checking Validation Set: พบ Abandoned Lead {n_valid_abdn} รูป")
    print(f"🔍 Checking Training Set: พบ Abandoned Lead {n_train_abdn} รูป")
        
    summarize_dataset(df)
   
    myMEAN = [0.5026684403419495, 0.5026684403419495, 0.5026684403419495]
    mySTD  = [0.2409660518169403, 0.2409660518169403, 0.2409660518169403]
    stats_mean, stats_std = myMEAN, mySTD

    item_tfms_list = []
    if args.multiscale_crop:
        item_tfms_list.append(MultiScaleCrop(img_size=args.img_size))
    item_tfms_list.append(Resize(args.patch_size, method='pad', pad_mode='zeros'))

    dblock = DataBlock(
        blocks=(ImageBlock, MaskBlock(codes=CLASS_NAMES)),
        get_x=get_x, get_y=get_y,
        splitter=ColSplitter(col='is_valid'),
        item_tfms=item_tfms_list,
        batch_tfms=[
            *aug_transforms (do_flip=True, flip_vert=False, max_rotate=15, min_zoom=0.9, max_zoom=1.1, max_lighting=0.1, max_warp=0.0, p_affine=0.75, p_lighting=0.5),      
            Normalize.from_stats(stats_mean, stats_std)
            ],  
    )
    dls = dblock.dataloaders(df, bs=args.batch_size, num_workers=0, pin_memory=True, persistent_workers=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = torch.tensor(args.class_weights, dtype=torch.float32).to(device)
    
    loss_func = FocalDiceLoss(weights=weights, focal_w=args.focal_w, dice_w=args.dice_w, gamma=args.focal_gamma)

    learner = unet_learner(
        dls, resnet50, n_out=4,
        loss_func=loss_func,
        metrics=[dice_generator, dice_lead, AbdnLeadSensitivity(), dice_abdn],
        cbs=[CSVLogger(fname=str(out / "training_history.csv"), append=True)]).to_fp16()

    learner = load_pretrained_weights(learner, args.model_path)

    # Phase 0
    learner.freeze_to(1)          
    learner.fit_one_cycle(args.epochs_decoder, args.lr_phase0, cbs=GradientAccumulation(n_acc=args.grad_accum))

    # Phase 1
    learner.freeze()              
    learner.fit_one_cycle(args.epochs_head, args.lr_phase1, cbs=GradientAccumulation(n_acc=args.grad_accum))

    # Phase 2
    learner.unfreeze()
    learner.path      = out
    learner.model_dir = ""
    learner.fit_one_cycle(
        args.epochs_full,
        lr_max=args.lr_phase2,
        cbs=[GradientAccumulation(n_acc=args.grad_accum),
         SaveModelCallback(monitor='dice_generator', fname='best_gen', with_opt=False),
         SaveModelCallback(monitor='dice_abdn', fname='best_abdn', with_opt=False)]
    )

    weights_path = out / f"seg_abdnL_weights_fold{args.fold_idx}.pth"
    torch.save(learner.model.state_dict(), weights_path)
    print(f"\n✅ Weights saved → {weights_path}")

# ==============================
# 7. ENTRY POINT
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument("--model_path", default="C:/CIEDID_data/pkl/segmentation.pkl")
    parser.add_argument("--new_imgs", default="C:/CIEDID_data/AbdnL/data")
    parser.add_argument("--new_masks", default="C:/CIEDID_data/AbdnL/mask")
    parser.add_argument("--output_dir", default="C:/CIEDID_data/AbdnL/models")
    
    # 🔥 พารามิเตอร์ใหม่สำหรับทำ Cross-Validation
    parser.add_argument("--n_splits", type=int, default=5, help="Total number of folds for Cross-Validation")
    parser.add_argument("--fold_idx", type=int, default=0, help="Index of the fold to run (0 to n_splits-1)")
    
    # training config (ละไว้ตามเดิมของคุณ)
    parser.add_argument("--oversample_new", type=int,   default=2) 
    parser.add_argument("--class_weights", nargs=4, type=float, default=[1.0, 10, 10, 45])
    parser.add_argument("--img_size",      type=int,   default=512) 
    parser.add_argument("--patch_size",     type=int,   default=320)  
    parser.add_argument("--batch_size",     type=int,   default=2) 
    parser.add_argument("--grad_accum",   type=int,   default=4) 
    parser.add_argument("--valid_split", type=float,   default=0.2) # <- ตัวนี้ไม่ได้ใช้แล้วใน CV แต่ปล่อยไว้เผื่อรันปกติ
    parser.add_argument("--abdn_min_512", type=int,   default=2000)  
    parser.add_argument("--epochs_decoder",  type=int,   default=5)   
    parser.add_argument("--epochs_head",    type=int,   default=5)    
    parser.add_argument("--epochs_full",    type=int,   default=10)  
    parser.add_argument("--lr_phase0",   type=float,   default=5e-4)  
    parser.add_argument("--lr_phase1",   type=float,   default=3e-4)  
    parser.add_argument("--lr_phase2",   type=parse_lr_arg, default="slice(1e-6, 1e-4)")  
    parser.add_argument("--focal_w",    type=float, default=0.5)
    parser.add_argument("--dice_w",     type=float, default=0.5)
    parser.add_argument("--focal_gamma",type=float, default=2.0)
    parser.add_argument("--calc_stats",action="store_true", default=False)
    parser.add_argument("--multiscale_crop", action="store_true", default=True)

    args = parser.parse_args()
    args.new_imgs    = pathlib.Path(args.new_imgs)
    args.new_masks   = pathlib.Path(args.new_masks)
    args.model_path  = pathlib.Path(args.model_path)

    # เช็คว่า fold_idx ใส่มาถูกต้องไหม
    if args.fold_idx >= args.n_splits or args.fold_idx < 0:
        raise ValueError(f"Fold index ต้องอยู่ระหว่าง 0 ถึง {args.n_splits - 1}")

    finetune(args)