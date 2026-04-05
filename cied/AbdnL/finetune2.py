# slow
import sys
import pathlib
import platform
import argparse
import os
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore")

os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts=false"

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

from fastai.vision.all import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==============================
# CONFIG
# ==============================
BASE_DIR = pathlib.Path("cied")
MODEL_PATH = BASE_DIR / "segmentation.pkl"
NEW_IMGS = BASE_DIR / "AbdnL/data"
NEW_MASKS = BASE_DIR / "AbdnL/mask"
OUTPUT_DIR = BASE_DIR / "AbdnL/models"

N_OUT_NEW = 5
CLASS_NAMES = ["background","generator","monitor","lead","abandoned_lead"]

SEED = 42

# ==============================
# DATAFRAME
# ==============================
def build_dataframe(args):
    rows = []
    exts = {".jpg",".png",".jpeg",".bmp"}

    for img in args.new_imgs.iterdir():
        if img.suffix.lower() not in exts: continue
        mask = args.new_masks / f"{img.stem}_mask.png"
        if not mask.exists():
            mask = args.new_masks / f"{img.stem}.png"
        if not mask.exists(): continue

        arr = np.array(PILImage.create(mask))
        rows.append({
            "image": str(img),
            "mask": str(mask),
            "has_abandoned": 4 in np.unique(arr)
        })

    df = pd.DataFrame(rows)

    # stratified split
    train_idx, valid_idx = train_test_split(
        df.index,
        test_size=args.valid_split,
        stratify=df["has_abandoned"],
        random_state=SEED
    )

    df["is_valid"] = False
    df.loc[valid_idx, "is_valid"] = True

    # =========================
    # 🔥 Oversampling (เฉพาะ train)
    # =========================
    train_df = df[~df["is_valid"]]

    with_abl = train_df[train_df["has_abandoned"]]
    without_abl = train_df[~train_df["has_abandoned"]]

    if len(with_abl) > 0 and len(with_abl) < len(without_abl):
        ratio = min(len(without_abl) // len(with_abl), args.oversample_new)

        if ratio > 1:
            extra = pd.concat([with_abl] * (ratio - 1))
            df = pd.concat([df, extra]).reset_index(drop=True)
            print(f"🔥 Oversampled Abandoned Lead x{ratio}")

    return df

def summarize_dataset(df):
    print("\n========== DATASET SUMMARY ==========")

    # ---------------------------
    # IMAGE COUNT
    # ---------------------------
    total = len(df)
    train = len(df[~df["is_valid"]])
    valid = len(df[df["is_valid"]])

    print(f"Total images : {total}")
    print(f"Train images : {train}")
    print(f"Valid images : {valid}")

    # ---------------------------
    # IMAGE-LEVEL CLASS PRESENCE
    # ---------------------------
    print("\n--- Image-level class presence ---")

    class_counts = {i: 0 for i in range(len(CLASS_NAMES))}

    for _, row in df.iterrows():
        mask = np.array(PILImage.create(row["mask"]))
        unique = np.unique(mask)
        for c in unique:
            class_counts[int(c)] += 1

    for i, name in enumerate(CLASS_NAMES):
        print(f"Class {i} ({name}): present in {class_counts[i]} images")

    # ---------------------------
    # PIXEL-LEVEL DISTRIBUTION
    # ---------------------------
    print("\n--- Pixel-level distribution ---")

    pixel_counts = {i: 0 for i in range(len(CLASS_NAMES))}

    for m in df["mask"].unique():
        mask = np.array(PILImage.create(m))
        vals, counts = np.unique(mask, return_counts=True)
        for v, c in zip(vals, counts):
            pixel_counts[int(v)] += int(c)

    total_pixels = sum(pixel_counts.values())

    for i, name in enumerate(CLASS_NAMES):
        pct = 100 * pixel_counts[i] / total_pixels if total_pixels > 0 else 0
        print(f"Class {i} ({name}): {pixel_counts[i]:,} pixels ({pct:.2f}%)")

    print("====================================\n")

# ==============================
# LOSS
# ==============================
class WeightedSegLoss(Module):
    """
    Weighted CrossEntropy + DiceLoss
    - CrossEntropy: ใช้ class weights โดยตรง
    - DiceLoss: คำนวณแยกต่างหากต่อ class แล้ว weighted average
    """
    def __init__(self, class_weights, alpha=0.5, axis=1):
        self.alpha   = alpha
        self.axis    = axis
        self.weights = class_weights          # tensor shape (n_classes,)
        self.ce      = CrossEntropyLossFlat(axis=axis, weight=class_weights)

    def forward(self, pred, targ):
        ce_loss = self.ce(pred, targ)

        # Dice per class (weighted)
        pred_soft = pred.softmax(dim=self.axis)   # (B, C, H, W)
        n_cls     = pred_soft.shape[1]
        dice_loss = 0.0
        w_sum     = 0.0
        eps       = 1e-6

        for c in range(n_cls):
            w        = self.weights[c].item()
            p        = pred_soft[:, c]
            t        = (targ == c).float()
            inter    = (p * t).sum()
            union    = p.sum() + t.sum()
            dice_c   = 1 - (2 * inter + eps) / (union + eps)
            dice_loss += w * dice_c
            w_sum    += w

        dice_loss = dice_loss / (w_sum + eps)
        return self.alpha * ce_loss + (1 - self.alpha) * dice_loss

# ==============================
# METRICS
# ==============================
def dice_abandoned(inp, targ, eps=1e-6):
    # logits → class
    pred = inp.argmax(dim=1)

    # mask class 4
    pred_4 = (pred == 4).float()
    targ_4 = (targ == 4).float()

    inter = (pred_4 * targ_4).sum()
    union = pred_4.sum() + targ_4.sum()

    return (2. * inter + eps) / (union + eps)

# ==============================
# LOAD WEIGHTS
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
    del old
    return learner

# ==============================
# PLOT LEARNING CURVE
# ==============================
def plot_learning_curves(learner, save_path):
    rec = learner.recorder

    train_losses = rec.losses
    val_vals = rec.values

    epochs = range(len(val_vals))

    val_loss = [v[0] for v in val_vals]
    dice_all = [v[1] for v in val_vals]
    dice_abl = [v[2] for v in val_vals]

    plt.figure(figsize=(12,5))

    # loss
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train")
    plt.plot([i*(len(train_losses)//len(val_loss)) for i in epochs],
             val_loss, label="Valid")
    plt.title("Loss")
    plt.legend()

    # dice
    plt.subplot(1,2,2)
    plt.plot(epochs, dice_all, label="Dice All")
    plt.plot(epochs, dice_abl, label="Dice Abandoned", linestyle="--")
    plt.title("Dice")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

# ==============================
# TRAIN
# ==============================
def finetune(args):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # GPU check
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] GPU not found — running on CPU (will be slow)")

    df = build_dataframe(args)
    summarize_dataset(df)

    dls = ImageDataLoaders.from_df(
        df,
        fn_col="image",
        label_col="mask",
        valid_col="is_valid",
        y_block=MaskBlock(codes=CLASS_NAMES),
        item_tfms=Resize(args.img_size),
        batch_tfms=[
            *aug_transforms(size=args.patch_size,
                            max_rotate=15,
                            max_zoom=1.2,
                            max_lighting=0.2,
                            max_warp=0.1),
            #Normalize.from_stats(*imagenet_stats)
        ],
        bs=args.batch_size,
        num_workers=0
    )

    # loss — class weights จาก pixel distribution
    # background 96.4% → 1.0, generator 1.69% → 20, monitor 0.01% → 50
    # lead 1.50% → 20, abandoned_lead 0.41% → 80
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights   = torch.tensor([1.0, 20.0, 50.0, 20.0, 80.0], dtype=torch.float32).to(device)
    loss_func = WeightedSegLoss(class_weights=weights, alpha=0.5)
    
    learner = unet_learner(
        dls,
        resnet50,
        n_out=N_OUT_NEW,
        loss_func=loss_func,
        metrics=[DiceMulti(), dice_abandoned]
    ).to_fp16()

    learner = load_pretrained_weights(learner, args.model_path)

    # Phase 1
    learner.freeze_to(-2)
    learner.fit_one_cycle(args.epochs_head, 1e-3)

    # Phase 2
    learner.unfreeze()
    learner.fit_one_cycle(args.epochs_full, lr_max=slice(1e-6,1e-4))
    # lr_max = slice(1e-5, 1e-3) for more

    # save
    model_path = OUTPUT_DIR / "seg_finetune.pkl"
    learner.export(model_path)

    # plot
    plot_learning_curves(learner, OUTPUT_DIR / "learning_curve.png")

    print("DONE")

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=str(MODEL_PATH))
    parser.add_argument("--new_imgs", default=str(NEW_IMGS))
    parser.add_argument("--new_masks", default=str(NEW_MASKS))
    parser.add_argument("--epochs_head", type=int, default=5)
    parser.add_argument("--epochs_full", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--valid_split", type=float, default=0.2)
    parser.add_argument("--oversample_new", type=int, default=5)

    args = parser.parse_args()
    args.model_path = pathlib.Path(args.model_path)
    args.new_imgs = pathlib.Path(args.new_imgs)
    args.new_masks = pathlib.Path(args.new_masks)

    finetune(args)