# finetuning script for abandoned lead segmentation 
# using encoder-only transfer from the original segmentation model (segmentation.pkl).
#
# FIXES vs finetuneYN1.py:
#   [BUG 1] build_dataframe: oversampling now runs only on confirmed train rows,
#           then is re-merged with the untouched valid set.
#   [BUG 2] abdn_lead_sensitivity: returns float('nan') instead of 0.0 when no
#           positive ground-truth exists in the batch.
#
# STRATEGY CHANGE (encoder-only transfer):
#   The pretrained model (segmentation.pkl) was trained on 3 classes:
#   background / generator / MONITOR — not lead.
#   Transferring decoder weights trained on "monitor" into a model that must
#   learn "lead" actively hurts: the decoder features are wrong for the task
#   and the small dataset cannot overwrite them efficiently.
#   Solution: transfer ENCODER only (ResNet50 backbone, layers 0–4).
#   The decoder is randomly initialised and trained from scratch.
#   This adds Phase 0: freeze encoder → train decoder only before the
#   usual head-only and full-finetune phases.
#
# NEW: show_batch_inspection() — visualise raw image + mask overlay + pixel
#      distribution for every train/valid sample before training starts.
# model export: save state_dict (.pth) instead of learner.export() to avoid pickle 'code' object error
# NEW: custom dataset statistics calculation (mean/std) with --calc_stats flag, used for normalization instead of ImageNet stats.
       # default is False. Enable at argparse.
# NEW: added GradientAccumulation callback to simulate larger batch size (effective batch size = batch_size * n_acc)
    
# NEW:  remove lr_find, as it is not suitable while using GradientAccumulation
# method = pad 
# New: multiscale crop: randomly crop a window of size drawn from `scales` (fractions of img_size) before resizing to patch_size, applied only on training items. This allows the model to see the same image region at different "zoom levels" without changing VRAM usage or batch size. Validation always uses the full image via the fixed Resize(patch_size).

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
from sklearn.model_selection import train_test_split

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
    """Parse a learning-rate argument into float or slice for fit_one_cycle."""
    value = value.strip()
    if value.startswith("slice(") and value.endswith(")"):
        inner = value[len("slice("):-1]
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                "lr_phase2 slice must be in form slice(start, end)"
            )
        try:
            start = float(parts[0])
            end = float(parts[1])
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "lr_phase2 slice values must be floats"
            ) from exc
        return slice(start, end)
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "lr_phase2 must be a float or slice(start, end)"
        ) from exc

# ==============================
# 1. GLOBAL FUNCTIONS (Must be outside for Pickle/export)
# ==============================
CLASS_NAMES  = ["background", "generator", "lead", "abandoned_lead"]
CLASS_COLORS = [
    (0.15, 0.15, 0.15),   # background  — dark grey
    (0.20, 0.60, 1.00),   # generator   — blue
    (0.20, 0.85, 0.45),   # lead        — green
    (1.00, 0.25, 0.25),   # abandoned_lead — red  (the new class)
]

def get_x(r): return r["image"]
def get_y(r): return r["mask"]

def dice_generator(inp, targ, eps=1e-6):
    """Dice coefficient for Class 1 (generator) only."""
    pred = inp.argmax(dim=1)
    p = (pred == 1).float()
    t = (targ == 1).float()
    inter = (p * t).sum()
    union = p.sum() + t.sum()
    return (2. * inter + eps) / (union + eps)

# ------------------------------------------------------------------
# BUG FIX 2: return nan instead of 0.0 when no positives in batch.
# FastAI's AccumMetric skips nan values, so the epoch average
# is computed only over batches that actually contain abandoned leads.
# Returning 0.0 would drag the average down unfairly.
# ------------------------------------------------------------------
# Yes/No sensitivity: did the model detect abandoned lead when present?

class AbdnLeadSensitivity(Metric):
    def reset(self):
        self.tp  = 0
        self.pos = 0

    def accumulate(self, learn):
        # find the threshold pixels at the current patch size
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
    """Dice for class 3 (abandoned_lead).
    Returns nan when no abandoned_lead pixels in batch so AccumMetric
    ignores batch rather than unfairly penalising it.
    """
    pred  = inp.argmax(dim=1)
    p     = (pred == 3).float()
    t     = (targ == 3).float()
    if t.sum() == 0:
        return torch.tensor(float('nan'))
    inter = (p * t).sum()
    union = p.sum() + t.sum()
    return (2. * inter + eps) / (union + eps)

def dice_lead(inp, targ, eps=1e-6):
    """Dice for class 2 (active lead).
    Returns nan when no lead pixels in batch.
    """
    pred  = inp.argmax(dim=1)
    p     = (pred == 2).float()
    t     = (targ == 2).float()
    if t.sum() == 0:
        return torch.tensor(float('nan'))
    inter = (p * t).sum()
    union = p.sum() + t.sum()
    return (2. * inter + eps) / (union + eps)


# ──────────────────────────────────────────────────────────────────
# Multi-scale patch sampling
# ──────────────────────────────────────────────────────────────────
# Applied as an item_tfm BEFORE the fixed Resize.
# Strategy: randomly crop a window of size drawn from `scales`,
# then let the subsequent Resize(patch_size) bring it to a uniform size.
# Result: model sees the same image region at different "zoom levels"
# without changing VRAM usage or batch size.
#
# Scales are expressed as fractions of img_size, e.g. 0.6 means
# crop = 0.6 * img_size px → zoomed-in view of lead detail.
# 1.0 means no crop (full image) → global context.
#
# Only applied on training items (is_train=True in FastAI pipeline).
# Validation always uses the full image via the fixed Resize(patch_size).
# ──────────────────────────────────────────────────────────────────
class MultiScaleCrop(ItemTransform):
    """
    Randomly crop a square window sized `scale * img_size` from the image,
    then the downstream Resize(patch_size) resamples it to patch_size.
    Applied to (image, mask) pairs so spatial alignment is preserved.

    Args:
        img_size  : full image dimension (px) — same as args.img_size
        scales    : list of scale fractions to sample from uniformly
        split_idx : FastAI split index for train (0). Skipped on valid (1).
    """
    split_idx = 0   # train only

    def __init__(self, img_size, scales=(0.6, 0.75, 0.875, 1.0)):
        self.img_size = img_size
        self.scales   = scales

    def encodes(self, xy: tuple):
        img, mask = xy
        scale     = float(np.random.choice(self.scales))
        crop_px   = int(self.img_size * scale)

        # PIL size is (W, H); torch tensor is (C, H, W)
        w, h = img.size if hasattr(img, 'size') else (img.shape[-1], img.shape[-2])
        max_x = max(w - crop_px, 0)
        max_y = max(h - crop_px, 0)
        x0    = np.random.randint(0, max_x + 1)
        y0    = np.random.randint(0, max_y + 1)

        img  = img.crop( (x0, y0, x0 + crop_px, y0 + crop_px))
        mask = mask.crop((x0, y0, x0 + crop_px, y0 + crop_px))
        return img, mask

# ──────────────────────────────────────────────────────────────────
# NEW: Custom Normalization Logic
# ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def get_segmentation_stats(dls):
    """Calculates mean and std from a FastAI DataLoaders object."""
    print("  📊 Calculating custom dataset statistics (mean/std)...")
    # Get device from first batch
    device = dls.train.device
    sum_ = torch.zeros(3, device=device)
    sum_sq = torch.zeros(3, device=device)
    count = 0

    # We use the train loader to calculate stats
    for xb, yb in dls.train:
        # xb is [B, 3, H, W]
        b, c, h, w = xb.shape
        num_pixels = b * h * w
        sum_ += torch.sum(xb, dim=[0, 2, 3])
        sum_sq += torch.sum(xb**2, dim=[0, 2, 3])
        count += num_pixels

    mean = sum_ / count
    std = torch.sqrt((sum_sq / count) - (mean**2))
    print(f"  📊 Calculated Mean: {mean.tolist()}, Std: {std.tolist()}")
    return mean.cpu(), std.cpu()

# Standard ImageNet Fallbacks
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])

# Calculated Mean: [0.4987, 0.4987, 0.4987], 
# Std: [0.2240, 0.2240, 0.2240]


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
# 3. BATCH INSPECTION VISUALISER
#    "What does the model actually see?"
# ==============================
def _mask_to_rgb(mask_arr):
    """Convert integer class mask (H×W) to RGB overlay (H×W×3)."""
    h, w = mask_arr.shape
    rgb = np.zeros((h, w, 3), dtype=float)
    for cls_idx, color in enumerate(CLASS_COLORS):
        where = mask_arr == cls_idx
        for ch, v in enumerate(color):
            rgb[:, :, ch][where] = v
    return rgb


def show_batch_inspection(dls, n=4, save_path=None):
    """
    Visualise what the model sees for n TRAIN and n VALID samples.

    For each sample the figure shows:
      Col 1 — raw image (exactly as it enters the model after augmentation)
      Col 2 — ground-truth mask overlaid on the image (alpha blend)
      Col 3 — per-class pixel-count bar chart for that mask

    Colour legend:
      dark grey = background | blue = generator | green = lead | red = abandoned_lead
    """
    fig_rows = []
    labels   = []

    for split_name, dl in [("TRAIN", dls.train), ("VALID", dls.valid)]:
        xb, yb = next(iter(dl))
        xb = xb.cpu(); yb = yb.cpu()
        n_show = min(n, xb.shape[0])
        for i in range(n_show):
            img_t = xb[i]                          # (3, H, W) float tensor 0..1
            msk_t = yb[i].squeeze().numpy()        # (H, W) int

            # --- denormalize image for display ---
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_np = (img_t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

            mask_rgb = _mask_to_rgb(msk_t)
            overlay  = 0.55 * img_np + 0.45 * mask_rgb
            overlay  = overlay.clip(0, 1)

            # --- pixel counts per class ---
            px_counts = [(msk_t == c).sum() for c in range(len(CLASS_NAMES))]
            total_px  = msk_t.size

            fig_rows.append((img_np, overlay, msk_t, px_counts, total_px))
            labels.append(f"{split_name} #{i+1}")

    n_samples = len(fig_rows)
    fig, axes = plt.subplots(n_samples, 3, figsize=(13, 3.6 * n_samples),
                             gridspec_kw={"width_ratios": [1, 1, 0.9]})
    if n_samples == 1:
        axes = [axes]

    for row_idx, ((img_np, overlay, msk_t, px_counts, total_px), label) in \
            enumerate(zip(fig_rows, labels)):

        ax_img, ax_ov, ax_bar = axes[row_idx]

        # Col 1 — raw image
        ax_img.imshow(img_np)
        ax_img.set_title(f"{label}\nraw image (model input)", fontsize=9, pad=4)
        ax_img.axis("off")

        # Col 2 — overlay
        ax_ov.imshow(overlay)
        ax_ov.set_title("ground-truth mask overlay", fontsize=9, pad=4)
        ax_ov.axis("off")

        # Col 3 — bar chart
        pcts  = [100 * c / total_px for c in px_counts]
        y_pos = np.arange(len(CLASS_NAMES))
        bars  = ax_bar.barh(y_pos, pcts, color=CLASS_COLORS, height=0.55)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(CLASS_NAMES, fontsize=8)
        ax_bar.set_xlabel("% of pixels", fontsize=8)
        ax_bar.set_title("pixel distribution", fontsize=9, pad=4)
        ax_bar.invert_yaxis()
        for bar, pct in zip(bars, pcts):
            if pct > 0.1:
                ax_bar.text(pct + 0.3, bar.get_y() + bar.get_height() / 2,
                            f"{pct:.1f}%", va="center", fontsize=7.5)
        ax_bar.spines[["top", "right"]].set_visible(False)

    # Legend
    patches = [mpatches.Patch(color=c, label=n)
               for c, n in zip(CLASS_COLORS, CLASS_NAMES)]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("What the model sees — batch inspection\n"
                 "(after all resizing and augmentation transforms)",
                 fontsize=11, y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"📊 Batch inspection saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ==============================
# 4. DATAFRAME BUILDER
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

    # Guard: stratified split needs ≥2 members per stratum
    n_pos = df["has_abandoned"].sum()
    use_stratify = df["has_abandoned"] if n_pos >= 5 else None
    if use_stratify is None:
        print("⚠️  Too few abandoned-lead images for stratified split — using random split.")

    train_idx, valid_idx = train_test_split(
        df.index, test_size=args.valid_split,
        stratify=use_stratify, random_state=42
    )
    df["is_valid"] = False
    df.loc[valid_idx, "is_valid"] = True

    # ------------------------------------------------------------------
    # BUG FIX 1: oversample ONLY from confirmed train rows, then merge
    # back with the untouched valid set.
    # Original code did pd.concat([df, extra]) where `df` still contained
    # valid rows — if any valid rows were accidentally duplicated they
    # would silently appear in training after reset_index().
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
# 5. LOAD PRETRAINED WEIGHTS (encoder only)
# ==============================
# ENCODER prefix in FastAI UNet state_dict: keys that start with
# "0." through "4." (layer0–layer4 of the ResNet50 backbone).
# Everything else (middle conv, UnetBlocks, head) belongs to the
# decoder and is intentionally left randomly initialised because
# the pretrained decoder was trained on a different class set
# (background / generator / monitor) — not lead.
# All keys in this model start with "layers."
# Encoder = layers.0.* only (entire ResNet50 backbone is wrapped in one group)
# Decoder = layers.1 to layers.11
# Head    = layers.12.* (shape mismatch: old=3 classes, new=4 classes)
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
            # Decoder / head from old model — intentionally ignored
            skipped_decoder += 1
            continue
        if k in new_state and new_state[k].shape == v.shape:
            new_state[k] = v.to(device)
            loaded += 1
        else:
            print(f"  ↳ skipped (shape mismatch): {k}  "
                  f"old={v.shape}  new={new_state.get(k, torch.tensor([])).shape}")
            skipped_mismatch += 1

    learner.model.load_state_dict(new_state)
    print(f"✅ Encoder weights loaded: {loaded} layers transferred")
    print(f"   {skipped_decoder} decoder/head layers intentionally skipped "
          f"(pretrained on monitor ≠ lead — random init is correct)")
    if skipped_mismatch:
        print(f"   ⚠️  {skipped_mismatch} encoder layers skipped due to shape mismatch")
    return learner


# ==============================
# 6. MAIN TRAINING FUNCTION
# ==============================

# ──────────────────────────────────────────────────────────────────
# Combined Focal + Dice Loss
# ──────────────────────────────────────────────────────────────────
# WHY:
#   FocalLoss alone optimises pixel-wise cross-entropy with down-weighting
#   of easy examples — good for class imbalance but does NOT directly
#   optimise the Dice overlap metric we care about.
#   Dice Loss directly maximises overlap between prediction and ground truth
#   but can be unstable when combined with very small classes.
#   Combined loss gets the best of both: stable training + direct Dice optimisation.
#
# FORMULA:
#   L = focal_w * FocalLoss + dice_w * (1 - mean_Dice_per_class)
#
# Default: 50/50 split. Raise dice_w if dice metrics plateau.
# ──────────────────────────────────────────────────────────────────
class FocalDiceLoss(nn.Module):
    """Combined Focal + Dice loss for multi-class segmentation."""

    def __init__(self, weights: torch.Tensor,
                 focal_w: float = 0.5,
                 dice_w:  float = 0.5,
                 gamma:   float = 2.0):
        super().__init__()
        self.focal_w = focal_w
        self.dice_w  = dice_w
        self.focal   = FocalLossFlat(axis=1, weight=weights, gamma=gamma)

    def forward(self, inp: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        # ── Focal loss ───────────────────────────────────────────
        focal = self.focal(inp, targ)

        # ── Soft Dice loss (differentiable) ─────────────────────
        # Use softmax probabilities so gradients flow through all classes
        n_classes = inp.shape[1]
        probs  = inp.softmax(dim=1)                          # (B, C, H, W)
        # One-hot encode target
        targ_oh = torch.zeros_like(probs)
        targ_oh.scatter_(1, targ.unsqueeze(1).long(), 1.0)   # (B, C, H, W)

        # Per-class soft Dice: average over batch and spatial dims
        inter = (probs * targ_oh).sum(dim=(0, 2, 3))        # (C,)
        union = probs.sum(dim=(0, 2, 3)) + targ_oh.sum(dim=(0, 2, 3))  # (C,)
        dice_per_class = (2.0 * inter + 1.0) / (union + 1.0)           # (C,)

        # Mean over all classes (background included — it helps stability)
        dice_loss = 1.0 - dice_per_class.mean()

        return self.focal_w * focal + self.dice_w * dice_loss

    # FastAI requires activation & decodes for inference
    def activation(self, out):  return out.softmax(dim=1)
    def decodes(self, out):     return out.argmax(dim=1)


# NEW: plot_learning_curves() to visualise training history across all phases in one graph.
def plot_learning_curves(csv_path, output_path):
    """อ่านไฟล์ CSV และสร้างกราฟ Learning Curves (Loss, Dice, Sensitivity)"""
    if not csv_path.exists():
        return

    # ใหม่ — อ่าน manual กรอง # ก่อนส่งให้ pandas
    import io
    lines = csv_path.read_text(encoding='utf-8').splitlines()
    clean = '\n'.join(l for l in lines if not l.startswith('#') and l.strip())
    df = pd.read_csv(io.StringIO(clean), on_bad_lines='skip')

    # กรองเอาแถวที่เป็นหัวตารางซ้ำออก (กรณี append หลาย Phase)
    df = df[df['epoch'] != 'epoch'].reset_index(drop=True)
    # แปลงเฉพาะคอลัมน์ที่เป็นตัวเลข
    numeric_cols = ['epoch', 'train_loss', 'valid_loss', 'dice_generator', 'dice_lead', 'abdn_sensitivity', 'dice_abdn']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    phase_boundaries = []
    if 'epoch' in df.columns:
        ep = df['epoch'].dropna().astype(int)
        for i in range(1, len(ep)):
            if ep.iloc[i] < ep.iloc[i-1]:
                phase_boundaries.append(i)


    fig, ax1 = plt.subplots(figsize=(10, 6))

    # พล็อต Train/Valid Loss (แกนซ้าย)
    ax1.plot(df['train_loss'], label='Train Loss', color='blue', linestyle='--')
    ax1.plot(df['valid_loss'], label='Valid Loss', color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # สร้างแกนขวาสำหรับ Metrics (0.0 - 1.0)
    ax2 = ax1.twinx()
    if 'dice_generator' in df.columns:
        ax2.plot(df['dice_generator'], label='Dice (Generator)', color='green')
    if 'abdn_lead_sensitivity' in df.columns:
        ax2.plot(df['abdn_lead_sensitivity'], label='Sensitivity (Abdn Lead)', color='red')
    
    ax2.set_ylabel('Score (0-1)', color='black')
    ax2.set_ylim(0, 1.1)

    plt.title('Training History (All Phases)')
    fig.tight_layout()
    
    # รวม Legend จากทั้งสองแกน
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150)
    print(f"📈 Learning curves saved → {output_path}")
    plt.close()

def finetune(args):
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Save Training Config ────────────────────────────────
    import csv
    from datetime import datetime
    csv_path = out / "training_history.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["# TRAINING CONFIG", datetime.now().strftime("%Y-%m-%d %H:%M")])
        for k, v in {
            "img_size":        args.img_size,
            "patch_size":      args.patch_size,
            "batch_size":      args.batch_size,
            "epochs_decoder":  args.epochs_decoder,
            "epochs_head":     args.epochs_head,
            "epochs_full":     args.epochs_full,
            "model_path":      args.model_path,
            "class_weights":   args.class_weights,
            "loss_func":       "FocalDiceLoss",
            "focal_w":         args.focal_w,
            "dice_w":          args.dice_w,
            "focal_gamma":     args.focal_gamma,
            "backbone":        "resnet50",
            "grad_accum":      args.grad_accum,
            "fp16":            True,
            "lr_phase0":       args.lr_phase0,
            "lr_phase1":       args.lr_phase1,
            "lr_phase2":       args.lr_phase2,
            "multiscale_crop": args.multiscale_crop,
        }.items():
            writer.writerow([f"# {k}", v])
        writer.writerow([])  # blank line คั่นก่อน metrics
    # ────────────────────────────────────────────────────────

    df = build_dataframe(args)
    
    # --- วางโค้ดเช็คจำนวน Abandoned Lead ตรงนี้ ---
    n_valid_abdn = df[df["is_valid"] & df["has_abandoned"]].shape[0]
    n_train_abdn = df[~df["is_valid"] & df["has_abandoned"]].shape[0]
    print(f"🔍 Checking Validation Set: พบ Abandoned Lead {n_valid_abdn} รูป")
    print(f"🔍 Checking Training Set: พบ Abandoned Lead {n_train_abdn} รูป")
    
    if n_valid_abdn == 0:
        print("⚠️ Warning: ไม่มี Abandoned Lead ใน Validation Set เลย! Metric จะขึ้น nan แน่นอน")
    # ------------------------------------------
        
    summarize_dataset(df)
    # 1. Determine Normalization Stats
    
    # Calculated Mean: [0.4987, 0.4987, 0.4987], 
    # CalculatedStd: [0.2240, 0.2240, 0.2240]
    # new Mean=[0.49453404545783997, 0.49453404545783997, 0.49453404545783997], 
    # Std=[0.22670553624629974, 0.22670553624629974, 0.22670553624629974]

    myMEAN = [0.4945, 0.4945, 0.4945]
    mySTD  = [0.2267, 0.2267, 0.2267]
    # stats_mean, stats_std = IMAGENET_MEAN, IMAGENET_STD
    stats_mean, stats_std = myMEAN, mySTD


    if args.calc_stats:
        # Temporary DataBlock without normalization to measure raw pixels
        print("📊 เริ่มคำนวณ stats ใหม่จาก Dataset...")
        stats_db = DataBlock(
            blocks=(ImageBlock, MaskBlock(codes=CLASS_NAMES)),
            get_x=get_x, get_y=get_y,
            splitter=ColSplitter(col='is_valid'),
            item_tfms=Resize(args.img_size, method='pad'), 
            batch_tfms=[IntToFloatTensor()] # Convert 0-255 to 0-1 but NO normalization
        )
        stats_dls = stats_db.dataloaders(df, bs=args.batch_size, num_workers=0)
        stats_mean, stats_std = get_segmentation_stats(stats_dls)
        print(f"  ✅ Custom Stats: Mean={stats_mean.tolist()}, Std={stats_std.tolist()}")

    # DataBlock
    item_tfms_list = []
    if args.multiscale_crop:
        item_tfms_list.append(MultiScaleCrop(img_size=args.img_size))
        print(f"🔀 Multi-scale crop ENABLED (scales: 0.6 / 0.75 / 0.875 / 1.0 × {args.img_size}px) — train only")
    item_tfms_list.append(Resize(args.patch_size, method='pad', pad_mode='zeros'))

    dblock = DataBlock(
        blocks=(ImageBlock, MaskBlock(codes=CLASS_NAMES)),
        get_x=get_x, get_y=get_y,
        splitter=ColSplitter(col='is_valid'),
        item_tfms=item_tfms_list,
                         # resize to patch_size directly — no random crop in batch_tfms
                         # avoids abandoned lead pixels being cropped out on validation
        batch_tfms=[
            *aug_transforms (
                # size removed → no RandomResizedCrop, validation sees full image
                do_flip      = True,
                flip_vert    = False,
                max_rotate   = 15,
                min_zoom     = 0.9,   # tighter zoom — no crop buffer anymore
                max_zoom     = 1.1,
                max_lighting = 0.1,
                max_warp     = 0.0,
                p_affine     = 0.75,
                p_lighting   = 0.5,
            ),      
            Normalize.from_stats(stats_mean, stats_std)
            ],  
    )
    dls = dblock.dataloaders(df, bs=args.batch_size, 
                             num_workers=0, pin_memory=True, persistent_workers=False)
    

    # --- What does the model see? ---
    print("\n📸 Generating batch inspection figures …")
    show_batch_inspection(dls, n=4,
                          save_path=str(out / "batch_inspection.png"))

    # Loss with class weights
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    #weights    = torch.tensor([1.0, 10, 20, 30]).to(device) 
    weights = torch.tensor(args.class_weights, dtype=torch.float32).to(device)
    # FocalDiceLoss = combined Focal (class-imbalance handling) +
    # soft Dice (direct overlap optimisation).
    # focal_w / dice_w control the balance — default 50/50.
    # Raise dice_w (e.g. 0.3/0.7) if dice metrics plateau.
    loss_func = FocalDiceLoss(
        weights = weights,
        focal_w = args.focal_w,
        dice_w  = args.dice_w,
        gamma   = args.focal_gamma,
    )

    learner = unet_learner(
        dls, resnet50, n_out=4,
        loss_func=loss_func,
        metrics=[dice_generator, dice_lead, AbdnLeadSensitivity(), dice_abdn],
        # Add CSVLogger here to capture all phases in one file
        # remove gradient accumulation to avoid NaN issues with abdn sensitivity.
        cbs=[CSVLogger(fname=str(out / "training_history.csv"), append=True)]).to_fp16()

    print("\n📦 Loading pretrained encoder weights …")
    learner = load_pretrained_weights(learner, args.model_path)
    print("image size:", args.img_size, "patch size:", args.patch_size)

    # ─────────────────────────────────────────────────────────────
 
    #print("🔍 finding optimal Learning Rate for Phase 0 (Full Fine-tuning)...")
    #suggestions = learner.lr_find(suggest_funcs=(minimum, steep, valley, slide))
    #print(f"suggestions (Valley): {suggestions.valley}")
    #phase 0 suggestions (Valley): 2.511886486900039e-05

    # ------------------------------------------------------------------
    # Phase 0 — Decoder warmup (encoder frozen)
    # Train the randomly-initialised decoder and head while the encoder
    # stays frozen.  Without this, Phase 1 head-only training starts
    # with a decoder that is completely random, which produces unstable
    # gradients that can corrupt the encoder weights later.
    # ------------------------------------------------------------------
    print("\n--- Phase 0: Decoder warmup (encoder frozen, decoder free) ---")
    # Freeze only the encoder (groups[0] in FastAI UNet = backbone)
    learner.freeze_to(1)          # freeze group 0 (encoder), leave rest free
    learner.fit_one_cycle(args.epochs_decoder, args.lr_phase0, cbs=GradientAccumulation(n_acc=args.grad_accum))  # ลด lr ลงจาก 2e-3
    #learner.show_results(max_n=4, vmin=0, vmax=3)
    #plt.savefig(out / "phase0_decoder_warmup.png")
    #learner.save(out / "after_decoder_warmup")

    # Phase 1 — head only
    print("\n--- Phase 1: Training Head (all except head frozen) ---")
    learner.freeze()              # freeze everything except last param group (head)
    learner.fit_one_cycle(args.epochs_head, args.lr_phase1, cbs=GradientAccumulation(n_acc=args.grad_accum))
    #learner.show_results(max_n=4, vmin=0, vmax=3)
    #plt.savefig(out / "phase1_head_loss.png")
    #learner.save(out / "after_head_only")
    #learner.recorder.plot.loss()


    # Phase 2 — full fine-tune with best-model checkpoint
    print("\n--- Phase 2: Full Fine-Tuning ---")
    learner.unfreeze()

    # 2. show lr_find suggestions for Phase 2 full fine-tuning
    #print("🔍 finding optimal Learning Rate for Phase 2 (Full Fine-tuning)...")
    #suggestions = learner.lr_find(suggest_funcs=(minimum, steep, valley, slide))
    #print(f"suggestions (Valley): {suggestions.valley}")
    # may use the valley suggestion with fit_one_cycle esp on this phase. 
    # for phase 0,1 -- may keep learning rate as they are
    # learner.fit_one_cycle(args.epochs_decoder, suggestions.valley)



    # SaveModelCallback at learner.path/learner.model_dir/
    learner.path      = out
    learner.model_dir = ""
    learner.fit_one_cycle(
        args.epochs_full,
        lr_max=args.lr_phase2,  # slowler ie. 1e-6, 5e-5
        cbs=[GradientAccumulation(n_acc=args.grad_accum),
         SaveModelCallback(monitor='dice_generator', fname='best_gen', with_opt=False),
         SaveModelCallback(monitor='dice_abdn', fname='best_abdn', with_opt=False)
         ]
    )

    # --- Save Sample Predictions ---
    print("🎨 Saving sample predictions...")
    learner.show_results(max_n=4, vmin=0, vmax=3)
    plt.savefig(out / "quick_peek.png")
    plt.close()

    # Load best checkpoint before export
    best_path = out / "best_seg.pth"
    if best_path.exists():
        learner.load(str(out / "best_seg"))
        print("🏆 Loaded best checkpoint (highest dice_generator).")

    # ------------------------------------------------------------------
    # Export — state dict (.pth)
    # no learner.export() to avoid pickle 'code' object error
    # use infer_abdnL.py to load the state dict into a model for inference instead.
    # ------------------------------------------------------------------
    weights_path = out / "seg_abdnL_weights.pth"
    torch.save(learner.model.state_dict(), weights_path)
    print(f"\n✅ Weights saved → {weights_path}")

    # --- Plot Learning Curves ---
    print ("\n📈 Plotting learning curves from training history …")
    plot_learning_curves(out / "training_history.csv", out / "learning_curves.png")

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
    # training config
    parser.add_argument("--oversample_new", type=int,   default=1) # if N increases, set as 1
    parser.add_argument("--class_weights", nargs=4, type=float, default=[1.0, 10, 10, 40],
                        help="Class weights for the loss function (background, generator, lead, abandoned_lead)")
    # model/hyperparameters
    # PS 384: Dice Generator 0.72 but Dice Abdn not that good, PS 320 is better.
    # Dont use PS 256. Numbers are not good.
    parser.add_argument("--img_size",      type=int,   default=512)  # try BS/PS 512/320, 640/384; 768/512
    parser.add_argument("--patch_size",     type=int,   default=320)  # 320 = 5px, 384 = 6px, 448 = 7px, 512 = 8px effective receptive field on original image
    parser.add_argument("--batch_size",     type=int,   default=3) #PS 384: BS 2x4, PS 320 BS: 3x3 
    parser.add_argument("--grad_accum",   type=int,   default=3) 
    parser.add_argument("--valid_split", type=float,   default=0.2)
    parser.add_argument("--abdn_min_512", type=int,   default=2800)  # minimum pixel count at 512x512 to consider "yes" for abandoned lead sensitivity metric
    
    # training epochs
    parser.add_argument("--epochs_decoder",  type=int,   default=5)   # Phase 0: decoder warmup 
    parser.add_argument("--epochs_head",    type=int,   default=5)    # Phase 1: head only
    parser.add_argument("--epochs_full",    type=int,   default=20)  # if N increases, set as 20
    # learning rates   
    parser.add_argument("--lr_phase0",   type=float,   default=2e-3)  # Phase 0: decoder warmup — reduced from 2e-3 to 5e-4 for more stable training with small dataset
    parser.add_argument("--lr_phase1",   type=float,   default=1e-3)  # Phase 1: head only — reduced from 1e-3 to 6e-4 to prevent overfitting and instability with small dataset
    parser.add_argument("--lr_phase2",   type=parse_lr_arg, default="slice(2e-6, 2e-4)")  # Phase 2: full fine-tuning — use a learning rate slice for gradual unfreezing and stable convergence
        
    # FocalDiceLoss parameters
    parser.add_argument("--focal_w",    type=float, default=0.5,
                        help="Weight for Focal component in FocalDiceLoss (0-1). "
                             "focal_w + dice_w should sum to 1.0.")
    parser.add_argument("--dice_w",     type=float, default=0.5,
                        help="Weight for Dice component in FocalDiceLoss (0-1).")
    parser.add_argument("--focal_gamma",type=float, default=2.0,
                        help="Gamma for Focal loss (higher = more focus on hard examples).")

    # Added --calc_stats to the argparse section so you can choose when to perform this calculation.
    parser.add_argument("--calc_stats",action="store_true", default=False,  # change to True to enable stats calculation
                        help="Calculate mean/std from the dataset instead of using ImageNet values")
    parser.add_argument("--multiscale_crop", action="store_true", default=True,
                        help="Enable multi-scale patch sampling: randomly crops at 0.6/0.75/0.875/1.0 x img_size "
                             "before Resize(patch_size). Train only — validation unaffected. No VRAM change.")

    args = parser.parse_args()
    args.new_imgs    = pathlib.Path(args.new_imgs)
    args.new_masks   = pathlib.Path(args.new_masks)
    args.model_path  = pathlib.Path(args.model_path)

    finetune(args)