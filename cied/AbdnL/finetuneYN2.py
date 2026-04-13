# Finetune โมเดล UNet จาก ResNet50 เพื่อเพิ่มคลาส "Abandoned Lead" โดยใช้ FastAI
# Yes/No Report for Abandoned Lead และ Dice เฉพาะ Class 1 (Generator) เพื่อวัดประสิทธิภาพของโมเดลในแต่ละคลาส
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

warnings.filterwarnings("ignore")

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

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
def abdn_lead_sensitivity(inp, targ):
    """Yes/No sensitivity: did the model detect abandoned lead when present?"""
    threshold_pixels = 20
    pred = inp.argmax(dim=1)
    pred_yes = (pred == 3).sum(dim=(1, 2)) > threshold_pixels
    targ_yes = (targ == 3).sum(dim=(1, 2)) > 0

    actual_pos = targ_yes.sum()
    if actual_pos == 0:
        # BUG FIX 2: was torch.tensor(0.0) — use nan so the metric
        # aggregator ignores this batch rather than counting it as a miss.
        return torch.tensor(float('nan'))

    tp = (pred_yes & targ_yes).sum()
    return tp.float() / actual_pos.float()


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
            if pct > 1:
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
def finetune(args):
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = build_dataframe(args)
    summarize_dataset(df)

    # DataBlock
    dblock = DataBlock(
        blocks=(ImageBlock, MaskBlock(codes=CLASS_NAMES)),
        get_x=get_x, get_y=get_y,
        splitter=ColSplitter(col='is_valid'),
        item_tfms=Resize(512),
        batch_tfms=[*aug_transforms(size=args.patch_size, max_warp=0)],  # warp off → preserves lead geometry
    )
    dls = dblock.dataloaders(df, bs=args.batch_size, num_workers=0)

    # --- What does the model see? ---
    print("\n📸 Generating batch inspection figures …")
    show_batch_inspection(dls, n=4,
                          save_path=str(out / "batch_inspection.png"))

    # Loss with class weights
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights    = torch.tensor([1.0, 10.0, 10.0, 50.0]).to(device)
    loss_func  = FocalLossFlat(axis=1, weight=weights)

    learner = unet_learner(
        dls, resnet50, n_out=4,
        loss_func=loss_func,
        metrics=[dice_generator, abdn_lead_sensitivity],
    ).to_fp16()

    print("\n📦 Loading pretrained encoder weights …")
    learner = load_pretrained_weights(learner, args.model_path)

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
    learner.fit_one_cycle(args.epochs_decoder, 1e-3)

    # Phase 1 — head only
    print("\n--- Phase 1: Training Head (all except head frozen) ---")
    learner.freeze()              # freeze everything except last param group (head)
    learner.fit_one_cycle(args.epochs_head, 3e-4)

    # Phase 2 — full fine-tune with best-model checkpoint
    print("\n--- Phase 2: Full Fine-Tuning ---")
    learner.unfreeze()
    learner.fit_one_cycle(
        args.epochs_full,
        lr_max=slice(1e-6, 1e-4),
        cbs=SaveModelCallback(monitor='dice_generator',
                              fname='best_seg',
                              with_opt=False)
    )

    # Load best checkpoint before export
    best_path = out / "best_seg.pth"
    if best_path.exists():
        learner.load(str(out / "best_seg"))
        print("🏆 Loaded best checkpoint (highest dice_generator).")

    model_export_path = out / "seg_abdnL_final.pkl"
    learner.export(model_export_path)
    print(f"\n✅ Model saved → {model_export_path}")


# ==============================
# 7. ENTRY POINT
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",     default="C:/CIEDID_data/pkl/segmentation.pkl")
    parser.add_argument("--new_imgs",       default="C:/CIEDID_data/AbdnL/data")
    parser.add_argument("--new_masks",      default="C:/CIEDID_data/AbdnL/mask")
    parser.add_argument("--output_dir",     default="C:/CIEDID_data/AbdnL/models")
    parser.add_argument("--epochs_decoder",  type=int,   default=10)   # Phase 0: decoder warmup
    parser.add_argument("--epochs_head",    type=int,   default=5)    # Phase 1: head only
    parser.add_argument("--epochs_full",    type=int,   default=20)
    parser.add_argument("--batch_size",     type=int,   default=4)
    parser.add_argument("--patch_size",     type=int,   default=256)
    parser.add_argument("--valid_split",    type=float, default=0.2)
    parser.add_argument("--oversample_new", type=int,   default=5)

    args = parser.parse_args()
    args.new_imgs    = pathlib.Path(args.new_imgs)
    args.new_masks   = pathlib.Path(args.new_masks)
    args.model_path  = pathlib.Path(args.model_path)

    finetune(args)