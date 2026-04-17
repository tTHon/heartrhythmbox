# finetuneYN_cv.py
# finetuning script for abandoned lead segmentation 
# using encoder-only transfer from the original segmentation model (segmentation.pkl).
#
# CHANGES vs finetuneYN_custom.py:
#   [CV] Fixed test set (20% of abandoned-lead cases, stratified) is held out BEFORE
#        any cross-validation.  The remaining abandoned-lead cases are split into K
#        stratified folds.  For each fold's TRAIN split, normal (non-abandoned) cases
#        are added at a 2:1 ratio (normal:abandoned) to help prevent catastrophic
#        forgetting.  Each fold trains all three phases (decoder warmup → head-only →
#        full fine-tune).  The fold with the best validation dice_generator is exported
#        as the final model.
#
# DATA FLOW:
#   all images
#     └─ abandoned-lead images (~50–55)
#         ├─ TEST SET  (~20%, fixed, never used in CV)  → evaluated once at the very end
#         └─ CV POOL  (~80%)
#             ├─ Fold 0 → val fold 0  |  train = remaining abandoned + normal (2:1)
#             ├─ Fold 1 → val fold 1  |  ...
#             └─ Fold K-1 …
#   normal (non-abandoned) images
#     └─ sampled at 2× the number of abandoned-lead TRAIN images per fold
#
# UNCHANGED from finetuneYN_custom.py:
#   - Encoder-only weight transfer from segmentation.pkl
#   - Three-phase training (Phase 0 decoder warmup / Phase 1 head / Phase 2 full)
#   - FocalLoss with class weights
#   - Custom dataset statistics (--calc_stats)
#   - Batch inspection visualiser
#   - State-dict export (.pth)

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
from sklearn.model_selection import StratifiedKFold, train_test_split

# PyTorch 2.6+ fix
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

# ==============================
# 1. GLOBAL FUNCTIONS (outside for Pickle/export)
# ==============================
CLASS_NAMES  = ["background", "generator", "lead", "abandoned_lead"]
CLASS_COLORS = [
    (0.15, 0.15, 0.15),   # background  — dark grey
    (0.20, 0.60, 1.00),   # generator   — blue
    (0.20, 0.85, 0.45),   # lead        — green
    (1.00, 0.25, 0.25),   # abandoned_lead — red
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

def abdn_lead_sensitivity(inp, targ):
    """Yes/No sensitivity: did the model detect abandoned lead when present?
    Returns nan when no positive ground-truth in the batch so FastAI's
    AccumMetric ignores that batch rather than penalising it."""
    threshold_pixels = 1
    pred = inp.argmax(dim=1)
    pred_yes = (pred == 3).sum(dim=(1, 2)) > threshold_pixels
    targ_yes = (targ == 3).sum(dim=(1, 2)) > 0

    actual_pos = targ_yes.sum()
    if actual_pos == 0:
        return torch.tensor(float('nan'))

    tp = (pred_yes & targ_yes).sum()
    return tp.float() / actual_pos.float()

# ──────────────────────────────────────────────────────────────────
# Custom Normalization
# ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def get_segmentation_stats(dls):
    """Calculates mean and std from a FastAI DataLoaders object."""
    print("  📊 Calculating custom dataset statistics (mean/std)...")
    sum_ = torch.zeros(3)
    sum_sq = torch.zeros(3)
    count = 0
    for xb, yb in dls.train:
        b, c, h, w = xb.shape
        num_pixels = b * h * w
        sum_ += torch.sum(xb, dim=[0, 2, 3])
        sum_sq += torch.sum(xb**2, dim=[0, 2, 3])
        count += num_pixels
    mean = sum_ / count
    std = torch.sqrt((sum_sq / count) - (mean**2))
    return mean.cpu(), std.cpu()

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])


# ==============================
# 2. DATASET SUMMARY
# ==============================
def summarize_dataset(df, label=""):
    header = f"========== DATASET SUMMARY {label} =========="
    print(f"\n{header}")
    total = len(df)
    train = len(df[~df["is_valid"]])
    valid = len(df[df["is_valid"]])
    print(f"Total images : {total}  (train={train}, valid={valid})")
    print(f"  Abandoned-lead in train : {df[~df['is_valid'] & df['has_abandoned']].shape[0]}")
    print(f"  Abandoned-lead in valid : {df[ df['is_valid'] & df['has_abandoned']].shape[0]}")

    print("\n--- Pixel-level distribution (train split) ---")
    pixel_counts = {i: 0 for i in range(len(CLASS_NAMES))}
    for m in df.loc[~df["is_valid"], "mask"].unique():
        mask = np.array(PILImage.create(m))
        vals, counts = np.unique(mask, return_counts=True)
        for v, c in zip(vals, counts):
            pixel_counts[int(v)] += int(c)
    total_pixels = sum(pixel_counts.values())
    for i, name in enumerate(CLASS_NAMES):
        pct = 100 * pixel_counts[i] / total_pixels if total_pixels > 0 else 0
        print(f"  Class {i} ({name}): {pixel_counts[i]:,} px ({pct:.2f}%)")
    print("=" * len(header) + "\n")


# ==============================
# 3. BATCH INSPECTION VISUALISER
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
    """Visualise raw image + mask overlay + pixel distribution for n samples."""
    fig_rows = []
    labels   = []

    for split_name, dl in [("TRAIN", dls.train), ("VALID", dls.valid)]:
        xb, yb = next(iter(dl))
        xb = xb.cpu(); yb = yb.cpu()
        n_show = min(n, xb.shape[0])
        for i in range(n_show):
            img_t = xb[i]
            msk_t = yb[i].squeeze().numpy()

            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_np = (img_t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

            mask_rgb = _mask_to_rgb(msk_t)
            overlay  = (0.55 * img_np + 0.45 * mask_rgb).clip(0, 1)
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
        ax_img.imshow(img_np)
        ax_img.set_title(f"{label}\nraw image (model input)", fontsize=9, pad=4)
        ax_img.axis("off")
        ax_ov.imshow(overlay)
        ax_ov.set_title("ground-truth mask overlay", fontsize=9, pad=4)
        ax_ov.axis("off")

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
# 4. DATA SPLITTING
# ==============================
def load_all_images(args):
    """
    Scan image/mask directories and return two DataFrames:
        abdn_df  — images that contain at least one abandoned-lead pixel (class 3)
        normal_df — all other images
    """
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
            "has_abandoned": bool(3 in np.unique(arr)),
        })

    df = pd.DataFrame(rows)
    abdn_df   = df[df["has_abandoned"]].reset_index(drop=True)
    normal_df = df[~df["has_abandoned"]].reset_index(drop=True)
    print(f"\n📂 Found {len(abdn_df)} abandoned-lead images, "
          f"{len(normal_df)} normal images.")
    return abdn_df, normal_df


def split_test_set(abdn_df, test_size=0.20, random_state=42):
    """
    Hold out a fixed test set from the abandoned-lead pool.
    Normal images are NOT included in the test set (they are not the primary
    evaluation target; the test set is used to judge abandoned-lead detection).

    Returns:
        cv_abdn_df   — abandoned-lead images available for cross-validation
        test_abdn_df — held-out test images (never touched during CV)
    """
    n_test = max(1, round(len(abdn_df) * test_size))
    # stratification not meaningful here (all positives); use random split
    test_idx = abdn_df.sample(n=n_test, random_state=random_state).index
    test_abdn_df = abdn_df.loc[test_idx].reset_index(drop=True)
    cv_abdn_df   = abdn_df.drop(index=test_idx).reset_index(drop=True)
    print(f"🔒 Test set  : {len(test_abdn_df)} abandoned-lead images (held out)")
    print(f"🔄 CV pool   : {len(cv_abdn_df)} abandoned-lead images")
    return cv_abdn_df, test_abdn_df


def build_fold_dataframe(fold_train_abdn, fold_val_abdn, normal_df,
                         normal_ratio=2, random_state=42):
    """
    Build a single fold's DataFrame.

    Train set  = fold_train_abdn  +  normal images sampled at normal_ratio:1
    Valid set  = fold_val_abdn    (abandoned-lead only — the held-out fold)

    normal_ratio : int
        Number of normal images to include per abandoned-lead training image.
        Default 2 → 2:1 normal:abandoned in the training split.
    """
    n_abdn_train = len(fold_train_abdn)
    n_normal_needed = normal_ratio * n_abdn_train

    if len(normal_df) >= n_normal_needed:
        sampled_normal = normal_df.sample(n=n_normal_needed,
                                          random_state=random_state)
    else:
        # Not enough normal images — use all (with a warning)
        sampled_normal = normal_df.copy()
        print(f"  ⚠️  Only {len(normal_df)} normal images available; "
              f"wanted {n_normal_needed} for {normal_ratio}:1 ratio.")

    train_df = pd.concat([fold_train_abdn, sampled_normal],
                         ignore_index=True)
    train_df["is_valid"] = False

    val_df = fold_val_abdn.copy()
    val_df["is_valid"] = True

    df = pd.concat([train_df, val_df], ignore_index=True)

    print(f"  Fold train : {len(train_df)} images  "
          f"({n_abdn_train} abandoned + {len(sampled_normal)} normal)")
    print(f"  Fold valid : {len(val_df)} images  "
          f"({len(val_df)} abandoned)")
    return df


# ==============================
# 5. LOAD PRETRAINED WEIGHTS (encoder only)
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
# 6. SINGLE-FOLD TRAINING
# ==============================
def train_one_fold(fold_idx, df, args, out, stats_mean, stats_std):
    """
    Run the full three-phase training for one CV fold.
    Returns the best validation dice_generator score for this fold.
    """
    fold_out = out / f"fold_{fold_idx}"
    fold_out.mkdir(parents=True, exist_ok=True)

    dblock = DataBlock(
        blocks=(ImageBlock, MaskBlock(codes=CLASS_NAMES)),
        get_x=get_x, get_y=get_y,
        splitter=ColSplitter(col='is_valid'),
        item_tfms=Resize(args.img_size),
        batch_tfms=[
            *aug_transforms(size=args.patch_size, max_warp=0),
            Normalize.from_stats(stats_mean, stats_std)
        ],
    )
    dls = dblock.dataloaders(df, bs=args.batch_size, num_workers=0)

    print(f"\n📸 [Fold {fold_idx}] Generating batch inspection …")
    show_batch_inspection(dls, n=4,
                          save_path=str(fold_out / "batch_inspection.png"))

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights   = torch.tensor([1.0, 10.0, 10.0, 20.0]).to(device)
    loss_func = FocalLossFlat(axis=1, weight=weights)

    learner = unet_learner(
        dls, resnet50, n_out=4,
        loss_func=loss_func,
        metrics=[dice_generator, abdn_lead_sensitivity],
        cbs=[CSVLogger(fname=str(fold_out / "training_history.csv"), append=True)]
    ).to_fp16()

    print(f"\n📦 [Fold {fold_idx}] Loading pretrained encoder weights …")
    learner = load_pretrained_weights(learner, args.model_path)

    # Phase 0 — decoder warmup
    print(f"\n--- [Fold {fold_idx}] Phase 0: Decoder warmup ---")
    learner.freeze_to(1)
    learner.fit_one_cycle(args.epochs_decoder, 1e-3)
    learner.show_results(max_n=4, vmin=0, vmax=3)
    plt.savefig(fold_out / "phase0_decoder_warmup.png")
    plt.close()

    # Phase 1 — head only
    print(f"\n--- [Fold {fold_idx}] Phase 1: Head only ---")
    learner.freeze()
    learner.fit_one_cycle(args.epochs_head, 3e-4)
    learner.show_results(max_n=4, vmin=0, vmax=3)
    plt.savefig(fold_out / "phase1_head.png")
    plt.close()

    # Phase 2 — full fine-tune with best-model checkpoint
    print(f"\n--- [Fold {fold_idx}] Phase 2: Full fine-tune ---")
    learner.unfreeze()
    learner.path      = fold_out
    learner.model_dir = ""
    learner.fit_one_cycle(
        args.epochs_full,
        lr_max=slice(1e-6, 1e-4),
        cbs=SaveModelCallback(monitor='dice_generator',
                              fname='best_seg',
                              with_opt=False)
    )
    learner.show_results(max_n=4, vmin=0, vmax=3)
    plt.savefig(fold_out / "phase2_predictions.png")
    learner.recorder.plot_loss()
    plt.savefig(fold_out / "learning_stats_plot.png")
    plt.close()

    # Load best checkpoint for this fold
    best_path = fold_out / "best_seg.pth"
    if best_path.exists():
        learner.load(str(fold_out / "best_seg"))

    # Record best validation dice_generator
    history = pd.read_csv(fold_out / "training_history.csv")
    best_dice = history["dice_generator"].max()
    print(f"\n🏅 [Fold {fold_idx}] Best dice_generator = {best_dice:.4f}")

    # Save fold weights
    weights_path = fold_out / "seg_abdnL_weights.pth"
    torch.save(learner.model.state_dict(), weights_path)
    print(f"✅ [Fold {fold_idx}] Weights saved → {weights_path}")

    return best_dice, learner


# ==============================
# 7. MAIN CROSS-VALIDATION FUNCTION
# ==============================
def finetune(args):
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 7a. Load all images ──────────────────────────────────────────
    abdn_df, normal_df = load_all_images(args)

    if len(abdn_df) == 0:
        raise RuntimeError("No abandoned-lead images found. Check --new_imgs / --new_masks.")

    # ── 7b. Hold out fixed test set ──────────────────────────────────
    cv_abdn_df, test_abdn_df = split_test_set(
        abdn_df, test_size=args.test_split, random_state=42
    )
    # Save test set index so it is always reproducible
    test_abdn_df.to_csv(out / "test_set.csv", index=False)
    print(f"💾 Test set saved → {out / 'test_set.csv'}")

    # ── 7c. Determine normalization stats from ALL cv data (quick pass) ──
    stats_mean, stats_std = IMAGENET_MEAN, IMAGENET_STD
    if args.calc_stats:
        # Use a temporary single-split DataLoaders built from the full CV pool
        # (treating everything as "train") to calculate pixel statistics.
        temp_df = pd.concat([cv_abdn_df, normal_df], ignore_index=True)
        temp_df["is_valid"] = False
        # Add a tiny dummy valid row so DataBlock doesn't complain
        dummy   = temp_df.iloc[[0]].copy()
        dummy["is_valid"] = True
        temp_df = pd.concat([temp_df, dummy], ignore_index=True)

        stats_db = DataBlock(
            blocks=(ImageBlock, MaskBlock(codes=CLASS_NAMES)),
            get_x=get_x, get_y=get_y,
            splitter=ColSplitter(col='is_valid'),
            item_tfms=Resize(args.img_size),
            batch_tfms=[IntToFloatTensor()]
        )
        stats_dls = stats_db.dataloaders(temp_df, bs=args.batch_size, num_workers=0)
        stats_mean, stats_std = get_segmentation_stats(stats_dls)
        print(f"  ✅ Custom Stats: Mean={stats_mean.tolist()}, Std={stats_std.tolist()}")

    # ── 7d. K-Fold Cross-Validation ──────────────────────────────────
    n_splits = args.n_folds
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # StratifiedKFold needs a label array — use all-ones (all are abandoned)
    # but we still benefit from even splits.
    dummy_labels = np.ones(len(cv_abdn_df), dtype=int)

    fold_scores  = []   # (fold_idx, best_dice)
    fold_models  = []   # learner reference for best fold

    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(cv_abdn_df, dummy_labels)):

        print(f"\n{'='*60}")
        print(f"  FOLD {fold_idx + 1} / {n_splits}")
        print(f"{'='*60}")

        fold_train_abdn = cv_abdn_df.iloc[train_idx].reset_index(drop=True)
        fold_val_abdn   = cv_abdn_df.iloc[val_idx].reset_index(drop=True)

        df = build_fold_dataframe(
            fold_train_abdn, fold_val_abdn, normal_df,
            normal_ratio=args.normal_ratio, random_state=42
        )
        summarize_dataset(df, label=f"[Fold {fold_idx + 1}/{n_splits}]")

        best_dice, learner = train_one_fold(
            fold_idx + 1, df, args, out, stats_mean, stats_std
        )
        fold_scores.append((fold_idx + 1, best_dice))

    # ── 7e. Cross-validation summary ─────────────────────────────────
    print(f"\n{'='*60}")
    print("  CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    scores_arr = np.array([s for _, s in fold_scores])
    for fold_idx, score in fold_scores:
        marker = " ← best" if score == scores_arr.max() else ""
        print(f"  Fold {fold_idx}: dice_generator = {score:.4f}{marker}")
    print(f"\n  Mean  ± Std : {scores_arr.mean():.4f} ± {scores_arr.std():.4f}")
    print(f"  Best fold   : Fold {fold_scores[scores_arr.argmax()][0]}")

    # Save CV summary
    cv_summary = pd.DataFrame(fold_scores, columns=["fold", "best_dice_generator"])
    cv_summary.loc[len(cv_summary)] = ["mean±std",
                                        f"{scores_arr.mean():.4f}±{scores_arr.std():.4f}"]
    cv_summary.to_csv(out / "cv_summary.csv", index=False)
    print(f"💾 CV summary saved → {out / 'cv_summary.csv'}")

    # ── 7f. Copy best fold weights as the final model ─────────────────
    best_fold_idx = fold_scores[scores_arr.argmax()][0]
    best_weights_src = out / f"fold_{best_fold_idx}" / "seg_abdnL_weights.pth"
    final_weights    = out / "seg_abdnL_final.pth"
    import shutil
    shutil.copy(best_weights_src, final_weights)
    print(f"\n🏆 Final model (Fold {best_fold_idx}) copied → {final_weights}")

    # ── 7g. Reminder: evaluate on held-out test set ───────────────────
    print(f"\n📋 Held-out test set ({len(test_abdn_df)} abandoned-lead images) "
          f"saved at: {out / 'test_set.csv'}")
    print("   Run inference with infer_abdnL.py using seg_abdnL_final.pth "
          "to get unbiased final performance metrics.")
    print(f"{'='*60}\n")


# ==============================
# 8. ENTRY POINT
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",      default="C:/CIEDID_data/pkl/segmentation.pkl")
    parser.add_argument("--new_imgs",        default="C:/CIEDID_data/AbdnL/data")
    parser.add_argument("--new_masks",       default="C:/CIEDID_data/AbdnL/mask")
    parser.add_argument("--output_dir",      default="C:/CIEDID_data/AbdnL/models")
    parser.add_argument("--img_size",        type=int,   default=512)
    parser.add_argument("--patch_size",      type=int,   default=256)
    parser.add_argument("--epochs_decoder",  type=int,   default=3,
                        help="Phase 0: decoder warmup epochs (per fold)")
    parser.add_argument("--epochs_head",     type=int,   default=3,
                        help="Phase 1: head-only epochs (per fold)")
    parser.add_argument("--epochs_full",     type=int,   default=5,
                        help="Phase 2: full fine-tune epochs (per fold)")
    parser.add_argument("--batch_size",      type=int,   default=2)
    parser.add_argument("--n_folds",         type=int,   default=5,
                        help="Number of CV folds (default 5; use 4 for ~40 CV images)")
    parser.add_argument("--test_split",      type=float, default=0.20,
                        help="Fraction of abandoned-lead images held out as final test set")
    parser.add_argument("--normal_ratio",    type=int,   default=2,
                        help="Normal images per abandoned-lead image in each fold's train split")
    parser.add_argument("--calc_stats",      action="store_true",
                        help="Calculate mean/std from the CV pool instead of ImageNet values")

    args = parser.parse_args()
    args.new_imgs   = pathlib.Path(args.new_imgs)
    args.new_masks  = pathlib.Path(args.new_masks)
    args.model_path = pathlib.Path(args.model_path)

    finetune(args)