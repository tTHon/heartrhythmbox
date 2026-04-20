# finetuneYN_cv.py
# finetuning script for abandoned lead segmentation 
# using encoder-only transfer from the original segmentation model (segmentation.pkl).
#
# DATA SOURCE:
#   Reads C:/cied_data/workbook.csv which must contain at minimum:
#       ID          — filename stem (image = <ID>.png, mask = <ID>.png in mask folder)
#       finalTest   — 1 = held-out test set, 0 = available for CV training
#   Images  : C:/cied_data/images/<ID>.png
#   Masks   : C:/cied_data/masks/<ID>.png   (or override with --mask_dir)
#   Output  : C:/cied_data/model/

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
from sklearn.model_selection import StratifiedKFold

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
# 4. DATA LOADING
# ==============================
def scan_from_folders(args):
    """
    Scan mask_dir for all .png mask files and pair each with its image.

    For every <ID>.png found in mask_dir:
      - image is looked up as <img_dir>/<ID>.png  (fallback: .jpg)
      - has_abandoned is detected automatically:
            any pixel == 3 in the mask → True

    No CSV needed. Put only the cases you want to train in mask_dir.

    Returns:
        cv_abdn_df  — masks containing class 3  (abandoned lead present)
        normal_df   — masks with no class 3     (normal cases)
    """
    img_dir  = pathlib.Path(args.img_dir)
    mask_dir = pathlib.Path(args.mask_dir)

    rows    = []
    missing = []

    mask_files = sorted(mask_dir.glob("*.png"))
    if not mask_files:
        raise RuntimeError(f"No .png mask files found in {mask_dir}")

    for msk_path in mask_files:
        sid = msk_path.stem

        img_path = img_dir / f"{sid}.png"
        if not img_path.exists():
            img_path = img_dir / f"{sid}.jpg"

        if not img_path.exists():
            missing.append(str(img_dir / sid))
            continue

        arr = np.array(PILImage.create(msk_path))
        rows.append({
            "ID":            sid,
            "image":         str(img_path.resolve()),
            "mask":          str(msk_path.resolve()),
            "has_abandoned": bool(3 in np.unique(arr)),
        })

    if missing:
        print(f"  ⚠️  {len(missing)} mask(s) have no matching image — skipped:")
        for m in missing[:10]:
            print(f"       {m}(.png/.jpg)")
        if len(missing) > 10:
            print(f"       … and {len(missing) - 10} more")

    df = pd.DataFrame(rows)
    cv_abdn_df = df[df["has_abandoned"]].reset_index(drop=True)
    normal_df  = df[~df["has_abandoned"]].reset_index(drop=True)

    print(f"\n📂 Scanned {len(df)} image-mask pairs from {mask_dir.name}/")
    print(f"   ↳ abandoned-lead : {len(cv_abdn_df)}")
    print(f"   ↳ normal         : {len(normal_df)}")

    if len(cv_abdn_df) == 0:
        raise RuntimeError(
            "No abandoned-lead cases found (no mask has class-3 pixels). "
            f"Check mask files in {mask_dir}"
        )
    return cv_abdn_df, normal_df


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

    # aug_transforms tuned for chest X-ray lead segmentation:
    #   do_flip=True        horizontal flip OK (pacemaker left OR right sided)
    #   flip_vert=False     upside-down X-ray never happens clinically
    #   max_rotate=10       mild tilt — patient positioning variation
    #   min_zoom=0.9        slight zoom-out to see full lead course
    #   max_zoom=1.15       slight zoom-in without losing context
    #   max_lighting=0.2    simulate exposure/contrast variation between scanners
    #   max_warp=0          MUST stay 0 — warp distorts lead geometry/trajectory
    #   p_affine=0.75       apply spatial augmentation 75% of the time
    #   p_lighting=0.75     apply lighting augmentation 75% of the time
    dblock = DataBlock(
        blocks=(ImageBlock, MaskBlock(codes=CLASS_NAMES)),
        get_x=get_x, get_y=get_y,
        splitter=ColSplitter(col='is_valid'),
        item_tfms=Resize(args.img_size),
        batch_tfms=[
            *aug_transforms(
                size         = args.patch_size,
                do_flip      = True,
                flip_vert    = False,
                max_rotate   = 10,
                min_zoom     = 0.9,
                max_zoom     = 1.15,
                max_lighting = 0.1, # reduce from 0.2 since we CLAHE the image
                max_warp     = 0.0,
                p_affine     = 0.75,
                p_lighting   = 0.5, # reduce from 0.75 due to CLAHE
            ),
            Normalize.from_stats(stats_mean, stats_std),
        ],
    )
    dls = dblock.dataloaders(df, bs=args.batch_size, num_workers=0)

    print(f"\n📸 [Fold {fold_idx}] Generating batch inspection …")
    show_batch_inspection(dls, n=4,
                          save_path=str(fold_out / "batch_inspection.png"))

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights   = torch.tensor([1.0, 10.0, 10.0, 20.0]).to(device)
    loss_func = FocalLossFlat(axis=1, weight=weights)

    # GradientAccumulation(n) simulates effective batch = batch_size × n
    # without extra VRAM cost. With batch_size=2 and n=4 → effective batch=8.
    # LRs below are scaled by √n ≈ 2× relative to batch_size=2 baseline.
    learner = unet_learner(
        dls, resnet50, n_out=4,
        loss_func=loss_func,
        metrics=[dice_generator, abdn_lead_sensitivity],
        cbs=[
            CSVLogger(fname=str(fold_out / "training_history.csv"), append=True),
            GradientAccumulation(n_acc=args.grad_accum),
        ]
    ).to_fp16()

    print(f"\n📦 [Fold {fold_idx}] Loading pretrained encoder weights …")
    learner = load_pretrained_weights(learner, args.model_path)

    # LR scaling note: base LRs assume effective batch=8 (batch_size=2 × grad_accum=4).
    # Using √grad_accum scaling: Phase0 2e-3, Phase1 6e-4, Phase2 slice(2e-6, 2e-4).
    # If you change --grad_accum, rescale LRs by √(new/4) accordingly.

    # Phase 0 — decoder warmup
    print(f"\n--- [Fold {fold_idx}] Phase 0: Decoder warmup ---")
    learner.freeze_to(1)
    learner.fit_one_cycle(args.epochs_decoder, 2e-3)
    learner.show_results(max_n=4, vmin=0, vmax=3)
    plt.savefig(fold_out / "phase0_decoder_warmup.png")
    plt.close()

    # Phase 1 — head only
    print(f"\n--- [Fold {fold_idx}] Phase 1: Head only ---")
    learner.freeze()
    learner.fit_one_cycle(args.epochs_head, 6e-4)
    learner.show_results(max_n=4, vmin=0, vmax=3)
    plt.savefig(fold_out / "phase1_head.png")
    plt.close()

    # Phase 2 — full fine-tune with best-model checkpoint + early stopping
    # EarlyStoppingCallback(patience=6): stop if val dice_generator does not
    # improve for 6 consecutive epochs — prevents overfitting on small dataset
    # and saves time when the model has already converged.
    print(f"\n--- [Fold {fold_idx}] Phase 2: Full fine-tune ---")
    learner.unfreeze()
    learner.path      = fold_out
    learner.model_dir = ""
    learner.fit_one_cycle(
        args.epochs_full,
        lr_max=slice(2e-6, 2e-4),
        cbs=[
            SaveModelCallback(monitor='dice_generator',
                              fname='best_seg',
                              with_opt=False),
            EarlyStoppingCallback(monitor='dice_generator',
                                  patience=args.early_stop_patience),
        ]
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

    summary_path = out / "cv_summary.csv"

    # ── 7a. Resume logic ─────────────────────────────────────────────
    completed_folds = []
    if summary_path.exists():
        ans = input("📊 พบงานเดิมที่ทำค้างไว้ ต้องการ Resume ต่อไหม? (y/n): ").strip().lower()
        if ans == "y":
            old_summary = pd.read_csv(summary_path)
            completed_folds = (
                old_summary[pd.to_numeric(old_summary["fold"], errors="coerce").notnull()]
                ["fold"].astype(int).tolist()
            )
            print(f"⏩ จะข้าม Fold ที่ทำเสร็จแล้ว: {completed_folds}")
        else:
            print("🧹 เริ่มต้นใหม่ — ลบข้อมูลเดิม...")
            shutil.rmtree(out)
            out.mkdir(parents=True, exist_ok=True)

    # ── 7b. Load data ─────────────────────────────────────────────────
    cv_abdn_df, normal_df = scan_from_folders(args)

    # ── 7c. Normalization stats ───────────────────────────────────────
    stats_mean, stats_std = IMAGENET_MEAN, IMAGENET_STD
    if args.calc_stats:
        temp_df = pd.concat([cv_abdn_df, normal_df], ignore_index=True)
        temp_df["is_valid"] = False
        dummy = temp_df.iloc[[0]].copy()
        dummy["is_valid"] = True
        temp_df = pd.concat([temp_df, dummy], ignore_index=True)

        stats_db = DataBlock(
            blocks=(ImageBlock, MaskBlock(codes=CLASS_NAMES)),
            get_x=get_x, get_y=get_y,
            splitter=ColSplitter(col="is_valid"),
            item_tfms=Resize(args.img_size),
            batch_tfms=[IntToFloatTensor()]
        )
        stats_dls = stats_db.dataloaders(temp_df, bs=args.batch_size, num_workers=0)
        stats_mean, stats_std = get_segmentation_stats(stats_dls)
        print(f"  ✅ Custom Stats: Mean={stats_mean.tolist()}, Std={stats_std.tolist()}")

    # ── 7d. K-Fold CV ─────────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    # All CV cases are abandoned (label=1); StratifiedKFold still gives even splits
    dummy_labels = np.ones(len(cv_abdn_df), dtype=int)

    fold_scores = []  # [(fold_num, best_dice), ...]

    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(cv_abdn_df, dummy_labels)):

        fold_num = fold_idx + 1

        # ── Resume: skip completed folds ──────────────────────────────
        if fold_num in completed_folds:
            print(f"\n⏩ Fold {fold_num} / {args.n_folds} — already done, skipping.")
            old_summary = pd.read_csv(summary_path)
            saved = old_summary[old_summary["fold"].astype(str) == str(fold_num)]
            if not saved.empty:
                fold_scores.append((fold_num, float(saved["best_dice_generator"].iloc[0])))
            continue

        print(f"\n{'='*60}")
        print(f"  FOLD {fold_num} / {args.n_folds}")
        print(f"{'='*60}")

        fold_train_abdn = cv_abdn_df.iloc[train_idx].reset_index(drop=True)
        fold_val_abdn   = cv_abdn_df.iloc[val_idx].reset_index(drop=True)

        # Normal cases: ALL normal rows go into every fold's train split.
        # Ratio is controlled by workbook composition, not by code.
        df = pd.concat([
            fold_train_abdn.assign(is_valid=False),
            normal_df.assign(is_valid=False),
            fold_val_abdn.assign(is_valid=True),
        ], ignore_index=True)

        summarize_dataset(df, label=f"[Fold {fold_num}/{args.n_folds}]")

        best_dice, _ = train_one_fold(
            fold_num, df, args, out, stats_mean, stats_std
        )
        fold_scores.append((fold_num, best_dice))

        # ── Save summary immediately after each fold ──────────────────
        new_row = pd.DataFrame([[fold_num, best_dice]],
                               columns=["fold", "best_dice_generator"])
        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
            summary_df = summary_df[summary_df["fold"].astype(str) != str(fold_num)]
            summary_df = pd.concat([summary_df, new_row], ignore_index=True)
        else:
            summary_df = new_row
        summary_df.to_csv(summary_path, index=False)
        print(f"💾 Progress saved → {summary_path}")

        # ── Ask before next fold ──────────────────────────────────────
        if fold_num < args.n_folds:
            cont = input(
                f"\n✅ จบ Fold {fold_num} แล้ว "
                f"จะทำ Fold {fold_num + 1} ต่อเลยไหม? (y/n): "
            ).strip().lower()
            if cont != "y":
                print("👋 บันทึกสถานะแล้ว รันใหม่แล้วเลือก Resume เพื่อทำต่อครับ")
                sys.exit(0)

    # ── 7e. Final summary (only when all folds done) ──────────────────
    if len(fold_scores) == args.n_folds:
        print(f"\n{'='*60}")
        print("  CROSS-VALIDATION SUMMARY")
        print(f"{'='*60}")
        scores_arr = np.array([s for _, s in fold_scores])
        for fn, score in fold_scores:
            marker = " ← best" if score == scores_arr.max() else ""
            print(f"  Fold {fn}: dice_generator = {score:.4f}{marker}")
        print(f"\n  Mean ± Std : {scores_arr.mean():.4f} ± {scores_arr.std():.4f}")

        # Append mean±std row to summary
        summary_df = pd.read_csv(summary_path)
        summary_df = summary_df[pd.to_numeric(summary_df["fold"], errors="coerce").notnull()]
        mean_row = pd.DataFrame(
            [["mean±std", f"{scores_arr.mean():.4f}±{scores_arr.std():.4f}"]],
            columns=["fold", "best_dice_generator"]
        )
        pd.concat([summary_df, mean_row], ignore_index=True).to_csv(summary_path, index=False)

        # Copy best fold weights as final model
        best_fold_num    = fold_scores[scores_arr.argmax()][0]
        best_weights_src = out / f"fold_{best_fold_num}" / "seg_abdnL_weights.pth"
        final_weights    = out / "seg_abdnL_final.pth"
        shutil.copy(best_weights_src, final_weights)
        print(f"\n🏆 Final model (Fold {best_fold_num}) → {final_weights}")
        print(f"{'='*60}\n")
    else:
        remaining = args.n_folds - len(fold_scores)
        print(f"\n⏸  {remaining} fold(s) remaining — run again and choose Resume.")


# ==============================
# 8. ENTRY POINT
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  default="C:/cied_data/pkl/segmentation.pkl",
                        help="Path to pretrained segmentation.pkl")
    parser.add_argument("--img_dir",    default="C:/cied_data/AbdnL/data",
                        help="Folder containing <ID>.png or <ID>.jpg image files")
    parser.add_argument("--mask_dir",   default="C:/cied_data/AbdnL/mask",
                        help="Folder containing <ID>.png mask files — "
                             "every mask here will be included in training")
    parser.add_argument("--output_dir", default="C:/cied_data/model")
    parser.add_argument("--img_size",   type=int, default=512,
                        help="Resize all images to this square size before augmentation")
    parser.add_argument("--patch_size", type=int, default=320,
                        help="aug_transforms crop size (model actual input). "
                             "320 suits 8 GB VRAM; try 384 if memory allows.")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps. Effective batch = batch_size x grad_accum.")
    parser.add_argument("--epochs_decoder",      type=int, default=5,
                        help="Phase 0: decoder warmup epochs (per fold)")
    parser.add_argument("--epochs_head",         type=int, default=5,
                        help="Phase 1: head-only epochs (per fold)")
    parser.add_argument("--epochs_full",         type=int, default=20,
                        help="Phase 2: full fine-tune epochs (per fold); "
                             "EarlyStoppingCallback stops early if val dice plateaus")
    parser.add_argument("--early_stop_patience", type=int, default=6,
                        help="Phase 2: stop if dice_generator does not improve for N epochs")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--n_folds",    type=int, default=5,
                        help="Number of CV folds")
    parser.add_argument("--calc_stats", action="store_true", default = False,   
                        help="Calculate mean/std from data instead of using ImageNet values")

    args = parser.parse_args()
    args.img_dir    = pathlib.Path(args.img_dir)
    args.mask_dir   = pathlib.Path(args.mask_dir)
    args.model_path = pathlib.Path(args.model_path)

    finetune(args)