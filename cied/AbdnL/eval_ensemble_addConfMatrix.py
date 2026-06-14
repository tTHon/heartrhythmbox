"""
eval_ensemble.py
================
Evaluate 5-fold ensemble segmentation model on held-out test set.

Instead of using a single model, loads all 5 fold weights and averages
their softmax probabilities before computing predictions:

    ensemble_prob = (prob_fold0 + prob_fold1 + ... + prob_fold4) / 5
    pred          = ensemble_prob.argmax(dim=1)

This reduces variance from individual fold performance (SD=0.072 on
dice_abdn) and produces more stable, generalizable predictions.

Usage:
    python eval_ensemble.py

    # With explicit fold weights directory
    python eval_ensemble.py --folds_dir C:/CIEDID_data/AbdnL/models

    # Compare single vs ensemble
    python eval_ensemble.py --also_single --single_weights fold_0/best_gen.pth
"""

import argparse
import pathlib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from fastai.vision.all import *
from tqdm import tqdm

# ── PyTorch 2.6+ fix ────────────────────────────────────────────────
_orig_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault('weights_only', False)
    return _orig_load(*a, **kw)
torch.load = _patched_load

import platform
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

CLASS_NAMES  = ["background", "generator", "lead", "abandoned_lead"]
CLASS_COLORS = [(0.15,0.15,0.15),(0.20,0.60,1.00),(0.20,0.85,0.45),(1.00,0.25,0.25)]
ABDN_CLASS   = 3
N_CLASSES    = len(CLASS_NAMES)
SMOOTH       = 1e-6


# ══════════════════════════════════════════════════════════════════════
# 1. Build test dataframe
# ══════════════════════════════════════════════════════════════════════
def build_test_dataframe(img_dir, mask_dir):
    rows = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in {".jpg", ".png", ".jpeg", ".bmp"}:
            continue
        mask = mask_dir / f"{img.stem}_mask.png"
        if not mask.exists():
            mask = mask_dir / f"{img.stem}.png"
        if not mask.exists():
            print(f"  ⚠️  No mask for {img.name} — skipped")
            continue
        arr = np.array(PILImage.create(mask))
        rows.append({
            "image":         str(img.resolve()),
            "mask":          str(mask.resolve()),
            "has_abandoned": ABDN_CLASS in np.unique(arr),
            "is_valid":      True,
        })
    df = pd.DataFrame(rows)
    print(f"\n✅ Test set: {len(df)} images "
          f"({df['has_abandoned'].sum()} with abandoned lead  |  "
          f"{(~df['has_abandoned']).sum()} without)")
    return df


# ══════════════════════════════════════════════════════════════════════
# 2. Build DataLoaders (shared across all fold models)
# ══════════════════════════════════════════════════════════════════════
def build_dls(test_df, img_size, device):
    def get_x(r): return r["image"]
    def get_y(r): return r["mask"]

    BS_DUMMY = 4
    dummy    = pd.concat([test_df.iloc[[0]]] * BS_DUMMY, ignore_index=True)
    dummy["is_valid"] = False
    eval_df  = pd.concat([dummy, test_df], ignore_index=True)

    dblock = DataBlock(
        blocks    = (ImageBlock, MaskBlock(codes=CLASS_NAMES)),
        get_x     = get_x,
        get_y     = get_y,
        splitter  = ColSplitter(col="is_valid"),
        item_tfms = Resize(img_size, method='pad'),
        batch_tfms = [Normalize.from_stats(
            [0.5027, 0.5027, 0.5027],
            [0.2410, 0.2410, 0.2410],
        )],
    )
    return dblock.dataloaders(eval_df, bs=BS_DUMMY,
                              num_workers=0, pin_memory=True)


# ══════════════════════════════════════════════════════════════════════
# 3. Load model weights (returns nn.Module only — no full Learner)
# ══════════════════════════════════════════════════════════════════════
def load_model_weights(weights_path, dls, device):
    """
    Load a UNet model and transfer weights from .pth file.
    Returns nn.Module in eval mode on device.
    """
    learner = unet_learner(dls, resnet50, n_out=N_CLASSES).to_fp16()
    state   = torch.load(weights_path, map_location=device)
    learner.model.load_state_dict(state)
    learner.model.to(device)
    learner.model.eval()
    return learner.model


def find_fold_weights(folds_dir, weight_filename="best_gen.pth"):
    """
    Auto-discover fold weight files.
    Looks for: folds_dir/fold_0/best_gen.pth, fold_1/best_gen.pth, ...
    Falls back to: folds_dir/best_gen.pth if no fold subdirs found.
    """
    folds_dir = pathlib.Path(folds_dir)
    weights   = []

    # Try fold_N subdirectories
    fold_dirs = sorted(folds_dir.glob("fold_*"))
    if fold_dirs:
        for fd in fold_dirs:
            wp = fd / weight_filename
            if wp.exists():
                weights.append(wp)
                print(f"  ✅ Found: {wp}")
            else:
                print(f"  ⚠️  Missing: {wp}")
    else:
        # Fallback: look for weight files directly
        wp = folds_dir / weight_filename
        if wp.exists():
            weights.append(wp)

    return weights


# ══════════════════════════════════════════════════════════════════════
# 4. Ensemble inference
# ══════════════════════════════════════════════════════════════════════
def run_ensemble_inference(models, dls, test_df, device, prob_thresholds):
    """
    For each batch:
      1. Run forward pass through ALL fold models
      2. Average softmax probabilities  ← this is the ensemble step
      3. argmax averaged probs → final prediction

    Returns per-image DataFrame with dice scores and pixel counts.
    """
    n_models   = len(models)
    records    = []
    records_cm = []   # list of (pred_flat, targ_flat) for confusion matrix
    paths      = test_df["image"].tolist()
    bs         = dls.valid.bs
    path_idx   = 0

    n_batches = len(dls.valid)
    with torch.no_grad():
        pbar = tqdm(dls.valid, total=n_batches,
                    desc="Ensemble inference", unit="batch")
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            # ── Ensemble: average softmax probs across all folds ─────
            avg_probs = None
            with torch.amp.autocast('cuda'):
                for model in models:
                    probs = model(xb).softmax(dim=1)   # (B, 4, H, W)
                    if avg_probs is None:
                        avg_probs = probs
                    else:
                        avg_probs = avg_probs + probs
            avg_probs = avg_probs / n_models            # mean across folds
            preds     = avg_probs.argmax(dim=1)         # (B, H, W)
            abdn_prob = avg_probs[:, ABDN_CLASS, :, :]  # (B, H, W)

            # accumulate pixel predictions for confusion matrix
            records_cm.append((
                preds.cpu().numpy().flatten(),
                yb.cpu().numpy().flatten()
            ))

            batch_paths = paths[path_idx:path_idx + len(xb)]
            path_idx   += len(xb)

            # update progress bar with current batch dice
            pbar.set_postfix({"images_done": path_idx})

            for i in range(len(xb)):
                row = {
                    "image":         batch_paths[i] if i < len(batch_paths) else "",
                    "has_abandoned": (yb[i] == ABDN_CLASS).sum().item() > 0,
                }

                # ── Dice per class ──────────────────────────────────
                for cls_idx, cls_name in enumerate(CLASS_NAMES[1:], start=1):
                    p     = (preds[i] == cls_idx).float()
                    t     = (yb[i]   == cls_idx).float()
                    inter = (p * t).sum().item()
                    union = p.sum().item() + t.sum().item()
                    dice  = (2*inter + SMOOTH) / (union + SMOOTH) if union > 0 else float('nan')
                    row[f"dice_{cls_name}"] = round(dice, 4)

                # ── Pixel counts for threshold analysis ────────────
                row["targ_px"] = (yb[i] == ABDN_CLASS).sum().item()
                row["pred_px"] = (preds[i] == ABDN_CLASS).sum().item()
                for pt in prob_thresholds:
                    row[f"prob_{int(pt*100):02d}"] = (abdn_prob[i] > pt).sum().item()

                records.append(row)

    return pd.DataFrame(records), records_cm


# ══════════════════════════════════════════════════════════════════════
# 5. Metrics helpers (identical to eval_test_set.py)
# ══════════════════════════════════════════════════════════════════════
def summarize_dice(df_img, label="Ensemble"):
    print(f"\n{'='*60}")
    print(f"  PIXEL-LEVEL PERFORMANCE — {label}")
    print(f"{'='*60}")
    for cls_name in ["generator", "lead", "abandoned_lead"]:
        col = f"dice_{cls_name}"
        if col not in df_img.columns:
            continue
        vals = df_img[col].dropna()
        print(f"  {cls_name:20s}: mean={vals.mean():.4f}  "
              f"median={vals.median():.4f}  std={vals.std():.4f}  n={len(vals)}")


def compute_pixel_confusion_matrix(records_cm, output_dir, label="Ensemble"):
    """
    Compute and plot pixel-level confusion matrix across all test images.

    Accumulates per-pixel predictions and ground truth across all batches
    (stored in records_cm as list of (pred_flat, targ_flat) numpy arrays).

    Rows = Ground Truth class
    Cols = Predicted class
    Normalised by row (= recall per class)
    """
    from sklearn.metrics import confusion_matrix, classification_report
    import itertools

    all_preds = np.concatenate([r[0] for r in records_cm])
    all_targs = np.concatenate([r[1] for r in records_cm])

    cm = confusion_matrix(all_targs, all_preds, labels=list(range(N_CLASSES)))

    # ── Print raw counts ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PIXEL-LEVEL CONFUSION MATRIX — {label}")
    print(f"{'='*60}")
    print(f"  Rows = Ground Truth   Cols = Predicted")
    print(f"  {'':20s}", end="")
    for name in CLASS_NAMES:
        print(f"  {name[:10]:>10s}", end="")
    print()
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:20s}", end="")
        for j in range(N_CLASSES):
            print(f"  {cm[i,j]:>10,}", end="")
        print()

    # ── Classification report ───────────────────────────────────────
    print(f"\n  Classification Report (pixel-level):")
    report = classification_report(
        all_targs, all_preds,
        labels=list(range(N_CLASSES)),
        target_names=CLASS_NAMES,
        digits=4, zero_division=0
    )
    for line in report.split("\n"):
        print(f"  {line}")

    # ── Plot normalised confusion matrix ────────────────────────────
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm_norm, row_sums,
                         out=np.zeros_like(cm_norm), where=row_sums != 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, title, fmt in [
        (axes[0], cm,      "Raw Counts",          "d"),
        (axes[1], cm_norm, "Normalised (by row)",  ".3f"),
    ]:
        im = ax.imshow(data, interpolation='nearest',
                       cmap='Blues' if fmt == "d" else 'Blues')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Pixel Confusion Matrix\n{title}", fontsize=10)
        ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
        tick_marks = np.arange(N_CLASSES)
        ax.set_xticks(tick_marks); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(tick_marks); ax.set_yticklabels(CLASS_NAMES, fontsize=8)

        thresh = data.max() / 2.0
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                val  = format(data[i,j], fmt) if fmt == "d" else f"{data[i,j]:.3f}"
                color = "white" if data[i,j] > thresh else "black"
                ax.text(j, i, val, ha="center", va="center",
                        color=color, fontsize=7)

    plt.suptitle(f"Pixel-Level Segmentation — {label}", fontsize=12)
    plt.tight_layout()
    out_path = pathlib.Path(output_dir) / "pixel_confusion_matrix.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n📊 Confusion matrix saved → {out_path}")
    plt.close()

    # ── Save raw CM as CSV ──────────────────────────────────────────
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_csv = pathlib.Path(output_dir) / "pixel_confusion_matrix.csv"
    cm_df.to_csv(cm_csv)
    print(f"💾 Confusion matrix CSV → {cm_csv}")

    return cm, cm_norm


def compute_metrics(df_img, pixel_col, pixel_threshold):
    has_pred = df_img[pixel_col] > pixel_threshold
    has_targ = df_img["targ_px"] > 0
    tp = int(( has_pred &  has_targ).sum())
    fp = int(( has_pred & ~has_targ).sum())
    fn = int((~has_pred &  has_targ).sum())
    tn = int((~has_pred & ~has_targ).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1   = 2*sens*ppv / (sens+ppv) if (sens+ppv) > 0 else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "sensitivity": round(sens, 4), "PPV": round(ppv, 4),
            "specificity": round(spec, 4), "F1": round(f1, 4)}


def evaluate_detection(df_img, pixel_thresholds, prob_thresholds):
    rows = []
    for pt in pixel_thresholds:
        m = compute_metrics(df_img, "pred_px", pt)
        rows.append({"prob_threshold": 0.50, "pixel_threshold": pt, **m})
    for prob_t in prob_thresholds:
        if abs(prob_t - 0.5) < 1e-6:
            continue
        col = f"prob_{int(prob_t*100):02d}"
        if col not in df_img.columns:
            continue
        for pt in pixel_thresholds:
            m = compute_metrics(df_img, col, pt)
            rows.append({"prob_threshold": prob_t, "pixel_threshold": pt, **m})
    return pd.DataFrame(rows).sort_values(
        ["prob_threshold", "pixel_threshold"]).reset_index(drop=True)


def print_detection_results(results, label="Ensemble"):
    print(f"\n{'='*60}")
    print(f"  CASE-LEVEL DETECTION — {label}")
    print(f"{'='*60}")
    for prob_t in sorted(results["prob_threshold"].unique()):
        sub   = results[results["prob_threshold"] == prob_t]
        lbl   = "argmax (prob>0.50)" if abs(prob_t-0.5) < 1e-6 else f"prob>{prob_t:.2f}"
        print(f"\n  [{lbl}]")
        print(f"  {'Pixel thr':>10} {'Sens':>7} {'PPV':>7} {'Spec':>7} "
              f"{'F1':>7} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
        print(f"  {'-'*68}")
        for _, row in sub.iterrows():
            print(f"  {int(row['pixel_threshold']):>10,} "
                  f"{row['sensitivity']:>7.3f} {row['PPV']:>7.3f} "
                  f"{row['specificity']:>7.3f} {row['F1']:>7.3f} "
                  f"{int(row['TP']):>4} {int(row['FP']):>4} "
                  f"{int(row['FN']):>4} {int(row['TN']):>4}")

    best = results.loc[results["F1"].idxmax()]
    print(f"\n  ⭐ Best F1: prob={best['prob_threshold']:.2f}  "
          f"pixel={int(best['pixel_threshold']):,}  "
          f"→ Sens={best['sensitivity']:.3f}  PPV={best['PPV']:.3f}  "
          f"F1={best['F1']:.3f}")
    high = results[results["sensitivity"] >= 0.90]
    if len(high) > 0:
        c = high.loc[high["PPV"].idxmax()]
        print(f"  🏥 Clinical (sens≥0.90, best PPV): "
              f"prob={c['prob_threshold']:.2f}  "
              f"pixel={int(c['pixel_threshold']):,}  "
              f"→ Sens={c['sensitivity']:.3f}  PPV={c['PPV']:.3f}")
    else:
        print("  ⚠️  No combination achieved sensitivity ≥ 0.90")


# ══════════════════════════════════════════════════════════════════════
# 6. Visualise ensemble predictions
# ══════════════════════════════════════════════════════════════════════
def _mask_to_rgb(arr):
    rgb = np.zeros((*arr.shape, 3), float)
    for i, c in enumerate(CLASS_COLORS):
        for ch, v in enumerate(c):
            rgb[:, :, ch][arr == i] = v
    return rgb


def plot_samples(models, dls, device, n=6, output_path=None):
    imgs, gts, preds_list = [], [], []
    mean = torch.tensor([0.4945, 0.4945, 0.4945]).view(3,1,1)
    std  = torch.tensor([0.2267, 0.2267, 0.2267]).view(3,1,1)

    with torch.no_grad():
        for xb, yb in dls.valid:
            xb = xb.to(device); yb = yb.to(device)
            avg_probs = None
            with torch.amp.autocast('cuda'):
                for model in models:
                    p = model(xb).softmax(dim=1)
                    avg_probs = p if avg_probs is None else avg_probs + p
            avg_probs /= len(models)
            pred = avg_probs.argmax(dim=1)
            for i in range(len(xb)):
                img_np = (xb[i].cpu() * std + mean).clamp(0,1).permute(1,2,0).numpy()
                imgs.append(img_np)
                gts.append(yb[i].cpu().numpy())
                preds_list.append(pred[i].cpu().numpy())
            if len(imgs) >= n:
                break

    n_show = min(n, len(imgs))
    fig, axes = plt.subplots(n_show, 3, figsize=(12, 4*n_show))
    if n_show == 1: axes = [axes]

    for row, (img, gt, pr) in enumerate(zip(imgs[:n_show], gts[:n_show], preds_list[:n_show])):
        gt_ov   = (0.55*img + 0.45*_mask_to_rgb(gt)).clip(0,1)
        pred_ov = (0.55*img + 0.45*_mask_to_rgb(pr)).clip(0,1)
        axes[row][0].imshow(img);     axes[row][0].set_title("Input",        fontsize=9); axes[row][0].axis("off")
        axes[row][1].imshow(gt_ov);   axes[row][1].set_title("Ground Truth", fontsize=9); axes[row][1].axis("off")
        axes[row][2].imshow(pred_ov); axes[row][2].set_title("Ensemble Prediction", fontsize=9); axes[row][2].axis("off")

    patches = [mpatches.Patch(color=c, label=n)
               for c, n in zip(CLASS_COLORS, CLASS_NAMES)]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Test Set — {len(models)}-Fold Ensemble Predictions",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"📸 Sample predictions saved → {output_path}")
    else:
        plt.show()
    plt.close()


def plot_detection_curves(results, output_path, label="Ensemble"):
    prob_levels = sorted(results["prob_threshold"].unique())
    colors = {0.30: "orange", 0.50: "blue", 0.70: "green", 0.90: "red"}
    styles = {0.30: "--", 0.50: "-", 0.70: "-.", 0.90: ":"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_idx, (metric, ylabel) in enumerate([("sensitivity","Sensitivity"),
                                                ("PPV","PPV (Precision)")]):
        ax = axes[ax_idx]
        for prob_t in prob_levels:
            sub = results[results["prob_threshold"] == prob_t]
            lbl = "argmax (>0.50)" if abs(prob_t-0.5) < 1e-6 else f"prob>{prob_t:.2f}"
            ax.plot(sub["pixel_threshold"], sub[metric],
                    marker="o", color=colors.get(prob_t,"grey"),
                    linestyle=styles.get(prob_t,"-"), linewidth=2, label=lbl)
        if metric == "sensitivity":
            ax.axhline(0.90, color="grey", linestyle=":", linewidth=1, label="0.90")
        ax.set_xlabel("Pixel threshold"); ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
        ax.set_title(f"{ylabel} vs Pixel Threshold")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle(f"Test Set [{label}] — Abandoned Lead Detection Threshold Analysis",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"📈 Detection curves saved → {output_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# 7. Main
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds_dir",
                        default="C:/CIEDID_data/AbdnL/models",
                        help="Directory containing fold_0/, fold_1/, ... subdirs")
    parser.add_argument("--weight_filename", default="best_gen.pth",
                        help="Weight file to use from each fold dir")
    parser.add_argument("--test_imgs",
                        default="C:/CIEDID_data/AbdnL/test_data")
    parser.add_argument("--test_masks",
                        default="C:/CIEDID_data/AbdnL/test_mask")
    parser.add_argument("--img_size",        type=int,   default=640)
    parser.add_argument("--thresholds",      type=int,   nargs="+",
                        default=[100, 250, 500, 1000, 1500, 2000, 2800, 4000])
    parser.add_argument("--prob_thresholds", type=float, nargs="+",
                        default=[0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--n_samples",       type=int,   default=6)
    parser.add_argument("--output_dir",
                        default="C:/CIEDID_data/AbdnL/test_results_ensemble")
    args = parser.parse_args()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"🖥️  Device: {device}")
    print(f"📁 Results → {output_dir}")

    # ── Test set ────────────────────────────────────────────────────
    test_df = build_test_dataframe(
        pathlib.Path(args.test_imgs),
        pathlib.Path(args.test_masks),
    )

    # ── DataLoaders (shared) ────────────────────────────────────────
    dls = build_dls(test_df, args.img_size, device)

    # ── Find and load all fold weights ──────────────────────────────
    print(f"\n🔍 Looking for fold weights in {args.folds_dir} ...")
    weight_paths = find_fold_weights(args.folds_dir, args.weight_filename)
    if len(weight_paths) == 0:
        raise RuntimeError("No fold weights found! Check --folds_dir and --weight_filename")

    print(f"\n📦 Loading {len(weight_paths)} fold models ...")
    models = []
    for wp in tqdm(weight_paths, desc="Loading fold models", unit="model"):
        m = load_model_weights(wp, dls, device)
        models.append(m)
    print(f"✅ {len(models)} models loaded")

    # ── Ensemble inference ──────────────────────────────────────────
    print(f"\n🔀 Running {len(models)}-fold ensemble inference ...")
    df_img, records_cm = run_ensemble_inference(
        models, dls, test_df, device, args.prob_thresholds
    )

    # ── Report ──────────────────────────────────────────────────────
    summarize_dice(df_img, label=f"{len(models)}-Fold Ensemble")

    # ── Pixel-level confusion matrix ────────────────────────────────
    compute_pixel_confusion_matrix(records_cm, output_dir,
                                   label=f"{len(models)}-Fold Ensemble")

    det_results = evaluate_detection(df_img, args.thresholds, args.prob_thresholds)
    print_detection_results(det_results, label=f"{len(models)}-Fold Ensemble")

    # ── Save ────────────────────────────────────────────────────────
    df_img.to_csv(output_dir / "per_image_results.csv", index=False)
    det_results.to_csv(output_dir / "detection_thresholds.csv", index=False)
    print(f"\n💾 Per-image results → {output_dir / 'per_image_results.csv'}")
    print(f"💾 Detection thresholds → {output_dir / 'detection_thresholds.csv'}")

    plot_detection_curves(det_results, output_dir / "detection_curves.png",
                          label=f"{len(models)}-Fold Ensemble")

    if args.n_samples > 0:
        print(f"\n📸 Generating {args.n_samples} sample predictions ...")
        plot_samples(models, dls, device, n=args.n_samples,
                     output_path=output_dir / "sample_predictions.png")
    else:
        print("\n⏭️  Skipping sample predictions (--n_samples 0)")

    print(f"\n🎉 Ensemble evaluation complete → {output_dir}")


if __name__ == "__main__":
    main()