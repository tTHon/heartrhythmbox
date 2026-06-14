"""
eval_test_set.py
================
Evaluate final segmentation model on held-out test set.
Reports:
  - Dice coefficient per class (generator, lead, abandoned_lead)
  - Abandoned lead detection: Sensitivity, PPV, Specificity, F1
    at multiple pixel threshold × probability threshold combinations
  - Per-image results CSV
  - Summary figures

Usage:
    python eval_test_set.py

Optional args:
    --weights      path to best_gen.pth or best_abdn.pth
    --test_imgs    test image directory
    --test_masks   test mask directory
    --img_size     512
    --thresholds   100 250 500 1000 1500 2000 2800 4000
    --prob_thresholds  0.3 0.5 0.7 0.9
    --output_dir   results/test_eval
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


# ══════════════════════════════════════════════════════════════════════
# 1. Build test dataframe
# ══════════════════════════════════════════════════════════════════════
def build_test_dataframe(img_dir: pathlib.Path,
                         mask_dir: pathlib.Path) -> pd.DataFrame:
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
            "is_valid":      True,   # all test rows are "val" for FastAI
        })

    df = pd.DataFrame(rows)
    print(f"\n✅ Test set: {len(df)} images "
          f"({df['has_abandoned'].sum()} with abandoned lead  |  "
          f"{(~df['has_abandoned']).sum()} without)")
    return df


# ══════════════════════════════════════════════════════════════════════
# 2. Load model
# ══════════════════════════════════════════════════════════════════════
def load_model(weights_path: pathlib.Path,
               img_size: int,
               test_df: pd.DataFrame,
               device: torch.device) -> Learner:
    def get_x(r): return r["image"]
    def get_y(r): return r["mask"]

    # FastAI needs ≥1 train batch — add dummy rows
    BS_DUMMY = 4
    dummy = pd.concat([test_df.iloc[[0]]] * BS_DUMMY, ignore_index=True)
    dummy["is_valid"] = False
    eval_df = pd.concat([dummy, test_df], ignore_index=True)

    dblock = DataBlock(
        blocks    = (ImageBlock, MaskBlock(codes=CLASS_NAMES)),
        get_x     = get_x,
        get_y     = get_y,
        splitter  = ColSplitter(col="is_valid"),
        item_tfms = Resize(img_size, method='pad'),
        batch_tfms = [Normalize.from_stats(
            [0.4945, 0.4945, 0.4945],
            [0.2267, 0.2267, 0.2267],
        )],
    )
    dls = dblock.dataloaders(eval_df, bs=BS_DUMMY,
                             num_workers=0, pin_memory=True)

    learner = unet_learner(dls, resnet50, n_out=4).to_fp16()
    state   = torch.load(weights_path, map_location=device)
    learner.model.load_state_dict(state)
    learner.model.to(device)
    learner.model.eval()
    print(f"✅ Weights loaded: {weights_path.name}")
    return learner


# ══════════════════════════════════════════════════════════════════════
# 3. Run inference — collect per-image metrics
# ══════════════════════════════════════════════════════════════════════
def run_inference(learner: Learner,
                  test_df: pd.DataFrame,
                  device: torch.device,
                  prob_thresholds: list):
    """
    Returns per-image DataFrame with:
      - dice_generator, dice_lead, dice_abdn  (pixel-level overlap)
      - targ_px, pred_px                      (argmax pixel counts)
      - prob_XX                               (pixel counts at each prob thr)
      - image, has_abandoned
    """
    SMOOTH = 1e-6
    records = []

    with torch.no_grad():
        for (xb, yb), img_paths in zip(
                learner.dls.valid,
                _batch_paths(test_df, learner.dls.valid)):

            xb = xb.to(device)
            yb = yb.to(device)

            with torch.cuda.amp.autocast():
                logits = learner.model(xb)
                probs  = logits.softmax(dim=1)   # (B, 4, H, W)
                preds  = logits.argmax(dim=1)    # (B, H, W)

            abdn_prob = probs[:, ABDN_CLASS, :, :]

            for i in range(len(xb)):
                row = {"image": img_paths[i],
                       "has_abandoned": (yb[i] == ABDN_CLASS).sum().item() > 0}

                # ── Dice per class ──────────────────────────────────
                for cls_idx, cls_name in enumerate(CLASS_NAMES[1:], start=1):
                    p = (preds[i] == cls_idx).float()
                    t = (yb[i]   == cls_idx).float()
                    inter = (p * t).sum().item()
                    union = p.sum().item() + t.sum().item()
                    dice  = (2*inter + SMOOTH) / (union + SMOOTH) if union > 0 else float('nan')
                    row[f"dice_{cls_name}"] = round(dice, 4)

                # ── Pixel counts ────────────────────────────────────
                row["targ_px"] = (yb[i] == ABDN_CLASS).sum().item()
                row["pred_px"] = (preds[i] == ABDN_CLASS).sum().item()  # argmax
                for pt in prob_thresholds:
                    row[f"prob_{int(pt*100):02d}"] = (abdn_prob[i] > pt).sum().item()

                records.append(row)

    return pd.DataFrame(records)


def _batch_paths(test_df, dl):
    """Yield image paths in the same order as DataLoader batches."""
    paths = test_df["image"].tolist()
    bs    = dl.bs
    for start in range(0, len(paths), bs):
        yield paths[start:start+bs]


# ══════════════════════════════════════════════════════════════════════
# 4. Dice summary
# ══════════════════════════════════════════════════════════════════════
def summarize_dice(df_img: pd.DataFrame):
    print(f"\n{'='*55}")
    print(f"  PIXEL-LEVEL PERFORMANCE (Dice coefficient)")
    print(f"{'='*55}")
    for cls_name in ["generator", "lead", "abandoned_lead"]:
        col = f"dice_{cls_name}"
        if col not in df_img.columns:
            continue
        vals = df_img[col].dropna()
        print(f"  {cls_name:20s}: "
              f"mean={vals.mean():.4f}  "
              f"median={vals.median():.4f}  "
              f"std={vals.std():.4f}  "
              f"n={len(vals)}")


# ══════════════════════════════════════════════════════════════════════
# 5. Detection metrics (same logic as eval_abdn_threshold.py)
# ══════════════════════════════════════════════════════════════════════
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
            "sensitivity": round(sens, 4),
            "PPV":         round(ppv,  4),
            "specificity": round(spec, 4),
            "F1":          round(f1,   4)}


def evaluate_detection(df_img, pixel_thresholds, prob_thresholds):
    rows = []
    # argmax (prob > 0.5)
    for pt in pixel_thresholds:
        m = compute_metrics(df_img, "pred_px", pt)
        rows.append({"prob_threshold": 0.50, "pixel_threshold": pt, **m})
    # explicit prob thresholds
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


def print_detection_results(results):
    print(f"\n{'='*55}")
    print(f"  CASE-LEVEL DETECTION (Abandoned Lead)")
    print(f"{'='*55}")
    for prob_t in sorted(results["prob_threshold"].unique()):
        sub   = results[results["prob_threshold"] == prob_t]
        label = "argmax (prob>0.50)" if abs(prob_t-0.5) < 1e-6 else f"prob>{prob_t:.2f}"
        print(f"\n  [{label}]")
        print(f"  {'Pixel thr':>10} {'Sens':>7} {'PPV':>7} "
              f"{'Spec':>7} {'F1':>7} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
        print(f"  {'-'*68}")
        for _, row in sub.iterrows():
            print(f"  {int(row['pixel_threshold']):>10,} "
                  f"{row['sensitivity']:>7.3f} "
                  f"{row['PPV']:>7.3f} "
                  f"{row['specificity']:>7.3f} "
                  f"{row['F1']:>7.3f} "
                  f"{int(row['TP']):>4} "
                  f"{int(row['FP']):>4} "
                  f"{int(row['FN']):>4} "
                  f"{int(row['TN']):>4}")

    best = results.loc[results["F1"].idxmax()]
    print(f"\n  ⭐ Best F1: prob={best['prob_threshold']:.2f}  "
          f"pixel={int(best['pixel_threshold']):,}  "
          f"→ Sens={best['sensitivity']:.3f}  "
          f"PPV={best['PPV']:.3f}  F1={best['F1']:.3f}")

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
# 6. Visualise sample predictions
# ══════════════════════════════════════════════════════════════════════
def _mask_to_rgb(arr):
    rgb = np.zeros((*arr.shape, 3), float)
    for i, c in enumerate(CLASS_COLORS):
        for ch, v in enumerate(c):
            rgb[:, :, ch][arr == i] = v
    return rgb


def plot_samples(learner, test_df, device, n=6, output_path=None):
    """Show n sample predictions side-by-side with ground truth."""
    imgs, gts, preds_list = [], [], []
    mean = torch.tensor([0.4945, 0.4945, 0.4945]).view(3,1,1)
    std  = torch.tensor([0.2267, 0.2267, 0.2267]).view(3,1,1)

    with torch.no_grad():
        for xb, yb in learner.dls.valid:
            xb = xb.to(device); yb = yb.to(device)
            with torch.cuda.amp.autocast():
                pred = learner.model(xb).argmax(dim=1)
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
        axes[row][2].imshow(pred_ov); axes[row][2].set_title("Prediction",   fontsize=9); axes[row][2].axis("off")

    patches = [mpatches.Patch(color=c, label=n)
               for c, n in zip(CLASS_COLORS, CLASS_NAMES)]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Test Set — Sample Predictions", fontsize=12, y=1.01)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"📸 Sample predictions saved → {output_path}")
    else:
        plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# 7. Plot detection curves
# ══════════════════════════════════════════════════════════════════════
def plot_detection_curves(results, output_path):
    prob_levels = sorted(results["prob_threshold"].unique())
    colors = {0.30: "orange", 0.50: "blue", 0.70: "green", 0.90: "red"}
    styles = {0.30: "--",     0.50: "-",    0.70: "-.",    0.90: ":"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (metric, ylabel) in enumerate([("sensitivity", "Sensitivity"),
                                                ("PPV",         "PPV (Precision)")]):
        ax = axes[ax_idx]
        for prob_t in prob_levels:
            sub   = results[results["prob_threshold"] == prob_t]
            label = "argmax (>0.50)" if abs(prob_t-0.5) < 1e-6 else f"prob>{prob_t:.2f}"
            ax.plot(sub["pixel_threshold"], sub[metric],
                    marker="o", color=colors.get(prob_t,"grey"),
                    linestyle=styles.get(prob_t,"-"), linewidth=2, label=label)
        if metric == "sensitivity":
            ax.axhline(0.90, color="grey", linestyle=":", linewidth=1, label="0.90")
        ax.set_xlabel("Pixel threshold")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.set_title(f"{ylabel} vs Pixel Threshold")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle("Test Set — Abandoned Lead Detection Threshold Analysis",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"📈 Detection curves saved → {output_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# 8. Main
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",
                        default="C:/CIEDID_data/AbdnL/models/fold1.pth")
    parser.add_argument("--test_imgs",
                        default="C:/CIEDID_data/AbdnL/test_data")
    parser.add_argument("--test_masks",
                        default="C:/CIEDID_data/AbdnL/test_mask")
    parser.add_argument("--img_size",        type=int,   default=512)
    parser.add_argument("--thresholds",      type=int,   nargs="+",
                        default=[100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250,2500, 2750, 3000, 3250, 3500, 3750, 4000])
    parser.add_argument("--prob_thresholds", type=float, nargs="+",
                        default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--n_samples",       type=int,   default=6,
                        help="Number of sample predictions to visualise")
    parser.add_argument("--output_dir",
                        default="C:/CIEDID_data/AbdnL/test_results")
    args = parser.parse_args()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"🖥️  Device: {device}")
    print(f"📁 Results will be saved to: {output_dir}")

    # ── Build test dataframe ────────────────────────────────────────
    test_df = build_test_dataframe(
        pathlib.Path(args.test_imgs),
        pathlib.Path(args.test_masks),
    )

    # ── Load model ──────────────────────────────────────────────────
    learner = load_model(
        pathlib.Path(args.weights),
        args.img_size, test_df, device
    )

    # ── Inference ───────────────────────────────────────────────────
    print("\n🔍 Running inference on test set …")
    df_img = run_inference(learner, test_df, device, args.prob_thresholds)

    # ── Dice summary ────────────────────────────────────────────────
    summarize_dice(df_img)

    # ── Detection metrics ───────────────────────────────────────────
    det_results = evaluate_detection(df_img, args.thresholds,
                                     args.prob_thresholds)
    print_detection_results(det_results)

    # ── Save per-image results ──────────────────────────────────────
    per_image_csv = output_dir / "per_image_results.csv"
    df_img.to_csv(per_image_csv, index=False)
    print(f"\n💾 Per-image results → {per_image_csv}")

    det_csv = output_dir / "detection_thresholds.csv"
    det_results.to_csv(det_csv, index=False)
    print(f"💾 Detection thresholds → {det_csv}")

    # ── Plots ───────────────────────────────────────────────────────
    plot_detection_curves(det_results,
                          output_dir / "detection_curves.png")
    plot_samples(learner, test_df, device,
                 n=args.n_samples,
                 output_path=output_dir / "sample_predictions.png")

    print(f"\n🎉 Test set evaluation complete → {output_dir}")


if __name__ == "__main__":
    main()