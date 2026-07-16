"""
eval_single.py
==============
Evaluate a single segmentation model.
grid search for abandoned lead detection thresholds (pixel count and probability).
* ALL CONSOLE OUTPUTS ARE AUTOMATICALLY SAVED TO A TEXT FILE *

Usage:
    python eval_single.py --weight_path C:/CIEDID_data/AbdnL/models/best_abdn.pth
"""

import argparse
import pathlib
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from fastai.vision.all import *
from tqdm import tqdm
from skimage import measure

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
# 0. DUAL LOGGER (Save Print to File)
# ══════════════════════════════════════════════════════════════════════
class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  
        if not self.log.closed:         
            self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        if not self.log.closed:         
            self.log.flush()

# ==========================================================
# 0. REPRODUCIBILITY
# ==========================================================
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

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
# 2. Build DataLoaders
# ══════════════════════════════════════════════════════════════════════
def build_dls(test_df, img_size, device):
    def get_x(r): return r["image"]
    def get_y(r): return r["mask"]

    BS_DUMMY = 4
    dummy    = pd.concat([test_df.iloc[[0]]] * BS_DUMMY, ignore_index=True)
    dummy["is_valid"] = False
    eval_df  = pd.concat([dummy, test_df], ignore_index=True)

    # Mean: [0.5037212371826172, 0.5037212371826172, 0.5037212371826172], 
    # Std: [0.24160641431808472, 0.24160641431808472, 0.24160641431808472]
    dblock = DataBlock(
        blocks    = (ImageBlock, MaskBlock(codes=CLASS_NAMES)),
        get_x     = get_x,
        get_y     = get_y,
        splitter  = ColSplitter(col="is_valid"),
        item_tfms = Resize(img_size, method='pad', pad_mode='zeros'),
        batch_tfms = [Normalize.from_stats(
            [0.5150052309036255, 0.5150052309036255, 0.515005230903625],
            [0.23788487911224365, 0.23788487911224365, 0.23788487911224365],
        )],
    )
    return dblock.dataloaders(eval_df, bs=BS_DUMMY,
                              num_workers=0, pin_memory=True)


# ══════════════════════════════════════════════════════════════════════
# 3. Load model weights (returns nn.Module only)
# ══════════════════════════════════════════════════════════════════════
def load_model_weights(weights_path, dls, device):
    learner = unet_learner(dls, resnet50, n_out=N_CLASSES).to_fp16()
    state   = torch.load(weights_path, map_location=device)
    learner.model.load_state_dict(state)
    learner.model.to(device)
    learner.model.eval()
    return learner.model


# ══════════════════════════════════════════════════════════════════════
# 4. Single Model Inference
# ══════════════════════════════════════════════════════════════════════
def run_single_inference(model, dls, test_df, device, prob_thresholds):
    records    = []
    records_cm = []    
    records_gen = []   
    paths      = test_df["image"].tolist()
    path_idx   = 0

    n_batches = len(dls.valid)
    with torch.no_grad():
        pbar = tqdm(dls.valid, total=n_batches,
                    desc="Single model inference", unit="batch")
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            with torch.amp.autocast('cuda'):
                probs = model(xb).softmax(dim=1)        
                
            preds     = probs.argmax(dim=1)             
            abdn_prob = probs[:, ABDN_CLASS, :, :]      

            records_cm.append((
                preds.cpu().numpy().flatten(),
                yb.cpu().numpy().flatten()
            ))

            batch_paths = paths[path_idx:path_idx + len(xb)]
            path_idx   += len(xb)

            pbar.set_postfix({"images_done": path_idx})

            for i in range(len(xb)):
                row = {
                    "image":         batch_paths[i] if i < len(batch_paths) else "",
                    "has_abandoned": (yb[i] == ABDN_CLASS).sum().item() > 0,
                    "has_lead":      (yb[i] == 2).sum().item() > 0,
                }

                for cls_idx, cls_name in enumerate(CLASS_NAMES[1:], start=1):
                    p     = (preds[i] == cls_idx).float()
                    t     = (yb[i]   == cls_idx).float()
                    inter = (p * t).sum().item()
                    union = p.sum().item() + t.sum().item()
                    dice  = (2*inter + SMOOTH) / (union + SMOOTH) if union > 0 else float('nan')
                    row[f"dice_{cls_name}"] = round(dice, 4)

                pred_gen = (preds[i] == 1).cpu().numpy().astype(np.uint8)
                gt_gen   = (yb[i]   == 1).cpu().numpy().astype(np.uint8)
                records_gen.append((pred_gen, gt_gen))

                row["targ_px"] = (yb[i] == ABDN_CLASS).sum().item()
                row["pred_px"] = (preds[i] == ABDN_CLASS).sum().item()
                for pt in prob_thresholds:
                    row[f"prob_{int(pt*100):02d}"] = (abdn_prob[i] > pt).sum().item()

                records.append(row)

    return pd.DataFrame(records), records_cm, records_gen


# ══════════════════════════════════════════════════════════════════════
# 5. Metrics helpers
# ══════════════════════════════════════════════════════════════════════
def summarize_dice(df_img, label="Single Model"):
    print(f"\n{'='*60}")
    print(f"  PIXEL-LEVEL PERFORMANCE — {label}")
    print(f"{'='*60}")
    for cls_name in ["generator", "lead", "abandoned_lead"]:
        col = f"dice_{cls_name}"
        if col not in df_img.columns:
            continue
        vals = df_img[col].dropna()
        q1, q3 = np.percentile(vals, [25, 75]) if len(vals) else (float('nan'), float('nan'))
        print(f"  {cls_name:20s}: mean={vals.mean():.4f}  "
              f"median={vals.median():.4f}  IQR=({q1:.4f}-{q3:.4f})  "
              f"std={vals.std():.4f}  n={len(vals)}")

    # ── report class-present-only subsets for lead and abandoned_lead ──
    # Pixel Dice for a class is 0 by construction whenever that class is
    # absent from the image (nothing to overlap with), so pooling
    # class-present and class-absent cases floors the median/IQR and is
    # not informative about segmentation quality on that class.
    #   - abandoned_lead: majority of cases are class-absent (no abandoned
    #     lead), so this matters a lot.
    #   - lead: only leadless pacemaker cases are class-absent (rare), so
    #     this has minor effect, but is applied for methodological
    #     consistency with abandoned_lead.
    for cls_name, presence_col in [("lead", "has_lead"),
                                    ("abandoned_lead", "has_abandoned")]:
        col = f"dice_{cls_name}"
        if col not in df_img.columns or presence_col not in df_img.columns:
            continue
        tp_vals = df_img.loc[df_img[presence_col] == True, col].dropna()
        if len(tp_vals):
            q1, q3 = np.percentile(tp_vals, [25, 75])
            label_str = f"{cls_name} (class-present only)"
            print(f"  {label_str:28s}: mean={tp_vals.mean():.4f}  "
                  f"median={tp_vals.median():.4f}  IQR=({q1:.4f}-{q3:.4f})  "
                  f"std={tp_vals.std():.4f}  n={len(tp_vals)}")


def compute_pixel_confusion_matrix(records_cm, output_dir, label="Single Model"):
    from sklearn.metrics import confusion_matrix, classification_report
    import itertools

    all_preds = np.concatenate([r[0] for r in records_cm])
    all_targs = np.concatenate([r[1] for r in records_cm])

    cm = confusion_matrix(all_targs, all_preds, labels=list(range(N_CLASSES)))

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

    print(f"\n  Classification Report (pixel-level):")
    report = classification_report(
        all_targs, all_preds,
        labels=list(range(N_CLASSES)),
        target_names=CLASS_NAMES,
        digits=4, zero_division=0
    )
    for line in report.split("\n"):
        print(f"  {line}")

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
    # print(f"\n📊 Confusion matrix saved → {out_path}")
    plt.close()

    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_csv = pathlib.Path(output_dir) / "pixel_confusion_matrix.csv"
    cm_df.to_csv(cm_csv)

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


def print_detection_results(results, label="Single Model"):
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
    high = results[results["sensitivity"] >= 0.95]
    if len(high) > 0:
        c = high.loc[high["PPV"].idxmax()]
        print(f"  🏥 Clinical (sens≥0.95, best PPV): "
              f"prob={c['prob_threshold']:.2f}  "
              f"pixel={int(c['pixel_threshold']):,}  "
              f"→ Sens={c['sensitivity']:.3f}  PPV={c['PPV']:.3f}")
    else:
        print("  ⚠️  No combination achieved sensitivity ≥ 0.95")


# ══════════════════════════════════════════════════════════════════════
# 6. Visualise predictions
# ══════════════════════════════════════════════════════════════════════
def _mask_to_rgb(arr):
    rgb = np.zeros((*arr.shape, 3), float)
    for i, c in enumerate(CLASS_COLORS):
        for ch, v in enumerate(c):
            rgb[:, :, ch][arr == i] = v
    return rgb


def plot_samples(model, dls, device, n=6, output_path=None):
    imgs, gts, preds_list = [], [], []
    mean = torch.tensor([0.4945, 0.4945, 0.4945]).view(3,1,1)
    std  = torch.tensor([0.2267, 0.2267, 0.2267]).view(3,1,1)

    items = dls.valid.items
    if isinstance(items, pd.DataFrame):
        filenames = [pathlib.Path(x).name for x in items['image'].tolist()]
    else:
        filenames = [pathlib.Path(x['image']).name for x in items]

    with torch.no_grad():
        for xb, yb in dls.valid:
            xb = xb.to(device); yb = yb.to(device)
            with torch.amp.autocast('cuda'):
                probs = model(xb).softmax(dim=1)
            pred = probs.argmax(dim=1)
            
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
        fname = filenames[row] if row < len(filenames) else f"Sample_{row}"

        axes[row][0].imshow(img)
        axes[row][0].set_title(f"Input: {fname}", fontsize=8)
        axes[row][0].axis("off")

        #axes[row][0].imshow(img);     axes[row][0].set_title("Input",        fontsize=9); axes[row][0].axis("off")
        axes[row][1].imshow(gt_ov);   axes[row][1].set_title("Ground Truth", fontsize=9); axes[row][1].axis("off")
        axes[row][2].imshow(pred_ov); axes[row][2].set_title("Prediction",   fontsize=9); axes[row][2].axis("off")



    patches = [mpatches.Patch(color=c, label=n)
               for c, n in zip(CLASS_COLORS, CLASS_NAMES)]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Test Set — Single Model Predictions",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_detection_curves(results, output_path, label="Single Model"):
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
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# 7. Generator Bounding Box Coverage
# ══════════════════════════════════════════════════════════════════════
def get_bbox(mask_2d):
    rows = np.any(mask_2d, axis=1)
    cols = np.any(mask_2d, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return int(rmin), int(cmin), int(rmax)+1, int(cmax)+1


def bbox_coverage(gt_bbox, pred_bbox):
    if gt_bbox is None or pred_bbox is None:
        return float('nan')
    gt_rmin, gt_cmin, gt_rmax, gt_cmax     = gt_bbox
    pr_rmin, pr_cmin, pr_rmax, pr_cmax     = pred_bbox
    i_rmin = max(gt_rmin, pr_rmin)
    i_cmin = max(gt_cmin, pr_cmin)
    i_rmax = min(gt_rmax, pr_rmax)
    i_cmax = min(gt_cmax, pr_cmax)
    inter_area = max(0, i_rmax - i_rmin) * max(0, i_cmax - i_cmin)
    gt_area    = (gt_rmax - gt_rmin) * (gt_cmax - gt_cmin)
    return inter_area / gt_area if gt_area > 0 else float('nan')


def bbox_iou(gt_bbox, pred_bbox):
    if gt_bbox is None or pred_bbox is None:
        return float('nan')
    gt_rmin, gt_cmin, gt_rmax, gt_cmax = gt_bbox
    pr_rmin, pr_cmin, pr_rmax, pr_cmax = pred_bbox
    i_rmin = max(gt_rmin, pr_rmin)
    i_cmin = max(gt_cmin, pr_cmin)
    i_rmax = min(gt_rmax, pr_rmax)
    i_cmax = min(gt_cmax, pr_cmax)
    inter = max(0, i_rmax - i_rmin) * max(0, i_cmax - i_cmin)
    gt_a  = (gt_rmax - gt_rmin) * (gt_cmax - gt_cmin)
    pr_a  = (pr_rmax - pr_rmin) * (pr_cmax - pr_cmin)
    union = gt_a + pr_a - inter
    return inter / union if union > 0 else float('nan')


def centroid_distance(gt_mask, pred_mask):
    def centroid(m):
        ys, xs = np.where(m)
        if len(ys) == 0:
            return None
        return float(ys.mean()), float(xs.mean())
    gc = centroid(gt_mask)
    pc = centroid(pred_mask)
    if gc is None or pc is None:
        return float('nan')
    return float(np.sqrt((gc[0]-pc[0])**2 + (gc[1]-pc[1])**2))


def evaluate_generator_crop(df_img: pd.DataFrame,
                             records_gen,
                             output_dir: pathlib.Path,
                             label: str = "Single Model"):
    GEN_CLASS = 1
    rows = []

    for idx, (pred_mask, gt_mask) in enumerate(records_gen):
        gt_bbox   = get_bbox(gt_mask)
        pred_bbox = get_bbox(pred_mask)

        rows.append({
            "image":            df_img["image"].iloc[idx] if idx < len(df_img) else "",
            "gt_detected":      gt_bbox is not None,
            "pred_detected":    pred_bbox is not None,
            "bbox_coverage":    bbox_coverage(gt_bbox, pred_bbox),
            "bbox_iou":         bbox_iou(gt_bbox, pred_bbox),
            "centroid_dist_px": centroid_distance(gt_mask.astype(bool),
                                                  pred_mask.astype(bool)),
        })

    crop_df = pd.DataFrame(rows)

    has_gt   = crop_df["gt_detected"].sum()
    has_pred = crop_df["pred_detected"].sum()
    detected = (crop_df["gt_detected"] & crop_df["pred_detected"]).sum()
    det_rate = detected / has_gt if has_gt > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  GENERATOR CROP EVALUATION — {label}")
    print(f"{'='*60}")
    print(f"  Images with GT generator    : {has_gt}/{len(crop_df)}")
    print(f"  Generator detected (pred)   : {has_pred}/{len(crop_df)}")
    print(f"  Detection rate              : {det_rate:.3f}  ({detected}/{has_gt})")

    valid = crop_df["bbox_coverage"].dropna()
    print("\n  Bounding Box Coverage (GT fully inside predicted region):")
    print(f"  mean={valid.mean():.4f}  median={valid.median():.4f}  "
          f"std={valid.std():.4f}  min={valid.min():.4f}")

    for thr in [0.50, 0.75, 0.90, 1.00]:
        n = (valid >= thr).sum()
        print(f"  Coverage ≥ {thr:.2f}: {n}/{len(valid)} ({100*n/len(valid):.1f}%)")

    iou_vals = crop_df["bbox_iou"].dropna()
    print("\n  Bounding Box IoU:")
    print(f"  mean={iou_vals.mean():.4f}  median={iou_vals.median():.4f}  "
          f"std={iou_vals.std():.4f}")

    dist_vals = crop_df["centroid_dist_px"].dropna()
    print(f"\n  Centroid Distance (pixels @ {label} resolution):")
    print(f"  mean={dist_vals.mean():.1f}  median={dist_vals.median():.1f}  "
          f"std={dist_vals.std():.1f}  max={dist_vals.max():.1f}")

    crop_csv = output_dir / "generator_crop_eval.csv"
    crop_df.to_csv(crop_csv, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].hist(valid, bins=20, color="steelblue", edgecolor="white")
    axes[0].axvline(0.90, color="red", linestyle="--", label="0.90")
    axes[0].set_xlabel("Bounding Box Coverage"); axes[0].set_ylabel("Count")
    axes[0].set_title("Generator BBox Coverage\n(GT inside predicted region)")
    axes[0].legend()

    axes[1].hist(iou_vals, bins=20, color="seagreen", edgecolor="white")
    axes[1].set_xlabel("BBox IoU"); axes[1].set_ylabel("Count")
    axes[1].set_title("Generator BBox IoU")

    axes[2].hist(dist_vals, bins=20, color="darkorange", edgecolor="white")
    axes[2].set_xlabel("Centroid Distance (px)"); axes[2].set_ylabel("Count")
    axes[2].set_title("Generator Centroid Distance")

    plt.suptitle(f"Generator Crop Evaluation — {label}", fontsize=12)
    plt.tight_layout()
    plot_path = output_dir / "generator_crop_eval.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return crop_df


# ══════════════════════════════════════════════════════════════════════
# 8. Main
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_path",
                        default="C:/CIEDID_data/AbdnL/models/best/best_abdn.pth",
                        help="Path to the specific .pth model weight file")
    parser.add_argument("--test_imgs",
                        default="C:/CIEDID_data/AbdnL/test_data")
    parser.add_argument("--test_masks",
                        default="C:/CIEDID_data/AbdnL/test_mask")
    parser.add_argument("--img_size",        type=int,   default=640)
    parser.add_argument("--thresholds",      type=int,   nargs="+",
                        default=[100, 200, 300,400, 500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900, 2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000,4100,4200,4300,4400,4500,4600,4700,4800,4900,5000])
    parser.add_argument("--prob_thresholds", type=float, nargs="+",
                        default=[0.1,0.2,0.3,0.4,0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,1.0])
    parser.add_argument("--n_samples",       type=int,   default=10)
    parser.add_argument("--output_dir",
                        default="C:/CIEDID_data/AbdnL/best/eval_results")
    args = parser.parse_args()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 🔥 เปิดใช้งานระบบบันทึก Log ลงไฟล์
    log_file_path = output_dir / "evaluation_report.txt"
    sys.stdout = DualLogger(log_file_path)

    print(f"🖥️  Device: {device}")
    print(f"📁 Results → {output_dir}")
    print(f"📄 Log File Output → {log_file_path}")

    # ── Test set ────────────────────────────────────────────────────
    test_df = build_test_dataframe(
        pathlib.Path(args.test_imgs),
        pathlib.Path(args.test_masks),
    )

    # ── DataLoaders ─────────────────────────────────────────────────
    dls = build_dls(test_df, args.img_size, device)

    # ── Load model ──────────────────────────────────────────────────
    print(f"\n📦 Loading model from {args.weight_path} ...")
    if not pathlib.Path(args.weight_path).exists():
        raise RuntimeError(f"Weight file not found at: {args.weight_path}")
    
    model = load_model_weights(args.weight_path, dls, device)
    print("✅ Model loaded successfully")

    # ── Single Inference ────────────────────────────────────────────
    print("\n🔀 Running single model inference ...")
    df_img, records_cm, records_gen = run_single_inference(
        model, dls, test_df, device, args.prob_thresholds
    )

    # ── Report ──────────────────────────────────────────────────────
    label = "Single Model"
    summarize_dice(df_img, label=label)

    compute_pixel_confusion_matrix(records_cm, output_dir, label=label)

    evaluate_generator_crop(df_img, records_gen, output_dir, label=label)

    det_results = evaluate_detection(df_img, args.thresholds, args.prob_thresholds)
    print_detection_results(det_results, label=label)

    # ── Save ────────────────────────────────────────────────────────
    df_img.to_csv(output_dir / "per_image_results.csv", index=False)
    det_results.to_csv(output_dir / "detection_thresholds.csv", index=False)
    print(f"\n💾 Per-image results → {output_dir / 'per_image_results.csv'}")
    print(f"💾 Detection thresholds → {output_dir / 'detection_thresholds.csv'}")

    plot_detection_curves(det_results, output_dir / "detection_curves.png", label=label)

    if args.n_samples > 0:
        print(f"\n📸 Generating {args.n_samples} sample predictions ...")
        plot_samples(model, dls, device, n=args.n_samples,
                     output_path=output_dir / "sample_predictions.png")
    else:
        print("\n⏭️  Skipping sample predictions (--n_samples 0)")

    print(f"\n🎉 Evaluation complete → {output_dir}")

    if hasattr(sys.stdout, 'log'):
        sys.stdout.flush()
        sys.stdout.log.close()


if __name__ == "__main__":
    main()