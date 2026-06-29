"""
eval_generator_crop.py
======================
Evaluate generator localisation/cropping performance using post-processed
region extraction, matching the production crop pipeline (testCropNewModel.py)
rather than raw pixel-wise Dice.

Rationale:
  Raw Dice penalises the model for producing smoother, more compact contours
  than the polygon-based human annotation, and is degraded by lead pixels
  competing with the generator class. The actual clinical/production question
  is: "does the final post-processed crop region contain the true generator?"
  — not "does every pixel match the human polygon exactly?"

Post-processing pipeline 
ิby strict order
1. Binary closing (dilate → erode) — merges fragments, smooths edges
2. Fill holes — removes internal gaps (e.g. lead crossing generator)
3. Label components, apply spatial prior filter, keep the largest
4. Convex hull — applied ONLY to the isolated largest component
5. Add border padding, force square aspect, clip to image bounds


Evaluation metrics (computed on the final post-processed region):
  - Detection rate       : fraction of images where a generator region was found
  - BBox coverage         : fraction of GT generator bbox area contained within
                            predicted bbox (most clinically relevant — does the
                            crop capture the whole device?)
  - BBox IoU              : intersection-over-union of GT vs predicted bbox
  - Centroid distance     : pixel distance between GT and predicted region centres
  - Area ratio             : predicted region area / GT region area

Usage:
    python eval_generator_crop.py

Optional:
    --folds_dir       C:/CIEDID_data/AbdnL/models
    --weight_filename best_abdn.pth
    --use_fold        0              # use a single fold instead of ensemble
    --test_imgs       C:/CIEDID_data/AbdnL/test_data
    --test_masks      C:/CIEDID_data/AbdnL/test_mask
    --img_size        640
    --border          0.05           # crop border fraction (matches production)
    --min_area_frac   0.00227         # min area fraction derived from check_generator_area.py to capture small leadless devices while excluding tiny artifacts
"""

import argparse
import pathlib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import scipy.ndimage as ndimage
import skimage.measure
from skimage.morphology import convex_hull_image
import json
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

CLASS_NAMES = ["background", "generator", "lead", "abandoned_lead"]
GEN_CLASS   = 1
N_CLASSES   = len(CLASS_NAMES)


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
        rows.append({"image": str(img.resolve()), "mask": str(mask.resolve()),
                     "is_valid": True})
    df = pd.DataFrame(rows)
    print(f"\n✅ Test set: {len(df)} images")
    return df


# ══════════════════════════════════════════════════════════════════════
# 2. Build DataLoaders
# ══════════════════════════════════════════════════════════════════════
def build_dls(test_df, img_size, device):
    def get_x(r): return r["image"]
    def get_y(r): return r["mask"]

    BS_DUMMY = 4
    dummy = pd.concat([test_df.iloc[[0]]] * BS_DUMMY, ignore_index=True)
    dummy["is_valid"] = False
    eval_df = pd.concat([dummy, test_df], ignore_index=True)

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
# 3. Load model(s)
# ══════════════════════════════════════════════════════════════════════
def load_model_weights(weights_path, dls, device):
    learner = unet_learner(dls, resnet50, n_out=N_CLASSES).to_fp16()
    state   = torch.load(weights_path, map_location=device)
    learner.model.load_state_dict(state)
    learner.model.to(device)
    learner.model.eval()
    return learner.model


def find_fold_weights(folds_dir, weight_filename, use_fold=None):
    """
    If use_fold is given (e.g. 0), return only that fold's weight file
    (single-model evaluation). Otherwise return all available folds
    (ensemble evaluation).
    """
    folds_dir = pathlib.Path(folds_dir)

    if use_fold is not None:
        wp = folds_dir / f"fold_{use_fold}" / weight_filename
        if not wp.exists():
            raise FileNotFoundError(f"Weight not found: {wp}")
        print(f"  ✅ Single-fold mode: {wp}")
        return [wp]

    fold_dirs = sorted(folds_dir.glob("fold_*"))
    weights = []
    if fold_dirs:
        for fd in fold_dirs:
            wp = fd / weight_filename
            if wp.exists():
                weights.append(wp)
                print(f"  ✅ Found: {wp}")
    else:
        wp = folds_dir / weight_filename
        if wp.exists():
            weights.append(wp)
    return weights


# ══════════════════════════════════════════════════════════════════════
# 4. Post-processing pipeline (mirrors testCropNewModel.py)
# ══════════════════════════════════════════════════════════════════════
def postprocess_generator_mask(raw_mask: np.ndarray,
                               img_size: int,
                               min_area_frac: float = 0.00227,
                               dilate_iter: int = 3,
                               erode_iter: int = 3,
                               spatial_prior: dict = None):
    """
    Apply the SAME post-processing used in production cropping:
      1. Binary closing (dilate → erode) — merges fragments, smooths edges
      2. Fill holes — removes internal gaps (e.g. lead crossing generator)
      3. Label components, apply spatial prior filter, keep the largest
         remaining component above min area
      4. Convex hull — applied ONLY to the isolated largest component

    IMPORTANT ORDERING FIX:
      convex_hull_image() must run AFTER component selection, not before.
      If applied to the whole mask first, two separate fragments (e.g. the
      true generator + a stray lead blob elsewhere in the image) get
      "bridged" by the hull into one giant convex region spanning both —
      producing a bounding box that covers most of the image. Isolating
      the largest component first guarantees the hull only wraps the
      generator candidate itself.

    SPATIAL PRIOR:
      Lowering min_area_frac to detect small leadless pacemakers also makes
      the pipeline more susceptible to false positives from small artifacts
      near the image periphery (e.g. text overlays like "AP-UPRIGHT", bone
      overlap at the shoulder/clavicle). Since generators — including
      leadless devices implanted in the RV — are physiologically confined
      to the cardiac silhouette region, components whose centroid falls
      outside an empirically-derived "plausible zone" (from GT centroid
      distribution in the training set) are excluded BEFORE largest-area
      selection. This is a prior on anatomical location, not a hard mask,
      and is intentionally generous (covers >=99% of training centroids)
      so true peripheral-lead generators are not excluded by accident.

    Args:
      spatial_prior: dict with keys 'row_min','row_max','col_min','col_max'
                     as FRACTIONS of image height/width (0-1). A component
                     is kept only if its centroid falls inside this box.
                     If None, no spatial filtering is applied.

    Returns:
      processed_mask : final binary mask (after all steps)
      bbox            : (minr, minc, maxr, maxc) of the largest component, or None
      area            : pixel area of the largest component (post-hull)
    """
    mask = raw_mask.astype(np.uint8)
    if mask.ndim > 2:
        mask = mask.squeeze()

    struct = ndimage.generate_binary_structure(2, 2)

    # ── Step 1: Closing (dilate → erode) ─────────────────────────────
    processed = ndimage.binary_dilation(mask, structure=struct,
                                        iterations=dilate_iter).astype(np.uint8)
    processed = ndimage.binary_erosion(processed, structure=struct,
                                       iterations=erode_iter).astype(np.uint8)

    # ── Step 2: Fill holes ────────────────────────────────────────────
    processed = ndimage.binary_fill_holes(processed).astype(np.uint8)

    # ── Step 3: Label components, apply spatial prior, keep largest ────
    labeled = skimage.measure.label(processed)
    props   = skimage.measure.regionprops(labeled)

    min_area = (img_size * img_size) * min_area_frac
    large_props = [p for p in props if p.area > min_area]

    if spatial_prior is not None:
        h, w = processed.shape
        row_lo = spatial_prior.get("row_min", 0.0) * h
        row_hi = spatial_prior.get("row_max", 1.0) * h
        col_lo = spatial_prior.get("col_min", 0.0) * w
        col_hi = spatial_prior.get("col_max", 1.0) * w

        filtered = []
        for p in large_props:
            cy, cx = p.centroid  # (row, col)
            if row_lo <= cy <= row_hi and col_lo <= cx <= col_hi:
                filtered.append(p)
        large_props = filtered

    if len(large_props) == 0:
        return processed, None, 0

    main_obj   = large_props[np.argmax([p.area for p in large_props])]
    isolated   = (labeled == main_obj.label).astype(np.uint8)

    # ── Step 4: Convex hull — applied ONLY to the isolated component ───
    final_mask = convex_hull_image(isolated).astype(np.uint8)
    bbox       = bbox_from_mask(final_mask)
    area       = int(final_mask.sum())

    return final_mask, bbox, area


def load_spatial_prior(json_path) -> dict:
    """
    Load a pre-computed spatial prior produced by check_generator_area.py.

    The prior defines the anatomically plausible zone for generator
    centroids (as fractions of image height/width), derived from GT
    masks in the train/val set. It is computed ONCE in check_generator_area.py
    rather than here, so that:
      (a) the same prior can be reused consistently across runs/configs,
      (b) it is computed alongside related GT statistics (area thresholds,
          leadless-pacemaker identification) in a single exploratory pass,
      (c) it is clearly visible as a precomputed pipeline hyperparameter,
          not something silently recomputed inside the evaluation loop.

    Returns dict with keys row_min, row_max, col_min, col_max, or None if
    the file doesn't exist (spatial filtering then falls back to disabled).
    """
    json_path = pathlib.Path(json_path)
    if not json_path.exists():
        print(f"  ⚠️  Spatial prior file not found: {json_path}")
        print(f"     Run check_generator_area.py first to generate it, "
              f"or use --no_spatial_prior to disable.")
        return None

    with open(json_path) as f:
        prior = json.load(f)

    print(f"  📍 Spatial prior loaded from {json_path.name}")
    print(f"     (derived from {prior.get('n_masks_used', '?')} GT masks, "
          f"margin={prior.get('margin', '?')})")
    print(f"     row range: {prior['row_min']:.3f}–{prior['row_max']:.3f}")
    print(f"     col range: {prior['col_min']:.3f}–{prior['col_max']:.3f}")

    return {"row_min": prior["row_min"], "row_max": prior["row_max"],
            "col_min": prior["col_min"], "col_max": prior["col_max"]}



def add_border(bbox, img_shape, border_frac=0.05, min_size=160):
    """
    Same border-padding logic as testCropNewModel.py's fix_bbox().
    Expands bbox by border_frac, then forces square aspect, then clips
    to image bounds.
    """
    minr, minc, maxr, maxc = bbox
    dr, dc = int((maxr-minr)*border_frac), int((maxc-minc)*border_frac)
    minr, minc, maxr, maxc = minr-dr, minc-dc, maxr+dr, maxc+dc

    h, w = maxr-minr, maxc-minc
    max_side = max(h, w, min_size)
    dr, dc   = max_side-h, max_side-w
    minr, maxr = max(0, minr - dr//2), min(img_shape[0], maxr + dr//2)
    minc, maxc = max(0, minc - dc//2), min(img_shape[1], maxc + dc//2)

    return int(minr), int(minc), int(maxr), int(maxc)


# ══════════════════════════════════════════════════════════════════════
# 5. Bounding box metrics
# ══════════════════════════════════════════════════════════════════════
def bbox_from_mask(mask: np.ndarray):
    """Get tight bounding box (minr, minc, maxr, maxc) from a binary mask."""
    if mask.sum() == 0:
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    minr, maxr = np.where(rows)[0][[0, -1]]
    minc, maxc = np.where(cols)[0][[0, -1]]
    return int(minr), int(minc), int(maxr)+1, int(maxc)+1


def bbox_coverage(gt_bbox, pred_bbox):
    """
    Fraction of GT bbox area that is contained within predicted bbox.
    This is the most clinically relevant metric: does the crop fully
    capture the device, even if the crop is larger than necessary?
    """
    if gt_bbox is None or pred_bbox is None:
        return 0.0
    gr0, gc0, gr1, gc1 = gt_bbox
    pr0, pc0, pr1, pc1 = pred_bbox

    inter_r0, inter_r1 = max(gr0, pr0), min(gr1, pr1)
    inter_c0, inter_c1 = max(gc0, pc0), min(gc1, pc1)
    inter_h = max(0, inter_r1 - inter_r0)
    inter_w = max(0, inter_c1 - inter_c0)
    inter_area = inter_h * inter_w

    gt_area = (gr1 - gr0) * (gc1 - gc0)
    return inter_area / gt_area if gt_area > 0 else 0.0


def bbox_iou(gt_bbox, pred_bbox):
    if gt_bbox is None or pred_bbox is None:
        return 0.0
    gr0, gc0, gr1, gc1 = gt_bbox
    pr0, pc0, pr1, pc1 = pred_bbox

    inter_r0, inter_r1 = max(gr0, pr0), min(gr1, pr1)
    inter_c0, inter_c1 = max(gc0, pc0), min(gc1, pc1)
    inter_h = max(0, inter_r1 - inter_r0)
    inter_w = max(0, inter_c1 - inter_c0)
    inter_area = inter_h * inter_w

    gt_area   = (gr1 - gr0) * (gc1 - gc0)
    pred_area = (pr1 - pr0) * (pc1 - pc0)
    union     = gt_area + pred_area - inter_area
    return inter_area / union if union > 0 else 0.0


def centroid_distance(gt_bbox, pred_bbox):
    if gt_bbox is None or pred_bbox is None:
        return None
    gr0, gc0, gr1, gc1 = gt_bbox
    pr0, pc0, pr1, pc1 = pred_bbox
    gt_centre   = np.array([(gr0+gr1)/2, (gc0+gc1)/2])
    pred_centre = np.array([(pr0+pr1)/2, (pc0+pc1)/2])
    return float(np.linalg.norm(gt_centre - pred_centre))


# ══════════════════════════════════════════════════════════════════════
# 6. Inference + evaluation
# ══════════════════════════════════════════════════════════════════════
def run_evaluation(models, dls, test_df, device, img_size,
                   border_frac, min_area_frac, spatial_prior=None):
    """
    For each test image:
      1. Get ensemble (or single-model) softmax probability for generator class
      2. argmax → raw predicted mask
      3. Apply post-processing pipeline (close → fill → hull → largest component)
      4. Compute bbox-based metrics against GT (also post-processed the SAME way
         for a fair comparison — GT is a polygon so we just take its tight bbox)
    """
    n_models = len(models)
    records  = []
    paths    = test_df["image"].tolist()
    path_idx = 0

    with torch.no_grad():
        pbar = tqdm(dls.valid, total=len(dls.valid),
                    desc="Generator crop evaluation", unit="batch")
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            avg_probs = None
            with torch.amp.autocast('cuda'):
                for model in models:
                    probs = model(xb).softmax(dim=1)
                    avg_probs = probs if avg_probs is None else avg_probs + probs
            avg_probs = avg_probs / n_models
            preds     = avg_probs.argmax(dim=1)  # (B, H, W)

            batch_paths = paths[path_idx:path_idx+len(xb)]
            path_idx   += len(xb)

            for i in range(len(xb)):
                pred_np = preds[i].cpu().numpy()
                targ_np = yb[i].cpu().numpy()

                raw_pred_mask = (pred_np == GEN_CLASS).astype(np.uint8)
                gt_mask       = (targ_np == GEN_CLASS).astype(np.uint8)

                # ── Post-process PREDICTED mask (production pipeline) ──
                proc_mask, pred_bbox_raw, pred_area = postprocess_generator_mask(
                    raw_pred_mask, img_size, min_area_frac,
                    spatial_prior=spatial_prior
                )
                pred_bbox = add_border(pred_bbox_raw, raw_pred_mask.shape,
                                       border_frac) if pred_bbox_raw else None

                # ── GT: just take tight bbox (no morphology needed — already clean) ──
                gt_bbox = bbox_from_mask(gt_mask)

                row = {
                    "image":          batch_paths[i] if i < len(batch_paths) else "",
                    "gt_present":     gt_mask.sum() > 0,
                    "pred_detected":  pred_bbox is not None,
                    "gt_area":        int(gt_mask.sum()),
                    "pred_area":      int(pred_area),
                    "bbox_coverage":  bbox_coverage(gt_bbox, pred_bbox),
                    "bbox_iou":       bbox_iou(gt_bbox, pred_bbox),
                    "centroid_dist":  centroid_distance(gt_bbox, pred_bbox),
                }
                records.append(row)

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════
# 7. Reporting
# ══════════════════════════════════════════════════════════════════════
def print_summary(df, label="Generator Crop"):
    print(f"\n{'='*60}")
    print(f"  {label.upper()} — POST-PROCESSED REGION EVALUATION")
    print(f"{'='*60}")

    n_gt   = df["gt_present"].sum()
    n_det  = (df["gt_present"] & df["pred_detected"]).sum()
    print(f"  Images with GT generator    : {n_gt}/{len(df)}")
    print(f"  Generator detected (pred)   : {n_det}/{n_gt}")
    print(f"  Detection rate              : {n_det/n_gt:.3f}" if n_gt > 0 else "  N/A")

    detected = df[df["gt_present"] & df["pred_detected"]]

    print(f"\n  Bounding Box Coverage (GT fully inside predicted region):")
    cov = detected["bbox_coverage"]
    print(f"  mean={cov.mean():.3f}  median={cov.median():.3f}  "
          f"  median={cov.median():.3f}  IQR1={cov.quantile(0.25):.3f}  IQR3={cov.quantile(0.75):.3f}  "
          f"std={cov.std():.3f}  min={cov.min():.3f}")
    for thresh in [0.50, 0.75, 0.90, 1.00]:
        n_ok = (cov >= thresh).sum()
        print(f"  Coverage ≥ {thresh:.2f}: {n_ok}/{len(cov)} ({n_ok/len(cov)*100:.1f}%)")

    print(f"\n  Bounding Box IoU:")
    iou = detected["bbox_iou"]
    print(f"  mean={iou.mean():.3f}  median={iou.median():.3f}  std={iou.std():.3f}"
          f"  IQR1={iou.quantile(0.25):.3f}  IQR3={iou.quantile(0.75):.3f}  ")

    print(f"\n  Centroid Distance (pixels @ model resolution):")
    cd = detected["centroid_dist"].dropna()
    print(f"  mean={cd.mean():.1f}  median={cd.median():.1f}  "
          f"  median={cd.median():.1f}  IQR1={cd.quantile(0.25):.1f}  IQR3={cd.quantile(0.75):.1f}  "
          f"std={cd.std():.1f}  max={cd.max():.1f}")

    area_ratio = detected["pred_area"] / detected["gt_area"].replace(0, np.nan)
    print(f"\n  Predicted/GT Area Ratio:")
    print(f"  mean={area_ratio.mean():.3f}  median={area_ratio.median():.3f}  "
          f"(1.0 = same size, >1.0 = over-segmentation, <1.0 = under-segmentation)")


def plot_summary(df, output_path):
    detected = df[df["gt_present"] & df["pred_detected"]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].hist(detected["bbox_coverage"], bins=20, color="steelblue",
                edgecolor="white")
    axes[0].axvline(1.0, color="red", linestyle="--", label="Full coverage")
    axes[0].set_xlabel("BBox Coverage (GT inside prediction)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Generator BBox Coverage\n(GT inside predicted region)")
    axes[0].legend(fontsize=8)

    axes[1].hist(detected["bbox_iou"], bins=20, color="seagreen",
                edgecolor="white")
    axes[1].set_xlabel("BBox IoU")
    axes[1].set_title("Generator BBox IoU")

    axes[2].hist(detected["centroid_dist"].dropna(), bins=20, color="coral",
                edgecolor="white")
    axes[2].set_xlabel("Centroid Distance (pixels)")
    axes[2].set_title("Centroid Distance\n(predicted vs ground truth)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n📊 Summary plot saved → {output_path}")
    plt.close()


def plot_sample_crops(models, dls, test_df, device, img_size,
                      border_frac, min_area_frac, n=6, output_path=None,
                      spatial_prior=None):
    """Show sample predicted bbox vs GT bbox overlaid on the image."""
    mean = torch.tensor([0.5027, 0.5027, 0.5027]).view(3,1,1)
    std  = torch.tensor([0.2410, 0.2410, 0.2410]).view(3,1,1)
    n_models = len(models)

    samples = []
    with torch.no_grad():
        for xb, yb in dls.valid:
            xb = xb.to(device); yb = yb.to(device)
            avg_probs = None
            with torch.amp.autocast('cuda'):
                for model in models:
                    p = model(xb).softmax(dim=1)
                    avg_probs = p if avg_probs is None else avg_probs + p
            avg_probs /= n_models
            preds = avg_probs.argmax(dim=1)

            for i in range(len(xb)):
                img_np = (xb[i].cpu()*std + mean).clamp(0,1).permute(1,2,0).numpy()
                pred_np = preds[i].cpu().numpy()
                targ_np = yb[i].cpu().numpy()

                raw_pred_mask = (pred_np == GEN_CLASS).astype(np.uint8)
                gt_mask       = (targ_np == GEN_CLASS).astype(np.uint8)
                proc_mask, pred_bbox_raw, _ = postprocess_generator_mask(
                    raw_pred_mask, img_size, min_area_frac,
                    spatial_prior=spatial_prior)
                pred_bbox = add_border(pred_bbox_raw, raw_pred_mask.shape,
                                       border_frac) if pred_bbox_raw else None
                gt_bbox = bbox_from_mask(gt_mask)

                samples.append((img_np, gt_bbox, pred_bbox))
            if len(samples) >= n:
                break

    n_show = min(n, len(samples))
    fig, axes = plt.subplots(2, (n_show+1)//2, figsize=(4*((n_show+1)//2), 8))
    axes = axes.flatten() if n_show > 1 else [axes]

    for idx, (img, gt_bbox, pred_bbox) in enumerate(samples[:n_show]):
        ax = axes[idx]
        ax.imshow(img)
        if gt_bbox:
            r0,c0,r1,c1 = gt_bbox
            ax.add_patch(Rectangle((c0,r0), c1-c0, r1-r0,
                                   fill=False, edgecolor='lime', linewidth=2,
                                   label='GT'))
        if pred_bbox:
            r0,c0,r1,c1 = pred_bbox
            ax.add_patch(Rectangle((c0,r0), c1-c0, r1-r0,
                                   fill=False, edgecolor='red', linewidth=2,
                                   linestyle='--', label='Predicted (post-proc)'))
        ax.set_title(f"Sample {idx+1}", fontsize=9)
        ax.axis("off")

    patches = [mpatches.Patch(edgecolor='lime', facecolor='none', label='GT bbox'),
               mpatches.Patch(edgecolor='red',  facecolor='none', label='Predicted bbox (post-processed)')]
    fig.legend(handles=patches, loc="lower center", ncol=2, fontsize=9,
              frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Generator Crop — GT vs Post-Processed Prediction", fontsize=12)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"📸 Sample crops saved → {output_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# 8. Main
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds_dir",
                        default="C:/CIEDID_data/AbdnL/models/best")
    parser.add_argument("--weight_filename", default="best_abdn.pth",
                        help="Use best_gen.pth for crop evaluation "
                             "(generator localisation is what matters here)")
    parser.add_argument("--use_fold", type=int, default=0,
                        help="If set, evaluate a SINGLE fold instead of the "
                             "5-fold ensemble (e.g. --use_fold 0)")
    parser.add_argument("--test_imgs",
                        default="C:/CIEDID_data/AbdnL/test_data")
    parser.add_argument("--test_masks",
                        default="C:/CIEDID_data/AbdnL/test_mask")
    parser.add_argument("--img_size",      type=int,   default=640)
    parser.add_argument("--border",        type=float, default=0.05,
                        help="Crop border fraction (matches production pipeline)")
    parser.add_argument("--min_area_frac", type=float, default=0.00227,
                        help="Min component area as fraction of img_size^2")
    parser.add_argument("--n_samples",     type=int,   default=50)
    parser.add_argument("--output_dir",
                        default="C:/CIEDID_data/AbdnL/test_results_gen_crop")

    # ── Spatial prior ──────────────────────────────────────────────────
    # The prior itself is computed by check_generator_area.py (from
    # train/val GT masks) and saved to spatial_prior.json. This script only
    # loads and applies it — run check_generator_area.py first.
    parser.add_argument("--use_spatial_prior", action="store_true", default=True,
                        help="Filter candidate components by the precomputed "
                             "anatomical location prior before largest-area "
                             "selection (rejects shoulder/text-overlay false "
                             "positives introduced by low min_area_frac). "
                             "Requires spatial_prior.json from "
                             "check_generator_area.py.")
    parser.add_argument("--no_spatial_prior", dest="use_spatial_prior",
                        action="store_false",
                        help="Disable spatial prior filtering.")
    parser.add_argument("--spatial_prior_json",
                        default="C:/CIEDID_data/AbdnL/spatial_prior.json",
                        help="Path to spatial_prior.json produced by "
                             "check_generator_area.py.")
    args = parser.parse_args()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"🖥️  Device: {device}")
    print(f"📁 Results → {output_dir}")

    # ── Load precomputed spatial prior (from check_generator_area.py) ───
    spatial_prior = None
    if args.use_spatial_prior:
        spatial_prior = load_spatial_prior(args.spatial_prior_json)
    else:
        print("\n📍 Spatial prior filtering disabled (--no_spatial_prior)")

    test_df = build_test_dataframe(
        pathlib.Path(args.test_imgs), pathlib.Path(args.test_masks))
    dls = build_dls(test_df, args.img_size, device)

    mode = f"Fold {args.use_fold}" if args.use_fold is not None else "5-Fold Ensemble"
    print(f"\n🔍 Mode: {mode}")
    weight_paths = find_fold_weights(args.folds_dir, args.weight_filename, args.use_fold)
    if len(weight_paths) == 0:
        raise RuntimeError("No weights found!")

    print(f"\n📦 Loading {len(weight_paths)} model(s) ...")
    models = [load_model_weights(wp, dls, device) for wp in tqdm(weight_paths)]
    print(f"✅ {len(models)} model(s) loaded")

    df = run_evaluation(models, dls, test_df, device,
                        args.img_size, args.border, args.min_area_frac,
                        spatial_prior=spatial_prior)

    print_summary(df, label=mode)

    crop_csv = output_dir / "generator_crop_eval.csv"
    df.to_csv(crop_csv, index=False)
    print(f"\n💾 Generator crop evaluation → {crop_csv}")

    plot_summary(df, output_dir / "generator_crop_eval.png")

    if args.n_samples > 0:
        plot_sample_crops(models, dls, test_df, device, args.img_size,
                          args.border, args.min_area_frac, n=args.n_samples,
                          output_path=output_dir / "sample_crops.png",
                          spatial_prior=spatial_prior)


    print(f"\n🎉 Done → {output_dir}")


if __name__ == "__main__":
    main()