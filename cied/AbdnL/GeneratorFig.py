"""
fig_generator_pipeline.py
==========================
Figure 2/3 (generator localization) — 5-panel pipeline visualization for a
single representative case, left to right:

  1. Ground truth generator mask (overlaid on image) + solid GT bbox outline
     (added for visibility — the translucent mask alone was hard to read
     against bone/soft tissue)
  2. Raw model prediction (argmax, generator class) — shows lead interference
     / fragmentation before any post-processing
  3. Post-processed mask (close -> fill holes -> largest component ->
     convex hull), BEFORE border padding / forced-square step
  4. Final bounding box (after add_border: padding + forced square + clip),
     drawn on the original image alongside the GT bbox for direct comparison
     (green solid = GT, red dashed = final prediction, yellow dotted =
     pre-square prediction)
  5. Final cropped output (image cropped to the bounding box from panel 4)

Reuses the exact post-processing functions from eval_generator.py
(postprocess_generator_mask, add_border, bbox_from_mask) so the figure
matches the locked evaluation pipeline exactly — do not reimplement these
independently, or the figure may silently drift from the actual results.

*** NORMALIZATION STATS — CONFIRMED ***
mean=0.5150052309036255, std=0.23788487911224365 (all channels). This is
the actual training-time dataset statistic (finetuneCV.py,
get_segmentation_stats(), labeled "Final Dataset") and matches build_dls()
in eval_generator.py, which performs real model inference. The alternate
value (0.5027/0.2410) previously seen in eval_generator.py's
plot_sample_crops() was a stale display-only value (cosmetic bug in that
one debug plot) and is NOT used here.
"""

import pathlib
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.ndimage as ndimage
import skimage.measure
from skimage.morphology import convex_hull_image
from fastai.vision.all import *

# ── PyTorch 2.6+ fix ────────────────────────────────────────────────
_orig_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault('weights_only', False)
    return _orig_load(*a, **kw)
torch.load = _patched_load

# ==========================================================
# 1. SETTINGS — fill in the chosen representative case
# ==========================================================
path_img_folder = "C:/CIEDID_data/AbdnL/test_data"
path_mask_folder = "C:/CIEDID_data/AbdnL/test_mask"
path_weights     = "C:/CIEDID_data/AbdnL/models/best/best_abdn.pth"
out_path         = "cied/AbdnL/figures/fig_generator_pipeline.png"

#SELECTED_FILENAME = "1570.png"   # <-- pick the representative case
#SELECTED_FILENAME = "1609.png"  #scICD
SELECTED_FILENAME = "944.png"   # leadless


IMG_Size      = 640
GEN_CLASS     = 1
CLASS_NAMES   = ["background", "generator", "lead", "abandoned_lead"]
min_area_frac = 0.00227
border_frac   = 0.05

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# *** confirmed correct value — see module docstring ***
MEAN = torch.tensor([0.5150052309036255, 0.5150052309036255, 0.5150052309036255]).view(3, 1, 1)
STD  = torch.tensor([0.23788487911224365, 0.23788487911224365, 0.23788487911224365]).view(3, 1, 1)


# ==========================================================
# 2. POST-PROCESSING (copied verbatim from eval_generator.py to
#    guarantee identical behavior — do not edit independently)
# ==========================================================
def bbox_from_mask(mask: np.ndarray):
    if mask.sum() == 0:
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    minr, maxr = np.where(rows)[0][[0, -1]]
    minc, maxc = np.where(cols)[0][[0, -1]]
    return int(minr), int(minc), int(maxr) + 1, int(maxc) + 1


def postprocess_generator_mask(raw_mask, img_size, min_area_frac=0.00227,
                               dilate_iter=3, erode_iter=3, spatial_prior=None):
    mask = raw_mask.astype(np.uint8)
    if mask.ndim > 2:
        mask = mask.squeeze()

    struct = ndimage.generate_binary_structure(2, 2)

    processed = ndimage.binary_dilation(mask, structure=struct,
                                        iterations=dilate_iter).astype(np.uint8)
    processed = ndimage.binary_erosion(processed, structure=struct,
                                       iterations=erode_iter).astype(np.uint8)
    processed = ndimage.binary_fill_holes(processed).astype(np.uint8)

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
        large_props = [p for p in large_props
                      if row_lo <= p.centroid[0] <= row_hi
                      and col_lo <= p.centroid[1] <= col_hi]

    if len(large_props) == 0:
        return processed, None, 0

    main_obj = large_props[np.argmax([p.area for p in large_props])]
    isolated = (labeled == main_obj.label).astype(np.uint8)

    final_mask = convex_hull_image(isolated).astype(np.uint8)
    bbox       = bbox_from_mask(final_mask)
    area       = int(final_mask.sum())
    return final_mask, bbox, area


def add_border(bbox, img_shape, border_frac=0.05, min_size=160):
    minr, minc, maxr, maxc = bbox
    dr, dc = int((maxr - minr) * border_frac), int((maxc - minc) * border_frac)
    minr, minc, maxr, maxc = minr - dr, minc - dc, maxr + dr, maxc + dc

    h, w = maxr - minr, maxc - minc
    max_side = max(h, w, min_size)
    dr, dc = max_side - h, max_side - w
    minr, maxr = max(0, minr - dr // 2), min(img_shape[0], maxr + dr // 2)
    minc, maxc = max(0, minc - dc // 2), min(img_shape[1], maxc + dc // 2)
    return int(minr), int(minc), int(maxr), int(maxc)


# ==========================================================
# 3. LOAD MODEL
# ==========================================================
img_files = get_image_files(path_img_folder)
target_path = next((p for p in img_files if p.name == SELECTED_FILENAME), None)
if target_path is None:
    raise FileNotFoundError(
        f"'{SELECTED_FILENAME}' not found in {path_img_folder}. "
        f"Set SELECTED_FILENAME to an actual test-set image."
    )
mask_path = pathlib.Path(path_mask_folder) / f"{target_path.stem}_mask.png"
if not mask_path.exists():
    mask_path = pathlib.Path(path_mask_folder) / f"{target_path.stem}.png"

dls = SegmentationDataLoaders.from_label_func(
    pathlib.Path("."), bs=1, fnames=img_files[:1],
    label_func=lambda x: x, codes=CLASS_NAMES,
    item_tfms=Resize(IMG_Size, method='squash', pad_mode='zeros')
)
learn = unet_learner(dls, resnet50, n_out=len(CLASS_NAMES))
learn.model.load_state_dict(torch.load(path_weights, map_location=device))
learn.model.to(device).eval()

timg_pipe = Pipeline([PILImage.create, Resize(IMG_Size, method='squash', pad_mode='zeros'),
                      ToTensor(), IntToFloatTensor()])
# NOTE: masks must use PILMask.create, NOT PILImage.create.
# PILImage.create loads the label PNG as RGB (H,W,3) and Resize applies
# bilinear interpolation by default, which blends label values at edges —
# this silently produced a 3-channel, interpolation-corrupted "mask" whose
# bbox_from_mask() (written for a 2D array) then returned a wildly oversized
# box. PILMask.create keeps it single-channel (H,W) and fastai dispatches
# Resize to nearest-neighbor for PILMask, preserving exact integer labels.
mask_pipe = Pipeline([PILMask.create, Resize(IMG_Size, method='squash', pad_mode='zeros')])

# ==========================================================
# 4. INFERENCE
# ==========================================================
timg = timg_pipe(target_path).to(device)
timg_norm = (timg - MEAN.to(device)) / STD.to(device)

with torch.no_grad():
    output = learn.model(timg_norm.unsqueeze(0))
    pred = output.argmax(dim=1)[0].cpu().numpy()

raw_pred_mask = (pred == GEN_CLASS).astype(np.uint8)

gt_mask_img = np.array(mask_pipe(mask_path))
gt_mask = (gt_mask_img == GEN_CLASS).astype(np.uint8)

# GT bounding box — tight box around the ground-truth generator mask.
# Drawn as a solid box (Panel 1 + Panel 4) so it reads clearly even when
# the translucent mask overlay is hard to see against bone/soft tissue.
gt_bbox = bbox_from_mask(gt_mask)

# Panel 3: post-processed mask, BEFORE border padding / forced square
proc_mask, tight_bbox, _ = postprocess_generator_mask(
    raw_pred_mask, IMG_Size, min_area_frac=min_area_frac, spatial_prior=None)

# Panel 4: final bbox AFTER border padding + forced square + clip
final_bbox = add_border(tight_bbox, raw_pred_mask.shape, border_frac) if tight_bbox else None

#img_np = (timg.cpu() * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
img_np = timg.cpu().clamp(0, 1).permute(1, 2, 0).numpy()

# Panel 5: cropped output
if final_bbox is not None:
    r0, c0, r1, c1 = final_bbox
    cropped_img = img_np[r0:r1, c0:c1]
else:
    cropped_img = img_np  # fallback, should not occur if detection succeeded


# ==========================================================
# 5. PLOT — 5 panels, left to right
# ==========================================================
plt.rcParams.update({'font.size': 16, 'font.family': 'inter'})
fig, axes = plt.subplots(1, 5, figsize=(22, 5))

# Panel 1 — Ground truth (mask overlay + solid bbox outline for clarity)
axes[0].imshow(img_np)
axes[0].imshow(np.ma.masked_where(gt_mask == 0, gt_mask), cmap='Greens', alpha=0.45, vmin=0, vmax=1)
#if gt_bbox is not None:
#    r0, c0, r1, c1 = gt_bbox
#    axes[0].add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0,
#                                fill=False, edgecolor='lime', linewidth=2.2))
axes[0].set_title("1. Ground truth", fontsize=14)
axes[0].axis('off')

# Panel 2 — Raw prediction (pre-post-processing, shows lead interference/fragmentation)
axes[1].imshow(img_np)
axes[1].imshow(np.ma.masked_where(raw_pred_mask == 0, raw_pred_mask), cmap='Blues', alpha=0.45, vmin=0, vmax=1)
axes[1].set_title("2. Raw prediction", fontsize=14)
axes[1].axis('off')

# Panel 3 — Post-processed mask, before forced-square border step
axes[2].imshow(img_np)
axes[2].imshow(np.ma.masked_where(proc_mask == 0, proc_mask), cmap='Oranges', alpha=0.45, vmin=0, vmax=1)
axes[2].set_title("3. Post-processing", fontsize=14)
#axes[2].set_title("3. Post-processed\n(close\u2192fill\u2192largest component\u2192hull)", fontsize=14)
axes[2].axis('off')

# Panel 4 — Final bounding box (padded + forced square) on original image,
# now shown against the GT bbox (solid green) so the reader can judge
# localization accuracy directly, matching the GT-vs-pred convention used
# elsewhere (solid green = GT, dashed red = prediction).
axes[3].imshow(img_np)
if gt_bbox is not None:
    r0, c0, r1, c1 = gt_bbox
    axes[3].add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0,
                                fill=False, edgecolor='lime', linewidth=2.2))
if final_bbox is not None:
    r0, c0, r1, c1 = final_bbox
    axes[3].add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0,
                                fill=False, edgecolor='red', linewidth=2,
                                linestyle='--'))
#if tight_bbox is not None:
#    r0, c0, r1, c1 = tight_bbox
#    axes[3].add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0,
#                                fill=False, edgecolor='yellow', linewidth=1.1,
#                                linestyle=':'))
axes[3].set_title("4. Bounding box\n(green = GT, red dashed = final)",
                  fontsize=14)
axes[3].axis('off')

# Panel 5 — Cropped output
axes[4].imshow(cropped_img)
axes[4].set_title("5. Final crop", fontsize=14)
axes[4].axis('off')

#fig.suptitle(f"Generator localization pipeline — {target_path.name}", fontsize=13, y=1.03)
plt.tight_layout()

out_file = pathlib.Path(out_path)
out_file.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved -> {out_file}")