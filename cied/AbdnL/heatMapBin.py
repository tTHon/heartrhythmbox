"""
Heatmap visualization — BINARY MASK version (solid color, no gradient)
=========================================================================
Unlike heatMapAll.py (which shows a continuous probability gradient via
alpha blending), this version shows a flat, solid-color mask: a pixel is
either "in" a class or not, matching how a reader would interpret a
segmentation result in a manuscript figure.

Class assignment logic:
    - Generator / Normal Lead: each pixel is assigned to whichever class
      the model considers most probable (the standard segmentation
      output) — this matches the production mask used for Stage II
      generator cropping.
    - Abandoned Lead: flagged only where the model's predicted
      probability for this class exceeds the pre-specified operating
      threshold, since case-level abandoned-lead detection uses a
      decision rule tuned independently of the raw segmentation output
      (see prior discussion on pred_px vs prob_XX).
    - Pixels meeting the abandoned-lead threshold are drawn with
      highest priority (drawn last / on top) so overlapping "most
      likely = lead" pixels are shown as abandoned lead, consistent
      with the clinical decision rather than the raw segmentation mask.

Usage: edit SETTINGS below, then run.
"""

import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from fastai.vision.all import *
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

# ==========================================================
# 1. SETTINGS & PARAMETERS
# ==========================================================
path_img_folder = "C:/CIEDID_data/AbdnL/test_data"
path_weights    = "C:/CIEDID_data/AbdnL/models/best_abdn.pth"
IMG_Size   = 640
threshold  = 0.8     # abandoned-lead probability threshold
pixel_min  = 600     # abandoned-lead pixel-area threshold

MASK_ALPHA = 0.5     # flat opacity for ALL classes (no gradient)

COLOR_GENERATOR = np.array([0, 1, 0])      # green
COLOR_LEAD      = np.array([0, 0.6, 1])    # blue
COLOR_ABDN      = np.array([1, 0, 0])      # red

# ระบุชื่อไฟล์ภาพที่ต้องการทำ heatmap
selected_filenames = [
    "1626.png",
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================================
# 2. LOAD MODEL
# ==========================================================
img_files = get_image_files(path_img_folder)
dls = SegmentationDataLoaders.from_label_func(
    pathlib.Path("."), bs=1, fnames=img_files[:1],
    label_func=lambda x: x, codes=["Background", "Generator", "Lead", "Abdn_Lead"],
    item_tfms=Resize(IMG_Size, method='pad')
)
learn = unet_learner(dls, resnet50, n_out=4)
learn.model.load_state_dict(torch.load(path_weights, map_location=device))
learn.model.to(device).eval()

timg_pipe = Pipeline([PILImage.create, Resize(IMG_Size, method='pad', pad_mode='zeros'),
                      ToTensor(), IntToFloatTensor()])
mean = torch.tensor([0.5150052309036255] * 3, device=device).view(3, 1, 1)
std  = torch.tensor([0.23788487911224365] * 3, device=device).view(3, 1, 1)

# ==========================================================
# 3. SELECT IMAGES
# ==========================================================
selected_imgs = [p for p in img_files if p.name in selected_filenames]
missing = set(selected_filenames) - {p.name for p in selected_imgs}
if missing:
    raise FileNotFoundError(
        f"ไม่พบไฟล์เหล่านี้ใน {path_img_folder}: {missing}\n"
        f"เช็คว่าพิมพ์ชื่อไฟล์ถูกต้อง (รวมนามสกุล .png/.jpg) และมีไฟล์อยู่จริงในโฟลเดอร์"
    )

fig, axes = plt.subplots(1, len(selected_imgs), figsize=(20, 8))
if len(selected_imgs) == 1:
    axes = [axes]

# ==========================================================
# 4. INFERENCE + BINARY MASK OVERLAY
# ==========================================================
for i, img_path in enumerate(selected_imgs):
    timg = timg_pipe(img_path).to(device)
    timg_norm = (timg - mean) / std

    with torch.no_grad():
        output = learn.model(timg_norm.unsqueeze(0))
        probs = F.softmax(output, dim=1)[0].cpu().numpy()

    abdn_prob = probs[3]
    most_likely_class = probs.argmax(axis=0)   # 0=bg, 1=generator, 2=lead, 3=abdn (per-pixel most-probable class)

    # Case-level decision (same rule as production pipeline)
    pixel_count = (abdn_prob > threshold).sum()
    detected = pixel_count > pixel_min

    # Binary class masks (flat, no gradient)
    abdn_mask_bool = abdn_prob > threshold                       # explicit clinical threshold
    generator_mask = (most_likely_class == 1) & (~abdn_mask_bool)
    lead_mask      = (most_likely_class == 2) & (~abdn_mask_bool)
    # abdn_mask_bool drawn last / on top -> takes visual priority over overlap

    raw_img = timg.permute(1, 2, 0).cpu().numpy()
    axes[i].imshow(raw_img)

    # Build one solid RGBA overlay per class and draw in priority order:
    # generator -> lead -> abandoned lead (top)
    for mask, color in [(generator_mask, COLOR_GENERATOR),
                         (lead_mask, COLOR_LEAD),
                         (abdn_mask_bool, COLOR_ABDN)]:
        rgba = np.zeros((*mask.shape, 4))
        rgba[..., :3] = color
        rgba[..., 3] = np.where(mask, MASK_ALPHA, 0.0)   # flat alpha, no gradient
        axes[i].imshow(rgba)

    status_text = "FOUND" if detected else "NOT FOUND"
    res_color = 'red' if detected else 'gray'
    axes[i].set_title(f"{img_path.name}\nAbdn Lead: {status_text} ({pixel_count:,} px)",
                       fontsize=10, color=res_color, fontweight='bold')
    axes[i].axis('off')

# ==========================================================
# 5. LEGEND
# ==========================================================
custom_lines = [Line2D([0], [0], color=COLOR_GENERATOR, lw=4),
                Line2D([0], [0], color=COLOR_LEAD, lw=4),
                Line2D([0], [0], color=COLOR_ABDN, lw=4)]
fig.legend(custom_lines,
           ['Generator', 'Normal Lead',
            f'Abandoned Lead (probability > {threshold})'],
           loc='lower center', ncol=3, fontsize=9)

fig.text(0.5, 0.1,
         "Solid overlay indicates predicted class membership: generator and normal lead "
         "show the model's most likely class at each pixel; abandoned lead is shown only "
         "where the calibrated probability threshold was exceeded.",
         ha='center', fontsize=8.5, style='italic', color='dimgray')

plt.tight_layout(rect=[0, 0.07, 1, 0.95])
plt.show()