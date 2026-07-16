"""
Probability heatmap visualization for abandoned-lead detection — Figure 4a/4b.

Adapted from heatMapAbdn.py:
  - fixed random seed for reproducibility
  - sorted all images in the test folder for consistent ordering
  - Replaced the full-folder loop (all 50 test images) with an explicit
    filename list, matching the selection pattern already used in
    heatMapAll.py.
  - Two separate figure calls: Figure 4a (false-negative cases) and
    Figure 4b (representative TP/TN/FP cases), each saved to its own file.
  - Panel titles now show ground-truth status alongside the model's
    decision, since Figure 4a/4b are error-analysis figures and a reader
    needs to see predicted vs. actual, not just predicted.
"""

import torch
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from fastai.vision.all import *
from PIL import Image
import torchvision.transforms.functional as TVF


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

# ==========================================================
# 0b. DETERMINISTIC RESIZE + PAD (replaces fastai Resize(method='pad'))
# ==========================================================
# WHY:
#   fastai's Resize(method='pad') is a RandTransform. Its padding position
#   is only guaranteed deterministic (centered) when the transform "knows"
#   it is operating on the valid/test split (split_idx=1), which happens
#   automatically inside a DataLoaders pipeline via ColSplitter/Datasets.
#   Here we call Resize directly inside a standalone Pipeline([...]),
#   so split_idx defaults to None and the transform can behave as if it
#   were on the train split — jittering the padding offset on every call,
#   even with a fixed global seed (the RNG has already been advanced by a
#   different number of draws by the time this call happens).
#   That shifts the image (and therefore the predicted abandoned-lead
#   region) by a few pixels between runs/scripts, changing the pixel
#   count used against the locked prob>0.80 / pixel>600 threshold.
#
# FIX:
#   Implement the resize+pad ourselves with plain PIL/torchvision ops that
#   have no randomness at all: resize preserving aspect ratio so the long
#   side == IMG_Size, then paste onto a zero-filled IMG_Size x IMG_Size
#   canvas, centered. This always produces the identical output for the
#   identical input, regardless of call context or how many other random
#   draws preceded it.
# ==========================================================
def resize_pad_deterministic(img: Image.Image, size: int) -> Image.Image:
    """Aspect-ratio-preserving resize (long side -> size) + centered zero-pad
    to size x size. Deterministic: no dependence on split_idx or RNG state."""
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new(img.mode, (size, size), 0)  # pad_mode='zeros'
    left = (size - new_w) // 2
    top  = (size - new_h) // 2
    canvas.paste(img_resized, (left, top))
    return canvas


def load_and_preprocess(img_path, size: int) -> torch.Tensor:
    """Deterministic replacement for
    Pipeline([PILImage.create, Resize(size, method='pad', pad_mode='zeros'),
              ToTensor(), IntToFloatTensor()])
    Returns a float tensor (3, size, size) in [0, 1], CPU."""
    img = Image.open(img_path).convert("RGB")
    img = resize_pad_deterministic(img, size)
    return TVF.to_tensor(img)  # already float32 in [0, 1], matches IntToFloatTensor output


# ==========================================================
# 1. SETTINGS
# ==========================================================
path_img_folder = "C:/CIEDID_data/AbdnL/test_data"
path_weights    = "C:/CIEDID_data/AbdnL/models/best/best_abdn.pth"
out_dir         = "cied/AbdnL/figures"

IMG_Size    = 640
ABDN_CLASS_IDX = 3
threshold   = 0.8    # softmax probability threshold
pixel_min   = 600    # locked operating threshold (pixel area)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================================
# 2. FIGURE 4a — false-negative cases (ground truth: abandoned lead present)
#    From locked error analysis (A2 in A_model_demographics.txt)
# ==========================================================

# sorted list of all test images (50 total) for reference
img_files = get_image_files(path_img_folder)
img_files_sorted = sorted(img_files)
n_imgs = len(img_files_sorted)

plt.rcParams.update({'font.family': 'inter'})

fig4a_cases = [
    {"filename": "a_x28.png", "label": "False Negative case 1: Single-chamber ICD"},
    {"filename": "a_x5.png",  "label": "False Negative case 2: Leadless Pacemaker"},
]

# ==========================================================
# 3. FIGURE 4b — representative TP / TN / FP cases
#    Fill in the actual chosen filenames once selected.
#    FP candidates available (locked error analysis, A3): 1122, 1417, 1477, 88, 964, 988
# ==========================================================
fig4b_cases = [
    {"filename": "a_x1.png", "label": "1)True positive"},
    {"filename": "1630.png", "label": "2)True negative"},
    {"filename": "988.png", "label": "3)False positive"},
]

# ==========================================================
# 4. LOAD MODEL
# ==========================================================
# NOTE: this `dls` is only used to give unet_learner() the model
# architecture/shape (input channels, n_out, etc.) so we can load the
# state_dict into it. It is NOT used to preprocess the actual inference
# images below — that now goes through load_and_preprocess() (section 0b)
# so the padding is always deterministic regardless of this dls' transforms.
img_files = get_image_files(path_img_folder)
dls = SegmentationDataLoaders.from_label_func(
    pathlib.Path("."), bs=1, fnames=img_files[:1],
    label_func=lambda x: x, codes=["B", "G", "L", "A"],
    item_tfms=Resize(IMG_Size, method='pad')
)
learn = unet_learner(dls, resnet50, n_out=4)
learn.model.load_state_dict(torch.load(path_weights, map_location=device))
learn.model.to(device).eval()

# แก้ไขใน Section 4 ของ heatMapAbdn.py
mean = torch.tensor([0.5150052309036255, 0.5150052309036255, 0.515005230903625], device=device).view(3, 1, 1)
std  = torch.tensor([0.23788487911224365, 0.23788487911224365, 0.23788487911224365], device=device).view(3, 1, 1)


# ==========================================================
# 5. CORE INFERENCE + PLOTTING FUNCTION
# ==========================================================
def run_and_plot(cases, out_path):
    """
    cases: list of dicts with keys "filename", "label", "gt_status"
    Produces an n_rows x 2 figure (raw image | abandoned-lead probability
    heatmap) and saves it to out_path.
    """
    name_to_path = {p.name: p for p in img_files}
    missing = [c["filename"] for c in cases if c["filename"] not in name_to_path]
    if missing:
        raise FileNotFoundError(
            f"Could not find these files in {path_img_folder}: {missing}\n"
            f"Update the filename lists in section 2/3 with the actual "
            f"selected case filenames before running."
        )

    n = len(cases)
    fig, axes = plt.subplots(n, 2, figsize=(11, 5 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, case in enumerate(cases):
        img_path = name_to_path[case["filename"]]

        timg = load_and_preprocess(img_path, IMG_Size).to(device)
        timg_norm = (timg - mean) / std

        with torch.no_grad():
            output = learn.model(timg_norm.unsqueeze(0))
            probs = F.softmax(output, dim=1)[0].cpu()
            abdn_prob_map = probs[ABDN_CLASS_IDX].numpy()

        pixel_count = int((abdn_prob_map > threshold).sum())
        is_detected = pixel_count >= pixel_min

        # --- Column 1: raw image ---
        raw_img = timg.permute(1, 2, 0).cpu().numpy()
        axes[i, 0].imshow(raw_img)
        axes[i, 0].set_title(f"{case['label']}", fontsize=16, fontweight='bold', pad=5)
        axes[i, 0].axis('off')

        # --- Column 2: probability heatmap ---
        im = axes[i, 1].imshow(abdn_prob_map, cmap='jet', vmin=0, vmax=1)
        if abdn_prob_map.max() > threshold:
            axes[i, 1].contour(abdn_prob_map, levels=[threshold], colors='white',
                                linewidths=0.1, linestyles='solid')

        status_text = "Model: DETECTED" if is_detected else "Model: NOT DETECTED"
        box_color = 'firebrick' if is_detected else 'seagreen'
        
        plt.rcParams.update({'font.family': 'inter'})
        """
        if cases == fig4a_cases:
            axes[i, 1].text(
                0.0, 0.98,
                f"{status_text}\nRegions>{threshold} probability: {pixel_count:,}px (threshold: >{pixel_min:,})",
                transform=axes[i, 1].transAxes, ha='left', va='top',
                color='white',  fontweight='bold', fontsize=14,
                bbox=dict(facecolor=box_color, alpha=0.75, edgecolor='none', pad=4)
            )
        if cases == fig4b_cases:
            axes[i, 1].text(
                0.0, 0.1,
                f"{status_text}\nRegions>{threshold} probability: {pixel_count:,}px (threshold: >{pixel_min:,})",
                transform=axes[i, 1].transAxes, ha='left', va='top',
                color='white', fontweight='bold', fontsize=14,
                bbox=dict(facecolor=box_color, alpha=0.75, edgecolor='none', pad=4)
            )
        """
        axes[i, 1].axis('off')
       
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04, label="Model-predicted probability")

    plt.rcParams.update({'font.size': 16, 'font.family': 'inter'})
    #fig.suptitle(suptitle, fontsize=16, y=1.0 if n == 1 else 1.01)
    plt.tight_layout()

    out_file = pathlib.Path(out_dir) / out_path
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved -> {out_file}")


# ==========================================================
# 6. GENERATE FIGURES
# ==========================================================
if __name__ == "__main__":
    run_and_plot(
        fig4a_cases,
        out_path="fig4a_false_negative_cases.png",
        #suptitle="Figure 4a. Abandoned-lead probability heatmaps: False-negative cases"
    )
    run_and_plot(
        fig4b_cases,
        out_path="fig4b_representative_cases.png",
        #suptitle="Figure 4b. Abandoned-lead probability heatmaps: Representative TP / TN / FP cases"
    )