# infer_abdnL.py — โหลด seg_abdnL_weights.pth แล้วรัน inference บนภาพใหม่
#
# Usage:
#   python infer_abdnL.py --img C:/path/to/image.png
#   python infer_abdnL.py --img C:/path/to/image.png --weights C:/path/to/seg_abdnL_weights.pth

import pathlib
import platform
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from fastai.vision.all import *

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# ==============================
# CONFIG — ต้องตรงกับที่ train ไว้
# ==============================
CLASS_NAMES  = ["background", "generator", "lead", "abandoned_lead"]
CLASS_COLORS = [
    (0.15, 0.15, 0.15),  # background    — dark grey
    (0.20, 0.60, 1.00),  # generator     — blue
    (0.20, 0.85, 0.45),  # lead          — green
    (1.00, 0.25, 0.25),  # abandoned_lead — red
]
IMG_SIZE = 512  # ต้องตรงกับ Resize ที่ใช้ตอน train


# ==============================
# LOAD MODEL
# ==============================
def load_model(weights_path: str, device=None):
    """
    สร้าง UNet architecture เหมือนตอน train แล้ว load weights กลับมา
    คืน model ที่พร้อม inference (.eval() แล้ว)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # สร้าง dummy dls เพื่อให้ unet_learner build architecture ได้
    # (ไม่ได้ใช้ train จริง — แค่ต้องการ model structure)
    path = pathlib.Path("/tmp/_dummy_infer")
    (path / "imgs").mkdir(parents=True, exist_ok=True)
    (path / "masks").mkdir(parents=True, exist_ok=True)
    from PIL import Image
    dummy_img  = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    dummy_mask = Image.fromarray(np.zeros((64, 64),    dtype=np.uint8))
    dummy_img.save(path  / "imgs"  / "d.png")
    dummy_mask.save(path / "masks" / "d.png")

    dblock = DataBlock(
        blocks=(ImageBlock, MaskBlock(codes=CLASS_NAMES)),
        get_items=get_image_files,
        get_y=lambda x: path / "masks" / x.name,
        splitter=RandomSplitter(0.5, seed=0),
        item_tfms=Resize(64),
    )
    dls = dblock.dataloaders(path / "imgs", bs=1, num_workers=0, verbose=False)

    learn = unet_learner(dls, resnet50, n_out=len(CLASS_NAMES))
    state = torch.load(weights_path, map_location=device)
    learn.model.load_state_dict(state)
    learn.model.to(device).eval()
    print(f"✅ Model loaded from {weights_path}  (device={device})")
    return learn.model, device


# ==============================
# PREPROCESS
# ==============================
def preprocess(img_path: str, size: int = IMG_SIZE):
    """โหลดภาพ → resize → normalize → tensor (1, 3, H, W)"""
    img = PILImage.create(img_path)
    img = img.resize((size, size))
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    arr  = (arr - mean) / std
    return torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).float()


# ==============================
# INFERENCE
# ==============================
@torch.no_grad()
def predict(model, img_tensor, device):
    """คืน predicted class mask (H, W) และ softmax prob map (C, H, W)"""
    img_tensor = img_tensor.to(device)
    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        logits = model(img_tensor)          # (1, C, H, W)
    probs  = torch.softmax(logits, dim=1)[0].cpu()   # (C, H, W)
    pred   = probs.argmax(dim=0).numpy()             # (H, W)
    return pred, probs.numpy()


# ==============================
# VISUALISE
# ==============================
def visualise(img_path, pred_mask, probs, save_path=None):
    img_orig = np.array(PILImage.create(img_path).convert("RGB"))
    h, w     = pred_mask.shape

    # Mask overlay
    overlay_rgb = np.zeros((h, w, 3), dtype=float)
    for c, col in enumerate(CLASS_COLORS):
        for ch, v in enumerate(col):
            overlay_rgb[:, :, ch][pred_mask == c] = v

    img_resized = np.array(
        PILImage.create(img_path).convert("RGB").resize((w, h))
    ).astype(float) / 255.0
    overlay = (0.55 * img_resized + 0.45 * overlay_rgb).clip(0, 1)

    # Pixel stats
    total_px  = pred_mask.size
    px_counts = [(pred_mask == c).sum() for c in range(len(CLASS_NAMES))]
    pcts      = [100 * c / total_px for c in px_counts]

    # Abandoned lead detection
    abdn_px  = px_counts[3]
    detected = abdn_px > 20
    status   = f"DETECTED ({abdn_px:,} px)" if detected else f"not detected ({abdn_px} px)"

    fig, axes = plt.subplots(1, 2 + len(CLASS_NAMES),
                             figsize=(4 * (2 + len(CLASS_NAMES)), 4))
    fig.suptitle(f"{pathlib.Path(img_path).name}   |   Abandoned lead: {status}",
                 fontsize=11)

    axes[0].imshow(img_orig); axes[0].set_title("original"); axes[0].axis("off")
    axes[1].imshow(overlay);  axes[1].set_title("predicted mask"); axes[1].axis("off")

    # Legend on overlay
    patches = [mpatches.Patch(color=c, label=f"{n} ({pcts[i]:.1f}%)")
               for i, (c, n) in enumerate(zip(CLASS_COLORS, CLASS_NAMES))
               if pcts[i] > 0.01]
    axes[1].legend(handles=patches, loc="lower left", fontsize=7, framealpha=0.7)

    # Per-class prob maps
    for c in range(len(CLASS_NAMES)):
        ax = axes[2 + c]
        im = ax.imshow(probs[c], cmap="hot", vmin=0, vmax=1)
        ax.set_title(f"prob: {CLASS_NAMES[c]}", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"📊 Saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)

    return {"abandoned_lead_detected": detected, "pixel_counts": dict(zip(CLASS_NAMES, px_counts))}


# ==============================
# ENTRY POINT
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img",     required=True,
                        help="path to input X-ray image")
    parser.add_argument("--weights", default="C:/CIEDID_data/AbdnL/models/seg_abdnL_weights.pth",
                        help="path to seg_abdnL_weights.pth")
    parser.add_argument("--out",     default=None,
                        help="save result image to this path (optional)")
    args = parser.parse_args()
    IMG_PATH     = r"C:/CIEDID_data/AbdnL/data/a_20.png"
    WEIGHTS_PATH = r"C:/CIEDID_data/AbdnL/models/seg_abdnL_weights.pth"
    SAVE_PATH    = None   # หรือใส่ path ถ้าอยากบันทึกภาพผล

    model, device = load_model(args.weights)
    img_tensor    = preprocess(args.img)
    pred, probs   = predict(model, img_tensor, device)
    result        = visualise(args.img, pred, probs, save_path=args.out)

    print("\n--- Detection result ---")
    for cls, px in result["pixel_counts"].items():
        pct = 100 * px / pred.size
        print(f"  {cls:20s}: {px:>8,} px  ({pct:.2f}%)")
    print(f"\n  Abandoned lead: {'✅ YES' if result['abandoned_lead_detected'] else '❌ NO'}")