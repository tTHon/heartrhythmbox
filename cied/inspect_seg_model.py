"""
Inspect Segmentation Model Mask Classes
----------------------------------------
รัน segmentation.pkl กับรูปภาพที่มี แล้วดูว่า mask มี class อะไรบ้าง

Usage:
    python inspect_seg_model.py --model cied/segmentation.pkl --image cied/Dataset/A3.png
"""

import sys
import pathlib
import platform
import argparse
import numpy as np

# ==========================================================
# COMPATIBILITY PATCHES (same as basic.py)
# ==========================================================
if not hasattr(np, 'int'):
    np.int = int

import fastai.callback.fp16
if not hasattr(fastai.callback.fp16, 'AMPMode'):
    class AMPMode:
        def __init__(self, *args, **kwargs): pass
    fastai.callback.fp16.AMPMode = AMPMode
    sys.modules['fastai.callback.fp16'].AMPMode = AMPMode

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

from fastai.vision.all import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ==========================================================
# CLASS NAME GUESSES (แก้ได้หลังจากรู้ผล)
# ==========================================================
CLASS_NAMES = {
    0: "Background",
    1: "Device Generator",
    2: "Lead / Other",
    3: "Class 3",
}

CLASS_COLORS = {
    0: (30,  30,  30),
    1: (255, 80,  80),
    2: (80,  180, 255),
    3: (80,  255, 130),
}

# ==========================================================
# MAIN
# ==========================================================
def inspect(model_path: str, image_path: str):
    model_path = pathlib.Path(model_path)
    image_path = pathlib.Path(image_path)

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}"); sys.exit(1)
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}"); sys.exit(1)

    # ── Load segmentation model ───────────────────────────
    print(f"Loading segmentation model: {model_path.name}  (อาจใช้เวลาสักครู่...)")
    learn = load_learner(model_path, cpu=True)
    print("Model loaded OK")
    print(f"  vocab (classes): {learn.dls.vocab}")

    # ── Run prediction ────────────────────────────────────
    print(f"\nRunning segmentation on: {image_path.name}")
    res = learn.predict(str(image_path))
    # res[0] = decoded mask (TensorMask), res[1] = class index mask, res[2] = softmax probs

    mask = np.array(res[1])   # pixel-level class indices
    print(f"  Mask shape : {mask.shape}")
    print(f"  Mask dtype : {mask.dtype}")

    # ── Unique classes in this prediction ─────────────────
    unique_vals, counts = np.unique(mask, return_counts=True)
    total = mask.size

    print(f"\n{'Class':>6}  {'Label':<22}  {'Pixels':>9}  {'%':>6}")
    print("-" * 50)
    for val, count in zip(unique_vals, counts):
        label = CLASS_NAMES.get(int(val), f"Unknown {val}")
        try:
            vocab_label = learn.dls.vocab[int(val)]
            label = f"{label}  [{vocab_label}]"
        except Exception:
            pass
        pct = count / total * 100
        print(f"{val:>6}  {label:<22}  {count:>9,}  {pct:>5.2f}%")

    # ── Visualize ─────────────────────────────────────────
    orig = np.array(PILImage.create(image_path).resize((mask.shape[1], mask.shape[0])))

    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for val in unique_vals:
        color_mask[mask == val] = CLASS_COLORS.get(int(val), (200, 200, 200))

    # Overlay: blend original + color mask
    if orig.ndim == 2:
        orig_rgb = np.stack([orig]*3, axis=-1)
    else:
        orig_rgb = orig[:, :, :3]
    overlay = (0.5 * orig_rgb + 0.5 * color_mask).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Segmentation Mask Inspector — {image_path.name}", fontsize=13, fontweight="bold")

    axes[0].imshow(orig_rgb, cmap="gray")
    axes[0].set_title("Original Image"); axes[0].axis("off")

    axes[1].imshow(color_mask)
    axes[1].set_title("Predicted Mask (color-coded)"); axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay"); axes[2].axis("off")

    # Legend
    patches = []
    for val in unique_vals:
        c = tuple(x/255 for x in CLASS_COLORS.get(int(val), (200,200,200)))
        label = CLASS_NAMES.get(int(val), f"Unknown {val}")
        count = counts[list(unique_vals).index(val)]
        patches.append(mpatches.Patch(color=c, label=f"Class {val}: {label} ({count/total*100:.1f}%)"))
    axes[2].legend(handles=patches, loc="lower left", fontsize=9, framealpha=0.85)

    plt.tight_layout()
    out = image_path.parent / f"seg_inspect_{image_path.stem}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n[Saved] → {out}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cied/segmentation.pkl",
                        help="Path to segmentation.pkl")
    parser.add_argument("--image", default="cied/Dataset/A3.png",
                        help="Path to any chest X-ray image")
    args = parser.parse_args()
    inspect(args.model, args.image)
