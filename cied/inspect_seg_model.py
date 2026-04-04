"""
Inspect Segmentation Model Mask Classes (fixed)
------------------------------------------------
Usage:
    python inspect_seg_model.py --model cied/segmentation.pkl --image cied/Dataset/A3.png
"""

import sys
import pathlib
import platform
import argparse
import numpy as np

# ==========================================================
# COMPATIBILITY PATCHES
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

CLASS_NAMES = {
    0: "Background",
    1: "Class 1",
    2: "Class 2",
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

    # ── Load model ────────────────────────────────────────
    print(f"Loading: {model_path.name}  (อาจใช้เวลาสักครู่...)")
    learn = load_learner(model_path, cpu=True)
    print("Model loaded OK\n")

    # ── Try to find vocab / class info from multiple places ──
    print("=== Model Info ===")
    print(f"  Model type : {type(learn.model).__name__}")
    print(f"  n_out      : {getattr(learn.model, 'n_out', 'N/A')}")

    for attr in ['vocab', 'classes', 'c']:
        for obj in [learn, learn.dls]:
            try:
                val = getattr(obj, attr)
                print(f"  {type(obj).__name__}.{attr} = {val}")
            except Exception:
                pass

    # ── Run prediction ────────────────────────────────────
    print(f"\nRunning segmentation on: {image_path.name}")
    res = learn.predict(str(image_path))

    print(f"\n=== Raw predict() output ===")
    print(f"  len(res) = {len(res)}")
    for i, r in enumerate(res):
        try:
            arr = np.array(r)
            print(f"  res[{i}]: type={type(r).__name__}  shape={arr.shape}  dtype={arr.dtype}  "
                  f"min={arr.min():.4f}  max={arr.max():.4f}  unique={np.unique(arr)[:10]}")
        except Exception as e:
            print(f"  res[{i}]: {type(r).__name__}  (could not convert: {e})")

    # ── Use res[1] as the class-index mask ────────────────
    mask = np.array(res[1])
    if mask.ndim > 2:
        mask = mask[0]

    print(f"\n=== Mask Analysis ===")
    print(f"  Shape  : {mask.shape}")
    print(f"  Dtype  : {mask.dtype}")
    unique_vals, counts = np.unique(mask, return_counts=True)
    total = mask.size

    print(f"\n{'Class':>6}  {'Label':<20}  {'Pixels':>10}  {'%':>6}")
    print("-" * 48)
    for val, count in zip(unique_vals, counts):
        label = CLASS_NAMES.get(int(val), f"Unknown {val}")
        pct = count / total * 100
        print(f"{val:>6}  {label:<20}  {count:>10,}  {pct:>5.2f}%")

    # ── Softmax probabilities (res[2]) ────────────────────
    probs = np.array(res[2])
    print(f"\n=== Probability Map (res[2]) ===")
    print(f"  Shape      : {probs.shape}  ← first dim = number of classes")
    print(f"  n_classes  : {probs.shape[0]}")
    for c in range(probs.shape[0]):
        label = CLASS_NAMES.get(c, f"Class {c}")
        p = probs[c]
        print(f"  Channel {c} ({label}):  mean prob = {p.mean():.4f}  max = {p.max():.4f}")

    # ── Visualize ─────────────────────────────────────────
    orig = np.array(PILImage.create(image_path).resize((mask.shape[1], mask.shape[0])))
    if orig.ndim == 2:
        orig_rgb = np.stack([orig]*3, axis=-1)
    else:
        orig_rgb = orig[:, :, :3]

    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for val in unique_vals:
        color_mask[mask == val] = CLASS_COLORS.get(int(val), (200, 200, 200))

    overlay = (0.5 * orig_rgb + 0.5 * color_mask).astype(np.uint8)

    ncols = 2 + probs.shape[0]
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    fig.suptitle(f"Seg Inspector — {image_path.name}", fontsize=12, fontweight="bold")

    axes[0].imshow(orig_rgb, cmap="gray")
    axes[0].set_title("Original"); axes[0].axis("off")

    axes[1].imshow(color_mask)
    axes[1].set_title("Predicted Mask"); axes[1].axis("off")
    patches = [mpatches.Patch(
        color=tuple(x/255 for x in CLASS_COLORS.get(int(v), (200,200,200))),
        label=f"Class {v}: {CLASS_NAMES.get(int(v), '?')} ({counts[i]/total*100:.1f}%)"
    ) for i, v in enumerate(unique_vals)]
    axes[1].legend(handles=patches, loc="lower left", fontsize=8, framealpha=0.85)

    for c in range(probs.shape[0]):
        ax = axes[2 + c]
        im = ax.imshow(probs[c], cmap="hot", vmin=0, vmax=1)
        ax.set_title(f"Prob: {CLASS_NAMES.get(c, f'Class {c}')}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out = image_path.parent / f"seg_inspect_{image_path.stem}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n[Saved] → {out}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cied/segmentation.pkl")
    parser.add_argument("--image", default="cied/Dataset/Micra full.png")
    args = parser.parse_args()
    inspect(args.model, args.image)