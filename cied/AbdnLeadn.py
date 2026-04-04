"""
Finetune Segmentation Model — Add Abandoned Lead Class
=======================================================
ต่อยอดจาก segmentation.pkl (Zhukov) เพิ่ม class ใหม่:
    0 = Background
    1 = Device Generator  (เดิม)
    2 = Active Lead       (annotate ใหม่)
    3 = Abandoned Lead    (annotate ใหม่)  ← เป้าหมาย

FOLDER STRUCTURE ที่ต้องเตรียม:
    cied/
    ├── segmentation.pkl          ← โมเดลเดิม
    ├── finetune_data/
    │   ├── images/               ← รูป X-ray ที่ annotate แล้ว
    │   │   ├── case001.jpg
    │   │   └── ...
    │   └── masks/                ← mask PNG จาก ls_to_mask.py
    │       ├── case001_mask.png
    │       └── ...

USAGE:
    python finetune_seg.py
    python finetune_seg.py --data_dir cied/finetune_data --epochs_head 5 --epochs_full 20
"""

import sys
import pathlib
import platform
import argparse
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

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

import torch
import pandas as pd
from fastai.vision.all import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ==========================================================
# CONFIGURATION — แก้ตรงนี้
# ==========================================================
MODEL_PATH   = pathlib.Path("cied/segmentation.pkl")   # โมเดลเดิม
DATA_DIR     = pathlib.Path("cied/finetune_data")       # folder ข้อมูลใหม่
IMAGE_DIR    = DATA_DIR / "images"                      # รูปต้นฉบับ
MASK_DIR     = DATA_DIR / "masks"                       # mask PNG
OUTPUT_DIR   = pathlib.Path("cied/models")              # save โมเดลใหม่

NEW_EXP_NAME = "seg_finetune_4cls"                      # ชื่อไฟล์โมเดลใหม่
N_OUT_NEW    = 4                                        # จำนวน class ใหม่
VALID_SPLIT  = 0.2                                      # 20% validation
BATCH_SIZE   = 8                                        # ลดถ้า OOM (ลองใช้ 4)
IMG_SIZE     = 1024                                     # resize ก่อน training
PATCH_SIZE   = 256                                      # random crop ระหว่าง train
SEED         = 42

CLASS_NAMES = ["background", "generator", "lead", "abandoned_lead"]

# ==========================================================
# STEP 1 — Build DataFrame จาก image/mask folder
# ==========================================================
def build_dataframe(image_dir, mask_dir, valid_split=0.2, seed=42):
    image_dir = pathlib.Path(image_dir)
    mask_dir  = pathlib.Path(mask_dir)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in exts])

    rows = []
    missing_masks = []
    for img_path in image_files:
        mask_path = mask_dir / f"{img_path.stem}_mask.png"
        if not mask_path.exists():
            missing_masks.append(img_path.name)
            continue
        rows.append({"image": str(img_path), "mask": str(mask_path)})

    if missing_masks:
        print(f"[WARN] {len(missing_masks)} image(s) have no matching mask — skipped:")
        for m in missing_masks[:5]:
            print(f"       {m}")
        if len(missing_masks) > 5:
            print(f"       ... and {len(missing_masks)-5} more")

    df = pd.DataFrame(rows)
    np.random.seed(seed)
    is_valid = np.random.rand(len(df)) < valid_split
    df["is_valid"] = is_valid

    print(f"\nDataset: {len(df)} images  "
          f"({(~is_valid).sum()} train / {is_valid.sum()} valid)")
    return df


# ==========================================================
# STEP 2 — Load old weights into new 4-class model
# ==========================================================
def load_pretrained_weights(learner_new, old_model_path):
    """
    Load weights from old 3-class model into new 4-class model.
    Layers with shape mismatch (final conv) are skipped → retrained from scratch.
    """
    print(f"\nLoading old weights from: {old_model_path}")
    old_learn = load_learner(old_model_path, cpu=True)
    old_state = old_learn.model.state_dict()
    new_state = learner_new.model.state_dict()

    loaded, skipped = [], []
    for k, v in old_state.items():
        if k in new_state and new_state[k].shape == v.shape:
            new_state[k] = v
            loaded.append(k)
        else:
            skipped.append(f"{k}  old={v.shape}  new={new_state.get(k, torch.tensor([])).shape}")

    learner_new.model.load_state_dict(new_state)

    print(f"  Loaded : {len(loaded)} layers")
    print(f"  Skipped (shape mismatch — will train from scratch):")
    for s in skipped:
        print(f"    {s}")

    del old_learn
    return learner_new


# ==========================================================
# STEP 3 — Validate masks (check class values)
# ==========================================================
def validate_masks(df, expected_classes={0,1,2,3}):
    print("\nValidating masks...")
    found_classes = set()
    for mask_path in df["mask"]:
        mask = np.array(PILImage.create(mask_path))
        found_classes |= set(np.unique(mask).tolist())

    print(f"  Class values found across all masks: {sorted(found_classes)}")
    missing = expected_classes - found_classes
    if missing:
        print(f"  [WARN] These classes have NO pixels in any mask: {missing}")
        print(f"         (ปกติถ้า abandoned lead ยังไม่มีในข้อมูล)")
    else:
        print(f"  All {len(expected_classes)} classes found ✓")
    return found_classes


# ==========================================================
# STEP 4 — Plot training history
# ==========================================================
def plot_history(recorder, save_path):
    losses = recorder.losses
    val_losses = [v[0] for v in recorder.values]
    dice_scores = [v[1] for v in recorder.values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(losses, label="Train Loss", alpha=0.7)
    ax1.plot(val_losses, label="Valid Loss", marker="o")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(True)

    ax2.plot(dice_scores, label="Dice (valid)", marker="o", color="green")
    ax2.set_title("Dice Score (Validation)"); ax2.legend(); ax2.grid(True)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"[Saved] Training plot → {save_path}")
    plt.show()


# ==========================================================
# STEP 5 — Preview predictions
# ==========================================================
def preview_predictions(learner, df, n=3):
    CLASS_COLORS = {
        0: (30,  30,  30),
        1: (255, 80,  80),
        2: (80,  180, 255),
        3: (80,  255, 130),
    }
    sample = df[df["is_valid"]].sample(min(n, df["is_valid"].sum()), random_state=SEED)

    fig, axes = plt.subplots(len(sample), 3, figsize=(12, 4 * len(sample)))
    if len(sample) == 1:
        axes = [axes]

    for row_idx, (_, row) in enumerate(sample.iterrows()):
        img_path  = row["image"]
        mask_path = row["mask"]

        res  = learner.predict(img_path)
        pred = np.array(res[1])
        gt   = np.array(PILImage.create(mask_path))
        orig = np.array(PILImage.create(img_path).convert("RGB"))

        def to_color(mask):
            h, w = mask.shape
            c = np.zeros((h, w, 3), dtype=np.uint8)
            for v in np.unique(mask):
                c[mask == v] = CLASS_COLORS.get(int(v), (200,200,200))
            return c

        axes[row_idx][0].imshow(orig, cmap="gray")
        axes[row_idx][0].set_title(f"Original\n{pathlib.Path(img_path).name}")
        axes[row_idx][0].axis("off")

        axes[row_idx][1].imshow(to_color(gt))
        axes[row_idx][1].set_title("Ground Truth Mask")
        axes[row_idx][1].axis("off")

        axes[row_idx][2].imshow(to_color(pred))
        axes[row_idx][2].set_title("Predicted Mask")
        axes[row_idx][2].axis("off")

    patches = [mpatches.Patch(
        color=tuple(x/255 for x in CLASS_COLORS[i]),
        label=f"Class {i}: {CLASS_NAMES[i]}"
    ) for i in range(4)]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9, framealpha=0.85)
    plt.tight_layout()
    out = OUTPUT_DIR / "preview_predictions.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"[Saved] Preview → {out}")
    plt.show()


# ==========================================================
# MAIN
# ==========================================================
def finetune(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Build dataframe ────────────────────────────────
    df = build_dataframe(args.image_dir, args.mask_dir, args.valid_split, SEED)
    if len(df) == 0:
        print("[ERROR] No image-mask pairs found. ตรวจสอบ path และชื่อไฟล์ mask (_mask.png)")
        sys.exit(1)

    # ── 2. Validate masks ─────────────────────────────────
    validate_masks(df)

    # ── 3. Build DataLoaders ──────────────────────────────
    print("\nBuilding DataLoaders...")
    # ImageDataLoaders.from_df ต้องใช้ relative path
    rel_df = df.copy()

    dls = ImageDataLoaders.from_df(
        rel_df,
        fn_col="image",
        label_col="mask",
        valid_col="is_valid",
        item_tfms=Resize(args.img_size),
        batch_tfms=[*aug_transforms(
            size=args.patch_size,
            min_scale=0.1,
            do_flip=True,
            max_rotate=30.0,
        )],
        y_block=MaskBlock(codes=CLASS_NAMES),
        bs=args.batch_size,
    )
    print(f"  Batch size : {args.batch_size}")
    print(f"  img_size   : {args.img_size} → patch: {args.patch_size}")
    print(f"  GPU        : {'Yes — ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No (CPU mode)'}")

    # ── 4. Create new 4-class model ───────────────────────
    print("\nCreating new 4-class U-Net...")
    set_seed(SEED, True)
    dls.rng.seed(SEED)

    learner = unet_learner(
        dls,
        resnet50,
        n_out=N_OUT_NEW,
        pretrained=False,       # ไม่ดึง ImageNet weights ใหม่ เพราะจะ load เองด้านล่าง
        metrics=DiceMulti,
    ).to_fp16()

    # ── 5. Load old weights ───────────────────────────────
    learner = load_pretrained_weights(learner, args.model_path)

    # ── 6. Phase 1: Train head only (encoder frozen) ──────
    print(f"\n{'='*50}")
    print(f"Phase 1: Training HEAD only ({args.epochs_head} epochs)")
    print(f"{'='*50}")
    learner.freeze()
    learner.fit_one_cycle(args.epochs_head, lr_max=3e-3)

    # ── 7. Phase 2: Train full network ────────────────────
    print(f"\n{'='*50}")
    print(f"Phase 2: Training FULL network ({args.epochs_full} epochs)")
    print(f"{'='*50}")
    learner.unfreeze()
    learner.fit_one_cycle(
        args.epochs_full,
        lr_max=slice(1e-5, 5e-5),   # encoder ช้า, decoder เร็ว
    )

    # ── 8. Save ───────────────────────────────────────────
    model_save_path = OUTPUT_DIR / NEW_EXP_NAME
    save_model(str(model_save_path), learner.model, learner.opt)
    print(f"\n[Saved] Model weights → {model_save_path}.pth")

    # Export เป็น .pkl ด้วย (สำหรับใช้กับ load_learner)
    pkl_path = OUTPUT_DIR / f"{NEW_EXP_NAME}.pkl"
    learner.export(pkl_path)
    print(f"[Saved] Learner pkl   → {pkl_path}")

    # ── 9. Plot & Preview ─────────────────────────────────
    plot_history(learner.recorder, OUTPUT_DIR / "training_history.png")
    print("\nGenerating predictions on validation samples...")
    preview_predictions(learner, df, n=3)

    print("\n✓ Finetuning complete!")
    print(f"  โมเดลใหม่อยู่ที่: {pkl_path}")


# ==========================================================
# CLI
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune segmentation model — add abandoned lead")
    parser.add_argument("--model_path",   default=str(MODEL_PATH),  help="Path to segmentation.pkl")
    parser.add_argument("--image_dir",    default=str(IMAGE_DIR),   help="Folder with X-ray images")
    parser.add_argument("--mask_dir",     default=str(MASK_DIR),    help="Folder with mask PNGs")
    parser.add_argument("--epochs_head",  type=int, default=5,      help="Epochs for head training (Phase 1)")
    parser.add_argument("--epochs_full",  type=int, default=20,     help="Epochs for full training (Phase 2)")
    parser.add_argument("--batch_size",   type=int, default=BATCH_SIZE)
    parser.add_argument("--img_size",     type=int, default=IMG_SIZE)
    parser.add_argument("--patch_size",   type=int, default=PATCH_SIZE)
    parser.add_argument("--valid_split",  type=float, default=VALID_SPLIT)
    args = parser.parse_args()

    finetune(args)