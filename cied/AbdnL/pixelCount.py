import numpy as np
from fastai.vision.all import PILImage
import pathlib

img_dir  = pathlib.Path("C:/CIEDID_data/AbdnL/data")
mask_dir = pathlib.Path("C:/CIEDID_data/AbdnL/mask")

TARGET = 512
scaled_counts = []

for mask_path in mask_dir.glob("*.png"):
    arr = np.array(PILImage.create(mask_path))
    px  = (arr == 3).sum()
    if px == 0:
        continue

    # ลองหา image ทั้ง .png และ .jpg
    stem = mask_path.stem.replace("_mask", "")
    img_path = None
    for ext in [".png", ".jpg", ".jpeg"]:
        candidate = img_dir / (stem + ext)
        if candidate.exists():
            img_path = candidate
            break

    if img_path is None:
        print(f"⚠️  No image found for: {mask_path.name}")
        continue

    orig_w, orig_h = PILImage.create(img_path).size
    scale     = (TARGET / orig_w) * (TARGET / orig_h)
    scaled_px = int(px * scale)
    scaled_counts.append(scaled_px)

scaled_counts = np.array(scaled_counts)
print(f"N                : {len(scaled_counts)}")
print(f"Min  @512px      : {scaled_counts.min():,}")
print(f"5th pct @512px   : {np.percentile(scaled_counts, 5):,.0f}")
print(f"Median @512px    : {np.median(scaled_counts):,.0f}")
print(f"Mean @512px      : {scaled_counts.mean():,.0f}")
print(f"\n→ Recommended threshold: {int(np.percentile(scaled_counts, 5)):,} px")