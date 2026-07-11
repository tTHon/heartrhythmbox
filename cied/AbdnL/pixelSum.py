# summarize_pixel_distribution.py
#
# Standalone script to summarize per-class pixel distribution across the
# full training/CV dataset (data + mask folders), independent of any
# fold split. Reuses the same class definitions as finetuneCV.py.
#
# Reports two distinct numbers (do not conflate):
#   1. OVERALL POOLED %       — pixels summed across ALL images, then
#                                divided by total pixels. This is the
#                                number that explains class-weight choice,
#                                since FocalDiceLoss weights operate on
#                                pooled pixels, not per-image.
#   2. PER-CASE MEDIAN [IQR]  — among images where abandoned_lead is
#                                PRESENT, the % of that image's pixels
#                                occupied by abandoned_lead. Characterizes
#                                typical lesion size, not class imbalance.
#
# Usage:
#   python summarize_pixel_distribution.py \
#       --new_imgs "C:/CIEDID_data/AbdnL/data" \
#       --new_masks "C:/CIEDID_data/AbdnL/mask"

import argparse
import pathlib
import numpy as np
import pandas as pd
from PIL import Image

CLASS_NAMES = ["background", "generator", "lead", "abandoned_lead"]


def find_mask_for_image(img_path: pathlib.Path, masks_dir: pathlib.Path):
    """Mirrors build_dataframe() logic in finetuneCV.py:
    try '<stem>_mask.png' first, then '<stem>.png'."""
    mask = masks_dir / f"{img_path.stem}_mask.png"
    if mask.exists():
        return mask
    mask = masks_dir / f"{img_path.stem}.png"
    if mask.exists():
        return mask
    return None


def build_file_list(imgs_dir: pathlib.Path, masks_dir: pathlib.Path):
    exts = {".jpg", ".png", ".jpeg", ".bmp"}
    rows = []
    n_missing_mask = 0
    for img in sorted(imgs_dir.iterdir()):
        if img.suffix.lower() not in exts:
            continue
        mask = find_mask_for_image(img, masks_dir)
        if mask is None:
            n_missing_mask += 1
            continue
        rows.append({"image": str(img.resolve()), "mask": str(mask.resolve())})
    if n_missing_mask:
        print(f"⚠️  Skipped {n_missing_mask} image(s) with no matching mask file.")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame):
    print("\n========== DATASET SUMMARY ==========")
    print(f"Total image/mask pairs: {len(df)}")

    # ---- Image-level class presence ----
    print("\n--- Image-level class presence ---")
    class_presence = {i: 0 for i in range(len(CLASS_NAMES))}
    for _, row in df.iterrows():
        arr = np.array(Image.open(row["mask"]))
        for c in np.unique(arr):
            c = int(c)
            if c in class_presence:
                class_presence[c] += 1
    for i, name in enumerate(CLASS_NAMES):
        print(f"Class {i} ({name}): present in {class_presence[i]} / {len(df)} images "
              f"({100 * class_presence[i] / len(df):.1f}%)")

    # ---- Pixel-level distribution: OVERALL POOLED ----
    print("\n--- Pixel-level distribution (OVERALL POOLED across all images) ---")
    pixel_counts = {i: 0 for i in range(len(CLASS_NAMES))}
    total_pixels = 0
    # per-case abandoned_lead % (for cases where present) — collected in same pass
    per_case_abdn_pct = []

    for _, row in df.iterrows():
        arr = np.array(Image.open(row["mask"]))
        vals, counts = np.unique(arr, return_counts=True)
        img_total = arr.size
        total_pixels += img_total

        case_counts = {i: 0 for i in range(len(CLASS_NAMES))}
        for v, c in zip(vals, counts):
            v = int(v)
            if v in pixel_counts:
                pixel_counts[v] += int(c)
                case_counts[v] = int(c)

        if case_counts[3] > 0:  # abandoned_lead present in this case
            per_case_abdn_pct.append(100 * case_counts[3] / img_total)

    for i, name in enumerate(CLASS_NAMES):
        pct = 100 * pixel_counts[i] / total_pixels if total_pixels > 0 else 0
        print(f"Class {i} ({name}): {pixel_counts[i]:,} pixels ({pct:.3f}%)")

    print("\n>>> Overall pooled abandoned_lead percentage (use for class-weight "
          "justification in Methods):")
    abdn_overall_pct = 100 * pixel_counts[3] / total_pixels if total_pixels > 0 else 0
    print(f"    {abdn_overall_pct:.3f}% of all pixels across the dataset")

    # ---- Per-case stats (cases WITH abandoned_lead present only) ----
    print("\n--- Per-case abandoned_lead pixel %, among cases where PRESENT "
          f"(n={len(per_case_abdn_pct)}) ---")
    if per_case_abdn_pct:
        arr = np.array(per_case_abdn_pct)
        median = np.median(arr)
        q1, q3 = np.percentile(arr, [25, 75])
        print(f"    Median [IQR]: {median:.3f}% [{q1:.3f}–{q3:.3f}%]")
        print(f"    Min–Max:      {arr.min():.3f}%–{arr.max():.3f}%")
        print("\n>>> Use this for optional supplementary lesion-size characterization:")
        print(f'    "Among cases with abandoned lead present, the lesion occupied a '
              f'median of {median:.3f}% [IQR {q1:.3f}\u2013{q3:.3f}%] of image pixels."')
    else:
        print("    No cases with abandoned_lead found.")

    print("\n>>> Suggested Methods sentence (overall, for class-weight justification):")
    print(f'    "Abandoned lead pixels constituted the smallest proportion of all '
          f'pixels in the training/CV dataset ({abdn_overall_pct:.3f}% overall), '
          f'motivating the highest class weight (45) among the four segmentation '
          f'classes."')

    print("======================================\n")

    return {
        "pixel_counts": pixel_counts,
        "total_pixels": total_pixels,
        "abdn_overall_pct": abdn_overall_pct,
        "per_case_abdn_pct": per_case_abdn_pct,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_imgs", default="C:/CIEDID_data/AbdnL/data")
    parser.add_argument("--new_masks", default="C:/CIEDID_data/AbdnL/mask")
    parser.add_argument("--save_csv", default="C:/CIEDID_data/AbdnL/per_case_abdn_pct.csv",
                         help="Optional path to save per-case abandoned_lead %% as CSV")
    args = parser.parse_args()

    imgs_dir = pathlib.Path(args.new_imgs)
    masks_dir = pathlib.Path(args.new_masks)

    if not imgs_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {imgs_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Mask folder not found: {masks_dir}")

    df = build_file_list(imgs_dir, masks_dir)
    if df.empty:
        raise RuntimeError("No matching image/mask pairs found — check folder paths.")

    results = summarize(df)

    if args.save_csv and results["per_case_abdn_pct"]:
        out_path = pathlib.Path(args.save_csv)
        pd.DataFrame({"abandoned_lead_pct": results["per_case_abdn_pct"]}).to_csv(
            out_path, index=False)
        print(f"📄 Per-case abandoned_lead %% saved → {out_path}")


if __name__ == "__main__":
    main()