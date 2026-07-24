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
#   2. PER-CASE MEDIAN [IQR]  — among images where a given class is
#                                PRESENT, the % of that image's pixels
#                                occupied by that class. Characterizes
#                                typical lesion/lead size, not class imbalance.
#
# CHANGE vs original: per-case stats are now computed for BOTH
#   class 2 (active lead) and class 3 (abandoned_lead), not just
#   abandoned_lead -- needed to fill the "Active lead" row of
#   Supplementary Table S2 (previously left blank).
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

# classes for which we want per-case (class-present-only) coverage stats
PER_CASE_CLASSES = {2: "active_lead", 3: "abandoned_lead"}


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
    # per-case % for each class in PER_CASE_CLASSES (collected in same pass)
    per_case_pct = {c: [] for c in PER_CASE_CLASSES}

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

        for cls_idx in PER_CASE_CLASSES:
            if case_counts[cls_idx] > 0:  # this class present in this case
                per_case_pct[cls_idx].append(100 * case_counts[cls_idx] / img_total)

    for i, name in enumerate(CLASS_NAMES):
        pct = 100 * pixel_counts[i] / total_pixels if total_pixels > 0 else 0
        print(f"Class {i} ({name}): {pixel_counts[i]:,} pixels ({pct:.3f}%)")

    print("\n>>> Overall pooled abandoned_lead percentage (use for class-weight "
          "justification in Methods):")
    abdn_overall_pct = 100 * pixel_counts[3] / total_pixels if total_pixels > 0 else 0
    print(f"    {abdn_overall_pct:.3f}% of all pixels across the dataset")

    # ---- Per-case stats (for each class in PER_CASE_CLASSES, cases where PRESENT only) ----
    per_case_stats = {}
    for cls_idx, cls_label in PER_CASE_CLASSES.items():
        vals = per_case_pct[cls_idx]
        print(f"\n--- Per-case {CLASS_NAMES[cls_idx]} pixel %, among cases where PRESENT "
              f"(n={len(vals)}) ---")
        if vals:
            arr = np.array(vals)
            median = np.median(arr)
            q1, q3 = np.percentile(arr, [25, 75])
            vmin, vmax = arr.min(), arr.max()
            print(f"    Median [IQR]: {median:.3f}% [{q1:.3f}\u2013{q3:.3f}%]")
            print(f"    Min\u2013Max:      {vmin:.3f}%\u2013{vmax:.3f}%")
            print(f'    For Supplementary Table S2 ("Pixel coverage among class-present '
                  f'cases only"):')
            print(f'    "Median {median:.1f}% (IQR {q1:.2f}-{q3:.2f}%), '
                  f'Range {vmin:.2f}-{vmax:.2f}%"')
            per_case_stats[cls_idx] = {
                "median": median, "q1": q1, "q3": q3, "min": vmin, "max": vmax,
                "n": len(vals),
            }
        else:
            print(f"    No cases with {CLASS_NAMES[cls_idx]} found.")
            per_case_stats[cls_idx] = None

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
        "per_case_pct": per_case_pct,
        "per_case_stats": per_case_stats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_imgs", default="C:/CIEDID_data/AbdnL/data")
    parser.add_argument("--new_masks", default="C:/CIEDID_data/AbdnL/mask")
    parser.add_argument("--save_csv_dir", default="C:/CIEDID_data/AbdnL",
                         help="Optional directory to save per-case %% CSVs for each tracked class")
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

    if args.save_csv_dir:
        out_dir = pathlib.Path(args.save_csv_dir)
        for cls_idx, cls_label in PER_CASE_CLASSES.items():
            vals = results["per_case_pct"][cls_idx]
            if vals:
                out_path = out_dir / f"per_case_{cls_label}_pct.csv"
                pd.DataFrame({f"{cls_label}_pct": vals}).to_csv(out_path, index=False)
                print(f"📄 Per-case {cls_label} %% saved → {out_path}")


if __name__ == "__main__":
    main()