"""
check_generator_area.py
=======================
Inspect ground-truth generator masks across the (train/val) dataset to
derive TWO pipeline hyperparameters consumed by eval_generator_crop.py:

  1. min_area_frac  — smallest plausible generator area, so the area
                       filter doesn't exclude small leadless pacemakers.
  2. spatial_prior   — the anatomically plausible zone for generator
                       centroids, used to reject false-positive components
                       (e.g. text overlays, shoulder/clavicle artifacts)
                       that a low min_area_frac would otherwise let through.

Both must be derived from TRAIN/VAL masks only (never the held-out test
set) since they are pipeline hyperparameters, not results — tuning them
on the test set would leak information into evaluation.

Outputs:
  - generator_area_stats.csv   : per-image area + centroid table
  - generator_area_stats.png   : area histogram + bbox scatter
  - spatial_prior.json         : {row_min, row_max, col_min, col_max}
                                  → pass to eval_generator_crop.py via
                                    --spatial_prior_json

Usage:
    python check_generator_area.py

Optional:
    --imgs         C:/CIEDID_data/AbdnL/data
    --masks        C:/CIEDID_data/AbdnL/mask
    --target_size  640
    --prior_margin 0.10
    --prior_json   C:/CIEDID_data/AbdnL/spatial_prior.json
"""

import argparse
import json
import pathlib
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

GEN_CLASS = 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs",  default="C:/CIEDID_data/AbdnL/data")
    parser.add_argument("--masks", default="C:/CIEDID_data/AbdnL/mask")
    parser.add_argument("--target_size", type=int, default=640,
                        help="Resolution to scale areas to (should match "
                             "the img_size used in eval_generator_crop.py)")
    parser.add_argument("--output_csv",
                        default="C:/CIEDID_data/AbdnL/generator_area_stats.csv")
    parser.add_argument("--prior_margin", type=float, default=0.10,
                        help="Safety margin added around the empirical "
                             "centroid range (fraction of image dimension) "
                             "when deriving the spatial prior.")
    parser.add_argument("--prior_json",
                        default="C:/CIEDID_data/AbdnL/spatial_prior.json",
                        help="Where to save the derived spatial prior, "
                             "consumed by eval_generator_crop.py via "
                             "--spatial_prior_json.")
    args = parser.parse_args()

    img_dir  = pathlib.Path(args.imgs)
    mask_dir = pathlib.Path(args.masks)

    rows = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".png", ".jpeg", ".bmp"}:
            continue
        mask_path = mask_dir / f"{img_path.stem}_mask.png"
        if not mask_path.exists():
            mask_path = mask_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            continue

        mask = np.array(Image.open(mask_path).convert("L"))
        h, w = mask.shape

        gen_mask = (mask == GEN_CLASS)
        gen_area_orig = int(gen_mask.sum())

        if gen_area_orig == 0:
            continue  # no generator annotated in this image

        # Scale area to target resolution.
        # Area scales with the SQUARE of the linear resize factor.
        scale = args.target_size / max(h, w)  # matches pad-resize behaviour
        gen_area_scaled = gen_area_orig * (scale ** 2)

        # Bounding box dimensions (helps distinguish leadless vs conventional)
        ys, xs = np.where(gen_mask)
        bbox_h = ys.max() - ys.min() + 1
        bbox_w = xs.max() - xs.min() + 1
        bbox_h_scaled = bbox_h * scale
        bbox_w_scaled = bbox_w * scale

        # Centroid as a FRACTION of image height/width — used to derive
        # the spatial prior (anatomically plausible generator location zone)
        centroid_row_frac = ys.mean() / h
        centroid_col_frac = xs.mean() / w

        rows.append({
            "image":            img_path.name,
            "orig_h":           h,
            "orig_w":           w,
            "gen_area_orig_px": gen_area_orig,
            "gen_area_scaled_px": round(gen_area_scaled, 1),
            "bbox_h_scaled":    round(bbox_h_scaled, 1),
            "bbox_w_scaled":    round(bbox_w_scaled, 1),
            "bbox_max_dim_scaled": round(max(bbox_h_scaled, bbox_w_scaled), 1),
            "centroid_row_frac": round(centroid_row_frac, 4),
            "centroid_col_frac": round(centroid_col_frac, 4),
        })

    df = pd.DataFrame(rows)
    print(f"\n✅ Found {len(df)} images with annotated generator\n")

    print(f"{'='*60}")
    print(f"  GENERATOR AREA STATISTICS (scaled to {args.target_size}x{args.target_size})")
    print(f"{'='*60}")
    area = df["gen_area_scaled_px"]
    print(f"  min    : {area.min():.0f} px")
    print(f"  5th pct: {area.quantile(0.05):.0f} px")
    print(f"  median : {area.median():.0f} px")
    print(f"  mean   : {area.mean():.0f} px")
    print(f"  95th pct: {area.quantile(0.95):.0f} px")
    print(f"  max    : {area.max():.0f} px")

    print(f"\n  As fraction of {args.target_size}x{args.target_size} image:")
    img_area = args.target_size ** 2
    print(f"  min    : {area.min()/img_area:.5f}")
    print(f"  5th pct: {area.quantile(0.05)/img_area:.5f}")
    print(f"  median : {area.median()/img_area:.5f}")

    # ── Identify smallest cases — likely candidates for leadless ───────
    print(f"\n{'='*60}")
    print(f"  10 SMALLEST GENERATORS (check if these are leadless)")
    print(f"{'='*60}")
    smallest = df.nsmallest(10, "gen_area_scaled_px")
    print(smallest[["image", "gen_area_scaled_px", "bbox_max_dim_scaled"]]
          .to_string(index=False))

    # ── Suggest threshold ────────────────────────────────────────────
    safe_min_area_frac = (area.min() * 0.8) / img_area  # 20% safety margin below true min
    print(f"\n{'='*60}")
    print(f"  SUGGESTED min_area_frac")
    print(f"{'='*60}")
    print(f"  Current default       : 0.0038  ({0.0038*img_area:.0f} px @ {args.target_size}px)")
    print(f"  Safe threshold (-20%) : {safe_min_area_frac:.5f}  "
          f"({safe_min_area_frac*img_area:.0f} px @ {args.target_size}px)")
    print(f"\n  → Use --min_area_frac {safe_min_area_frac:.5f} in eval_generator_crop.py")
    print(f"     to avoid filtering out the smallest real generators "
          f"(including leadless pacemakers if present).")

    # ── Histogram ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(area, bins=30, color="steelblue", edgecolor="white")
    axes[0].axvline(0.0038*img_area, color="red", linestyle="--",
                    label=f"Current threshold\n({0.0038*img_area:.0f}px)")
    axes[0].axvline(area.min(), color="green", linestyle=":",
                    label=f"Dataset min\n({area.min():.0f}px)")
    axes[0].set_xlabel(f"Generator area (px @ {args.target_size}x{args.target_size})")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Generator Area Distribution")
    axes[0].legend(fontsize=8)

    axes[1].scatter(df["bbox_w_scaled"], df["bbox_h_scaled"],
                    alpha=0.6, color="coral", edgecolor="white")
    axes[1].set_xlabel("BBox width (px)")
    axes[1].set_ylabel("BBox height (px)")
    axes[1].set_title("Generator BBox Dimensions\n(small/square cluster = possible leadless)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = pathlib.Path(args.output_csv).with_suffix(".png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\n📊 Histogram saved → {out_png}")
    plt.close()

    # ── Spatial prior (plausible generator location zone) ──────────────
    # Derived from the bounding range of GT centroids, expanded by a
    # safety margin. Used by eval_generator_crop.py to reject obvious
    # off-target false positives (e.g. text overlays, shoulder/clavicle
    # artifacts) BEFORE largest-component selection — this becomes more
    # important as min_area_frac is lowered to catch small leadless
    # pacemakers, since smaller area thresholds also let more noise through.
    print(f"\n{'='*60}")
    print(f"  SPATIAL PRIOR (plausible generator location zone)")
    print(f"{'='*60}")

    row_frac = df["centroid_row_frac"]
    col_frac = df["centroid_col_frac"]

    row_min = max(0.0, row_frac.min() - args.prior_margin)
    row_max = min(1.0, row_frac.max() + args.prior_margin)
    col_min = max(0.0, col_frac.min() - args.prior_margin)
    col_max = min(1.0, col_frac.max() + args.prior_margin)

    print(f"  Centroid row (height) range : {row_frac.min():.3f}–{row_frac.max():.3f}")
    print(f"  Centroid col (width)  range : {col_frac.min():.3f}–{col_frac.max():.3f}")
    print(f"  + {args.prior_margin:.2f} safety margin →")
    print(f"  Prior zone row range : {row_min:.3f}–{row_max:.3f}")
    print(f"  Prior zone col range : {col_min:.3f}–{col_max:.3f}")
    print(f"\n  (This zone is intentionally generous — it only rejects "
          f"components\n   far outside any observed generator position, "
          f"such as text overlays\n   or shoulder/clavicle artifacts.)")

    spatial_prior = {
        "row_min": round(row_min, 4), "row_max": round(row_max, 4),
        "col_min": round(col_min, 4), "col_max": round(col_max, 4),
        "n_masks_used": int(len(df)),
        "margin": args.prior_margin,
        "source": str(mask_dir),
    }
    prior_path = pathlib.Path(args.prior_json)
    prior_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prior_path, "w") as f:
        json.dump(spatial_prior, f, indent=2)
    print(f"\n💾 Spatial prior saved → {prior_path}")
    print(f"     Use --spatial_prior_json {prior_path} in eval_generator_crop.py")

    df.to_csv(args.output_csv, index=False)
    print(f"💾 Full results → {args.output_csv}")


if __name__ == "__main__":
    main()