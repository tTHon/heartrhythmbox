"""
oof_grid_search.py
===================
Sweep (probability threshold x pixel-area threshold) on the LEAKAGE-FREE
out-of-fold results produced by oof_inference.py, and compute case-level
TP/FP/FN/TN/sensitivity/specificity/PPV/F1 at every combination.

Output format matches the original detection_thresholds.csv so existing
analysis scripts (ROC plotting, Youden's J search, etc.) work unchanged —
just point them at oof_detection_thresholds.csv instead.

Usage:
    python oof_grid_search.py \
        --oof_results "C:/CIEDID_data/AbdnL/oof_results/oof_per_image_results.csv" \
        --output_dir  "C:/CIEDID_data/AbdnL/oof_results"
"""

import argparse
import pathlib
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof_results",
                     default="C:/CIEDID_data/AbdnL/oof_results/oof_per_image_results.csv")
    ap.add_argument("--output_dir",
                     default="C:/CIEDID_data/AbdnL/oof_results")
    ap.add_argument("--pixel_min", type=int, default=0)
    ap.add_argument("--pixel_max", type=int, default=5000)
    ap.add_argument("--pixel_step", type=int, default=100)
    args = ap.parse_args()

    df = pd.read_csv(args.oof_results)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    prob_thresholds = sorted(int(c.split("_")[1]) / 100 for c in prob_cols)

    print(f"Loaded {len(df)} out-of-fold images "
          f"({df['has_abandoned'].sum()} with abandoned lead, "
          f"{(~df['has_abandoned']).sum()} without)")
    print(f"Probability thresholds available: {prob_thresholds}")

    pixel_thresholds = list(range(args.pixel_min, args.pixel_max + 1, args.pixel_step))

    gt_pos = df["has_abandoned"].astype(bool)
    n_pos = gt_pos.sum()
    n_neg = (~gt_pos).sum()

    rows = []
    for pt in prob_thresholds:
        col = f"prob_{int(pt*100):02d}"
        if col not in df.columns:
            continue
        pixel_counts = df[col].values

        for px in pixel_thresholds:
            pred_pos = pixel_counts > px

            TP = int((pred_pos & gt_pos.values).sum())
            FP = int((pred_pos & ~gt_pos.values).sum())
            FN = int((~pred_pos & gt_pos.values).sum())
            TN = int((~pred_pos & ~gt_pos.values).sum())

            sens = TP / n_pos if n_pos > 0 else float("nan")
            spec = TN / n_neg if n_neg > 0 else float("nan")
            ppv  = TP / (TP + FP) if (TP + FP) > 0 else float("nan")
            f1   = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else float("nan")

            rows.append({
                "prob_threshold":  pt,
                "pixel_threshold": px,
                "TP": TP, "FP": FP, "FN": FN, "TN": TN,
                "sensitivity": round(sens, 4),
                "PPV":         round(ppv, 4),
                "specificity": round(spec, 4),
                "F1":          round(f1, 4),
            })

    result = pd.DataFrame(rows)
    out_path = output_dir / "oof_detection_thresholds.csv"
    result.to_csv(out_path, index=False)

    # ── Quick summary: best Youden's J and best "sensitivity=1.0, max spec" point
    result["youden"] = result["sensitivity"] + result["specificity"] - 1
    best_j = result.loc[result["youden"].idxmax()]

    sens1 = result[result["sensitivity"] == 1.0]
    best_sens1 = (sens1.loc[sens1["specificity"].idxmax()]
                  if len(sens1) > 0 else None)

    print(f"\n💾 Saved full grid → {out_path}  ({len(result)} rows)")
    print(f"\nBest Youden's J point (out-of-fold, leakage-free):")
    print(best_j[["prob_threshold", "pixel_threshold", "TP", "FP", "FN", "TN",
                   "sensitivity", "specificity", "youden"]])

    if best_sens1 is not None:
        print(f"\nBest specificity at sensitivity=1.0 (out-of-fold):")
        print(best_sens1[["prob_threshold", "pixel_threshold", "TP", "FP", "FN", "TN",
                           "sensitivity", "specificity"]])
    else:
        print("\n⚠️  No threshold combination achieved sensitivity=1.0 "
              "on the out-of-fold data — this is expected/healthy if the "
              "earlier in-sample grid was inflated by memorization.")


if __name__ == "__main__":
    main()