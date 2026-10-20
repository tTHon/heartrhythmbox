"""
plot_threshold_heatmap.py

Draw sensitivity and specificity heatmaps over a probability x pixel-area
threshold grid for abandoned-lead detection (case level).

Input CSV columns (as produced by the threshold-grid scripts):
    prob_threshold, pixel_threshold, TP, FP, FN, TN, sensitivity, specificity, ...

Usage:
    python plot_threshold_heatmap.py \
        --csv detection_thresholds_test_grid.csv \
        --out threshold_heatmap_test.png \
        --title "Held-out test set (n=50)" \
        --sel_prob 0.80 --sel_px 600 \
        --px_max 2000
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


def pivot(df, value):
    """Rows = probability threshold, columns = pixel-area threshold."""
    return df.pivot(index="prob_threshold", columns="pixel_threshold", values=value)


def draw(ax, grid, title, sel, tie, annotate_counts, counts=None):
    probs, pxs = list(grid.index), list(grid.columns)
    im = ax.imshow(grid.values, aspect="auto", origin="lower",
                   cmap="viridis", vmin=0, vmax=1)

    step = max(1, len(pxs) // 10)
    ax.set_xticks(range(0, len(pxs), step))
    ax.set_xticklabels([int(pxs[j]) for j in range(0, len(pxs), step)],
                       rotation=90, fontsize=7)
    ax.set_yticks(range(len(probs)))
    ax.set_yticklabels([f"{p:g}" for p in probs], fontsize=8)
    ax.set_xlabel("Pixel-area threshold (px)")
    ax.set_ylabel("Probability threshold")
    ax.set_title(title, fontsize=10, loc="left")

    def box(point, **kw):
        p, px = point
        if p in probs and px in pxs:
            ax.add_patch(Rectangle((pxs.index(px) - .5, probs.index(p) - .5),
                                   1, 1, fill=False, **kw))

    if sel:
        box(sel, ec="red", lw=2)
    if tie:
        box(tie, ec="white", lw=2, ls="--")

    if annotate_counts and counts is not None:   # write n/N in each cell
        for i, p in enumerate(probs):
            for j, px in enumerate(pxs):
                ax.text(j, i, counts.loc[p, px], ha="center", va="center",
                        fontsize=5, color="white")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="cied/AbdnL/detection_thresholds test.csv")
    ap.add_argument("--out", default="cied/AbdnL/threshold_heatmap.png")
    ap.add_argument("--title", default="Threshold grid")
    ap.add_argument("--sel_prob", type=float, default=0.80)
    ap.add_argument("--sel_px", type=int, default=600)
    ap.add_argument("--tie_prob", type=float, default=0.85,
                    help="optional second point to outline (dashed white)")
    ap.add_argument("--tie_px", type=int, default=300)
    ap.add_argument("--px_min", type=int, default=100)
    ap.add_argument("--px_max", type=int, default=2000)
    ap.add_argument("--annotate", action="store_true",
                    help="print TP/positives and TN/negatives in each cell")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[(df.pixel_threshold >= args.px_min) & (df.pixel_threshold <= args.px_max)]

    n_pos = int((df.TP + df.FN).iloc[0])
    n_neg = int((df.FP + df.TN).iloc[0])

    sens, spec = pivot(df, "sensitivity"), pivot(df, "specificity")
    sel = (args.sel_prob, args.sel_px)
    tie = (args.tie_prob, args.tie_px) if args.tie_prob is not None else None

    sens_n = pivot(df, "TP").astype(int).astype(str) + "/" + str(n_pos) if args.annotate else None
    spec_n = pivot(df, "TN").astype(int).astype(str) + "/" + str(n_neg) if args.annotate else None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=args.dpi)
    draw(axes[0], sens, f"A. Sensitivity (of {n_pos} positives)", sel, tie, args.annotate, sens_n)
    draw(axes[1], spec, f"B. Specificity (of {n_neg} negatives)", sel, tie, args.annotate, spec_n)
    fig.suptitle(f"{args.title}. Red box: selected point "
                 f"(p>{args.sel_prob:g}, area>{args.sel_px} px)"
                 + (f"; white dashed: (p>{args.tie_prob:g}, area>{args.tie_px} px)" if tie else ""),
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"saved {args.out}  (positives={n_pos}, negatives={n_neg})")


if __name__ == "__main__":
    main()