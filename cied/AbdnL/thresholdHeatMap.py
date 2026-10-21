"""
plot_threshold_heatmap_table.py

One figure: sensitivity and specificity heatmaps over a probability x
pixel-area threshold grid (top), and a table below comparing the selected
operating point and its neighbours in the OOF and held-out test sets.

All numbers in the table are read from the CSV files (nothing is hard-coded),
so rerunning after the OOF analysis is repeated on the final 195 cases
updates the table and its footnote automatically.

CSV columns required: prob_threshold, pixel_threshold, TP, FP, FN, TN

Example:
    python plot_threshold_heatmap_table.py \
        --heatmap_csv detection_thresholds_test_grid.csv \
        --test_csv    detection_thresholds_test_grid.csv \
        --oof_csv     oof_detection_thresholds.csv \
        --out         supp_threshold_sensitivity.png \
        --annotate
"""
import argparse
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "DejaVu Sans", "Arial", "Liberation Sans", "sans-serif"],
})

DEFAULT_POINTS = "0.80:600,0.80:500,0.80:700,0.80:300,0.75:600,0.85:600,0.85:300"


# ----------------------------------------------------------------------------
def load(path, px_min=None, px_max=None):
    df = pd.read_csv(path)
    if px_min is not None:
        df = df[df.pixel_threshold >= px_min]
    if px_max is not None:
        df = df[df.pixel_threshold <= px_max]
    return df


def totals(df):
    """Positives and negatives implied by the confusion-matrix counts."""
    r = df.iloc[0]
    return int(r.TP + r.FN), int(r.FP + r.TN)


def cell(df, p, px):
    m = df[(np.isclose(df.prob_threshold, p)) & (df.pixel_threshold == px)]
    return None if m.empty else m.iloc[0]


def fmt(num, den):
    return f"{int(num)}/{den} ({100 * num / den:.1f}%)"


def f1(c):
    """F1 from confusion-matrix counts: 2TP / (2TP + FP + FN)."""
    return f"{2 * c.TP / (2 * c.TP + c.FP + c.FN):.3f}"


def parse_points(s):
    pts = []
    for tok in s.split(","):
        p, px = tok.split(":")
        pts.append((float(p), int(px)))
    return pts


# ----------------------------------------------------------------------------
def heatmap(ax, df, value, n_total, title, sel, tie, annotate, count_col):
    probs = sorted(df.prob_threshold.unique())
    pxs = sorted(df.pixel_threshold.unique())
    grid = df.pivot(index="prob_threshold", columns="pixel_threshold", values=value)
    grid = grid.reindex(index=probs, columns=pxs)

    im = ax.imshow(np.ma.masked_invalid(grid.values), aspect="auto",
                   origin="lower", cmap="viridis", vmin=0, vmax=1)
    step = max(1, len(pxs) // 10)
    
    ax.set_xticks(range(0, len(pxs), step))
    ax.set_xticklabels([int(pxs[j]) for j in range(0, len(pxs), step)],
                       rotation=90, fontsize=12)
    ax.set_yticks(range(len(probs)))
    ax.set_yticklabels([f"{p:g}" for p in probs], fontsize=12)
    ax.set_xlabel("Pixel-area threshold (px)", fontsize=12)
    ax.set_ylabel("Probability threshold", fontsize=12)
    ax.set_title(title, fontsize=16, loc="left")

    def box(pt, **kw):
        if pt is None:
            return
        p, px = pt
        pi = [i for i, v in enumerate(probs) if np.isclose(v, p)]
        if pi and px in pxs:
            ax.add_patch(Rectangle((pxs.index(px) - .5, pi[0] - .5), 1, 1,
                                   fill=False, **kw))
    box(sel, ec="red", lw=2)
    box(tie, ec="white", lw=2, ls="--")

    if annotate:
        cnt = df.pivot(index="prob_threshold", columns="pixel_threshold",
                       values=count_col).reindex(index=probs, columns=pxs)
        for i in range(len(probs)):
            for j in range(len(pxs)):
                v = cnt.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{int(v)}/{n_total}", ha="center",
                            va="center", fontsize=12, color="white")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def table(ax, points, oof, test, sel):
    n_pos_t, n_neg_t = totals(test)
    n_pos_o, n_neg_o = totals(oof) if oof is not None else (None, None)

    
    header = ["Probability", "Area (px)"]
    if oof is not None:
        header += [f"OOF sensitivity\n(of {n_pos_o})", f"OOF specificity\n(of {n_neg_o})", "OOF F1"]
    header += [f"Sensitivity (of {n_pos_t})", f"Specificity (of {n_neg_t})"]

    rows, sel_row = [], None
    for k, (p, px) in enumerate(points):
        row = [f">{p:g}", f">{px}"]
        if oof is not None:
            c = cell(oof, p, px)
            row += ["n/a"] * 3 if c is None else [fmt(c.TP, n_pos_o), fmt(c.TN, n_neg_o)]
        c = cell(test, p, px)
        row += ["n/a"] * 3 if c is None else [fmt(c.TP, n_pos_t), fmt(c.TN, n_neg_t)]
        rows.append(row)
        if (p, px) == sel:
            sel_row = k

    ax.axis("off")
    ax.set_title("C. Selected operating point and neighbouring thresholds",
                 loc="left", fontsize=14, pad=8)
    t = ax.table(cellText=rows, colLabels=header, loc="upper center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(12)
    t.scale(1, 1.55)
    for (r, _), c in t.get_celld().items():
        if r == 0:
            c.set_facecolor("#d9d9d9")
            c.set_text_props(fontweight="bold")
            c.set_height(c.get_height() * 1.35)
        elif sel_row is not None and r == sel_row + 1:
            c.set_facecolor("#fadbd8")
            c.set_text_props(fontweight="bold")
    return n_pos_o, n_neg_o, n_pos_t, n_neg_t


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heatmap_csv", default="cied/AbdnL/detection_thresholds_test.csv", help="grid shown as heatmaps")
    ap.add_argument("--test_csv", default="cied/AbdnL/detection_thresholds_test.csv", help="test-set grid for the table")
    ap.add_argument("--oof_csv", default=None, help="OOF grid for the table (optional)")
    ap.add_argument("--out", default="cied/AbdnL/thresholds_heatmap.png")
    ap.add_argument("--heatmap_title", default="Held-out test set")
    ap.add_argument("--sel_prob", type=float, default=0.80)
    ap.add_argument("--sel_px", type=int, default=600)
    ap.add_argument("--tie_prob", type=float, default=None)   # off unless given
    ap.add_argument("--tie_px", type=int, default=None)
    ap.add_argument("--points", default=DEFAULT_POINTS,
                    help='table rows as "prob:px,prob:px,..."; first match of --sel is highlighted')
    ap.add_argument("--px_min", type=int, default=100)
    ap.add_argument("--px_max", type=int, default=1500)
    ap.add_argument("--annotate", action="store_true", help="n/N inside heatmap cells")
    ap.add_argument("--sens_floor", type=float, default=0.975,
                    help="pre-specified OOF sensitivity floor, quoted in the footnote")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()

    hm = load(a.heatmap_csv, a.px_min, a.px_max)
    test = load(a.test_csv)
    oof = load(a.oof_csv) if a.oof_csv else None
    points = parse_points(a.points)
    sel = (a.sel_prob, a.sel_px)
    tie = (a.tie_prob, a.tie_px) if a.tie_prob is not None else None

    n_pos, n_neg = totals(hm)
    n_rows = len(points)
    fig = plt.figure(figsize=(13, 5.0 + 0.36 * (n_rows + 2) + 0.9), dpi=a.dpi)
    gs = fig.add_gridspec(2, 2, height_ratios=[5.0, 0.36 * (n_rows + 2) + 0.2],
                          hspace=0.45, wspace=0.22)
    heatmap(fig.add_subplot(gs[0, 0]), hm, "sensitivity", n_pos,
            f"A. Sensitivity (of {n_pos} positives)", sel, tie, a.annotate, "TP")
    heatmap(fig.add_subplot(gs[0, 1]), hm, "specificity", n_neg,
            f"B. Specificity (of {n_neg} negatives)", sel, tie, a.annotate, "TN")
    axt = fig.add_subplot(gs[1, :])
    n_pos_o, n_neg_o, n_pos_t, n_neg_t = table(axt, points, oof, test, sel)

    foot = (f"C. Selected operating point (bold) and neighbouring thresholds. Red box in A and B: "
            f"p>{a.sel_prob:g}, area>{a.sel_px} px. ")
    if tie:
        foot += f"White dashed box: p>{a.tie_prob:g}, area>{a.tie_px} px. "
    foot += (f"Selection required OOF sensitivity of at least {a.sens_floor:.1%} before maximizing F1, "
             "so rows below this floor were ineligible despite a higher F1. F1 = 2TP/(2TP+FP+FN). ")
    foot += "Test-set values are descriptive and were not used to select the threshold."
    if oof is not None:
        foot += (f" OOF values are from {n_pos_o + n_neg_o} cases "
                 f"({n_pos_o} positive, {n_neg_o} negative) in the supplied file.")
    #fig.text(0.5, 0.0, textwrap.fill(foot, 170), ha="center", va="top", fontsize=12)
    #fig.suptitle(f"{a.heatmap_title}: threshold sensitivity analysis", fontsize=14, y=0.995)
    fig.savefig(a.out, bbox_inches="tight")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()