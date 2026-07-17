"""
plot_learning_curves_S3.py

Generates Supplementary Figure S3: training/validation loss and per-class Dice
curves across all three fine-tuning phases (decoder warmup, head-only,
full fine-tune) for a single reported fold.

Input:
    A training_history.csv (or .xlsx export of it) produced by finetuneCV.py's
    CSVLogger callback. Because CSVLogger appends across phases, the raw file
    contains repeated header rows and phase-local epoch numbering (each phase
    restarts at epoch 0) -- this script cleans both before plotting.

Usage:
    python plot_learning_curves_S3.py \
        --input training_and_valid_loss_data.xlsx \
        --fold 0 \
        --selection-metric dice_abdn \
        --output Supplementary_Figure_S3_learning_curves.png

Requires: pandas, matplotlib, openpyxl (for .xlsx input)
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Colors matched to the paper's own class-color scheme (see CLASS_COLORS in finetuneCV.py)
COL_GEN  = "#3399FF"   # generator     -> RGB (0.20, 0.60, 1.00)
COL_LEAD = "#33D973"   # active lead   -> RGB (0.20, 0.85, 0.45)
COL_ABDN = "#FF4040"   # abandoned lead -> RGB (1.00, 0.25, 0.25)
COL_LOSS = "#12263A"   # navy, used for both loss curves


def load_and_clean(path: str) -> pd.DataFrame:
    """Load a raw training_history export and strip repeated header rows
    that CSVLogger inserts each time a new fine-tuning phase begins."""
    if path.lower().endswith((".xlsx", ".xls")):
        df_raw = pd.read_excel(path)
    else:
        df_raw = pd.read_csv(path, comment="#")

    df = df_raw[df_raw["epoch"] != "epoch"].reset_index(drop=True)
    for col in ["epoch", "train_loss", "valid_loss", "dice_generator", "dice_lead", "dice_abdn"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["global_epoch"] = range(len(df))
    return df


def get_phase_ranges(df: pd.DataFrame):
    """Identify phase boundaries from points where the raw per-phase epoch
    counter resets to 0 (i.e. a new phase started)."""
    resets = df.index[df["epoch"] == 0].tolist()
    resets.append(len(df))
    ranges = [(resets[i], resets[i + 1]) for i in range(len(resets) - 1)]
    return ranges


PHASE_LABELS = [
    "Phase 0\nDecoder warmup\n(encoder frozen)",
    "Phase 1\nHead only\n(all but head frozen)",
    "Phase 2\nFull fine-tuning",
]
BAND_COLORS = ["#F4F6F8", "#FFFFFF", "#EAF2FA"]


def get_boundaries(phase_ranges, n_rows):
    """Single source of truth for band edges AND divider-line positions,
    so the shaded background and the dotted phase-divider lines always
    line up exactly (both are derived from the same boundary list)."""
    boundaries = [-0.5]
    for start, _ in phase_ranges[1:]:
        boundaries.append(start - 0.5)
    boundaries.append(n_rows - 0.5)
    return boundaries


def plot(df: pd.DataFrame, fold: int, selection_metric: str, output_path: str):
    phase_ranges = get_phase_ranges(df)
    boundaries = get_boundaries(phase_ranges, len(df))

    plt.rcParams.update({"font.family": "inter", "font.size": 16, "axes.labelsize": 16, "axes.titlesize": 18})

    fig, ax1 = plt.subplots(figsize=(18, 14))

    ax1.plot(df["global_epoch"], df["train_loss"], color=COL_LOSS, linestyle="--",
              linewidth=1.8, label="Train loss")
    ax1.plot(df["global_epoch"], df["valid_loss"], color=COL_LOSS, linestyle="-",
              linewidth=2.0, label="Valid loss")
    ax1.set_xlabel("Epoch (continuous across fine-tuning phases)", fontsize=18)
    ax1.set_ylabel("Focal-DSC loss", color=COL_LOSS, fontsize=18)
    ax1.tick_params(axis="y", labelcolor=COL_LOSS)
    ax1.set_xlim(-0.5, len(df) - 0.5)
    ax1.xaxis.set_major_locator(mticker.MultipleLocator(2))

    ax2 = ax1.twinx()
    ax2.plot(df["global_epoch"], df["dice_generator"], color=COL_GEN, linewidth=2.2,
              label="DSC \u2013 generator")
    ax2.plot(df["global_epoch"], df["dice_lead"], color=COL_LEAD, linewidth=2.2,
              label="DSC \u2013 active lead")
    ax2.plot(df["global_epoch"], df["dice_abdn"], color=COL_ABDN, linewidth=2.6,
              label="DSC \u2013 abandoned lead (selection metric)")
    ax2.set_ylabel("DSC coefficient", fontsize=18)
    ax2.set_ylim(0, 1.0)

    # phase shading + dividers + labels -- all derived from the same `boundaries`
    # list so the shaded bands and the dotted divider lines always coincide.
    for i, bc in enumerate(BAND_COLORS):
        ax1.axvspan(boundaries[i], boundaries[i + 1], color=bc, alpha=0.6, zorder=0)
    for b in boundaries[1:-1]:
        ax1.axvline(b, color="#888888", linestyle=":", linewidth=1.2, zorder=1)

    y_top = ax1.get_ylim()[1]
    for i, label in enumerate(PHASE_LABELS):
        mid = (boundaries[i] + boundaries[i + 1]) / 2
        ax1.text(mid, y_top * 0.99, label, ha="center", va="top", fontsize=18, color="#333333", fontstyle="italic")

    # mark the selected checkpoint (highest value of the selection metric)
    best_idx = df[selection_metric].idxmax()
    best_epoch = df.loc[best_idx, "global_epoch"]
    best_val = df.loc[best_idx, selection_metric]
    ax2.scatter([best_epoch], [best_val], marker="*", s=500, color=COL_ABDN,
                edgecolor="#7A0000", linewidth=1, zorder=5)
    ax2.annotate("Selected checkpoint\n(the best-performing model)", xy=(best_epoch, best_val),
                 xytext=(best_epoch - 4.2, best_val + 0.16), fontsize=18, color="#7A0000",
                 arrowprops=dict(arrowstyle="-", color="#7A0000", lw=0.8))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, loc="upper center", fontsize=18, framealpha=0.92,
               ncol=3, columnspacing=1.5, handlelength=1.5, bbox_to_anchor=(0.5, 0.85))

    fig.suptitle(f"Training and Validation History Across Fine-Tuning Phases \u2014 Fold {fold}\n", fontsize=20, 
                 fontweight="bold", y=0.88)
    #ax1.set_title(
    #    f"Checkpoint selected by highest validation DSC for the abandoned lead class",
    #    fontsize=18, pad=10, fontstyle="italic"
    #)
    ax1.grid(True, alpha=0.25, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.9], pad=2.5)
    fig.text(
        0.08, 0.01,
        "Loss = 0.5 \u00d7 Focal loss (\u03b3 = 2.0, class-weighted [1, 10, 10, 45] for "
        "background/generator/active lead/abandoned lead) + 0.5 \u00d7 (1 \u2212 mean per-class Dice).",
        ha="left", va="bottom", fontsize=15, color="#222222", style="italic"
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved -> {output_path}")
    print(f"Selected checkpoint: global epoch {best_epoch}, {selection_metric} = {best_val:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Generate Supplementary Figure S3 (learning curves).")
    parser.add_argument("--input", default="cied/abdnL/learningCurveCSV.csv", help="Path to training_history.csv or .xlsx export")
    parser.add_argument("--fold", type=int, default=0, help="Fold number this history belongs to")
    parser.add_argument("--selection-metric", default="dice_abdn",
                         help="Column used for checkpoint selection (default: dice_abdn)")
    parser.add_argument("--output", default="cied/AbdnL/figures/Figure_S3.png")
    args = parser.parse_args()

    df = load_and_clean(args.input)
    plot(df, args.fold, args.selection_metric, args.output)


if __name__ == "__main__":
    main()