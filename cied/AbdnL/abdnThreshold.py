"""
eval_abdn_threshold.py
======================
ทดสอบ sensitivity / PPV / specificity ของ abandoned lead detection
ที่ threshold ต่างๆ บน validation set โดยไม่ต้องเทรนใหม่

Usage:
    python eval_abdn_threshold.py \
        --weights  C:/CIEDID_data/AbdnL/models/best_gen.pth \
        --imgs     C:/CIEDID_data/AbdnL/data \
        --masks    C:/CIEDID_data/AbdnL/mask \
        --img_size 512 \
        --patch_size 320

Optional:
    --thresholds 500 1000 1500 2000 2800   # ลิสต์ threshold ที่อยากทดสอบ
    --valid_split 0.2                       # ต้องตรงกับที่ใช้ตอนเทรน
    --output_csv results/threshold_eval.csv
"""

import argparse
import pathlib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from fastai.vision.all import *
from sklearn.model_selection import train_test_split

# ── PyTorch 2.6+ fix ─────────────────────────────────────────────────
_orig_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault('weights_only', False)
    return _orig_load(*a, **kw)
torch.load = _patched_load

import platform
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# ── Constants ────────────────────────────────────────────────────────
CLASS_NAMES  = ["background", "generator", "lead", "abandoned_lead"]
ABDN_CLASS   = 3


# ══════════════════════════════════════════════════════════════════════
# 1. Build validation dataframe (same logic as training script)
# ══════════════════════════════════════════════════════════════════════
def build_val_dataframe(img_dir: pathlib.Path,
                        mask_dir: pathlib.Path,
                        valid_split: float = 0.2,
                        random_state: int = 42) -> pd.DataFrame:
    rows = []
    for img in img_dir.iterdir():
        if img.suffix.lower() not in {".jpg", ".png", ".jpeg", ".bmp"}:
            continue
        mask = mask_dir / f"{img.stem}_mask.png"
        if not mask.exists():
            mask = mask_dir / f"{img.stem}.png"
        if not mask.exists():
            continue
        arr = np.array(PILImage.create(mask))
        rows.append({
            "image":         str(img.resolve()),
            "mask":          str(mask.resolve()),
            "has_abandoned": ABDN_CLASS in np.unique(arr),
        })

    df       = pd.DataFrame(rows)
    n_pos    = df["has_abandoned"].sum()
    stratify = df["has_abandoned"] if n_pos >= 5 else None
    if stratify is None:
        print("⚠️  Too few abandoned-lead images — using random split.")

    _, val_idx = train_test_split(
        df.index, test_size=valid_split,
        stratify=stratify, random_state=random_state
    )
    val_df = df.loc[val_idx].reset_index(drop=True)
    print(f"✅ Val set: {len(val_df)} images "
          f"({val_df['has_abandoned'].sum()} with abandoned lead)")
    return val_df


# ══════════════════════════════════════════════════════════════════════
# 2. Load model
# ══════════════════════════════════════════════════════════════════════
def load_model(weights_path: pathlib.Path,
               img_size: int,
               patch_size: int,
               val_df: pd.DataFrame,
               device: torch.device) -> Learner:

    def get_x(r): return r["image"]
    def get_y(r): return r["mask"]

    # FastAI requires at least one full train batch to initialise unet_learner.
    # We duplicate the first val row 4 times as dummy train rows (bs=4).
    # These are never used for inference — only val rows are evaluated.
    BS_DUMMY = 4
    eval_df = val_df.copy()
    eval_df["is_valid"] = True
    dummy = pd.concat([eval_df.iloc[[0]]] * BS_DUMMY, ignore_index=True)
    dummy["is_valid"] = False
    eval_df = pd.concat([dummy, eval_df], ignore_index=True)

    dblock = DataBlock(
        blocks   = (ImageBlock, MaskBlock(codes=CLASS_NAMES)),
        get_x    = get_x,
        get_y    = get_y,
        splitter = ColSplitter(col="is_valid"),
        item_tfms = Resize(img_size, method='pad'),
        batch_tfms = [Normalize.from_stats(
            [0.4945, 0.4945, 0.4945],
            [0.2267, 0.2267, 0.2267],
        )],
    )
    dls = dblock.dataloaders(eval_df, bs=BS_DUMMY,
                             num_workers=0, pin_memory=True)

    learner = unet_learner(dls, resnet50, n_out=4).to_fp16()
    state   = torch.load(weights_path, map_location=device)
    learner.model.load_state_dict(state)
    learner.model.to(device)
    learner.model.eval()
    print(f"✅ Weights loaded from {weights_path.name}")
    return learner


# ══════════════════════════════════════════════════════════════════════
# 3. Evaluate thresholds
# ══════════════════════════════════════════════════════════════════════
def evaluate_thresholds(learner: Learner,
                        thresholds: list,
                        device: torch.device) -> pd.DataFrame:
    """
    For each threshold, compute per-image TP/FP/FN then aggregate:
        Sensitivity = TP / (TP + FN)   ← how often we catch abdn lead
        PPV         = TP / (TP + FP)   ← when we say yes, how often correct
        Specificity = TN / (TN + FP)   ← how often we correctly say no
        F1          = 2*Sens*PPV / (Sens+PPV)
    """
    # collect per-image (pred_px, targ_px)
    records = []
    learner.model.to(device)
    with torch.no_grad():
        for xb, yb in learner.dls.valid:
            xb  = xb.to(device)
            yb  = yb.to(device)
            with torch.cuda.amp.autocast():
                preds = learner.model(xb).argmax(dim=1)
            for i in range(len(xb)):
                pred_px = (preds[i] == ABDN_CLASS).sum().item()
                targ_px = (yb[i]   == ABDN_CLASS).sum().item()
                records.append({"pred_px": pred_px, "targ_px": targ_px})

    df_img = pd.DataFrame(records)
    print(f"\n📊 Val images evaluated: {len(df_img)}")
    print(f"   Images with abandoned lead (GT): {(df_img['targ_px'] > 0).sum()}")
    print(f"   Pred pixel stats: "
          f"min={df_img['pred_px'].min()} "
          f"median={df_img['pred_px'].median():.0f} "
          f"max={df_img['pred_px'].max()}")

    rows = []
    for thr in thresholds:
        has_pred = df_img["pred_px"] > thr
        has_targ = df_img["targ_px"] > 0

        tp = int(( has_pred &  has_targ).sum())
        fp = int(( has_pred & ~has_targ).sum())
        fn = int((~has_pred &  has_targ).sum())
        tn = int((~has_pred & ~has_targ).sum())

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1   = 2*sens*ppv / (sens+ppv) if (sens+ppv) > 0 else 0.0

        rows.append({
            "threshold":   thr,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "sensitivity": round(sens, 4),
            "PPV":         round(ppv,  4),
            "specificity": round(spec, 4),
            "F1":          round(f1,   4),
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# 4. Plot sensitivity / PPV tradeoff
# ══════════════════════════════════════════════════════════════════════
def plot_threshold_curve(results: pd.DataFrame, output_path: pathlib.Path):
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(results["threshold"], results["sensitivity"],
             "o-", color="red",   linewidth=2, label="Sensitivity (Recall)")
    ax1.plot(results["threshold"], results["PPV"],
             "s-", color="blue",  linewidth=2, label="PPV (Precision)")
    ax1.plot(results["threshold"], results["specificity"],
             "^-", color="green", linewidth=2, label="Specificity")
    ax1.plot(results["threshold"], results["F1"],
             "D-", color="purple",linewidth=2, label="F1")

    ax1.set_xlabel("Threshold (pixels @ prediction resolution)")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1.05)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))
    ax1.legend(loc="center right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    plt.title("Abandoned Lead Detection — Threshold Analysis\n"
              "(higher threshold = fewer FP, lower sensitivity)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"📈 Curve saved → {output_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# 5. Main
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",
                        default="C:/CIEDID_data/AbdnL/models/best_gen.pth",
                        help="Path to best_gen.pth or best_abdn.pth")
    parser.add_argument("--imgs",
                        default="C:/CIEDID_data/AbdnL/data",
                        help="Image directory")
    parser.add_argument("--masks",
                        default="C:/CIEDID_data/AbdnL/mask",
                        help="Mask directory")
    parser.add_argument("--img_size",    type=int,   default=512)
    parser.add_argument("--patch_size",  type=int,   default=320,
                        help="Used only for display info — not applied here "
                             "since we evaluate on full padded image.")
    parser.add_argument("--valid_split", type=float, default=0.2,
                        help="Must match the split used during training.")
    parser.add_argument("--thresholds",  type=int,   nargs="+",
                        default=[100, 250, 500, 1000, 1500, 2000, 2800, 4000],
                        help="List of pixel thresholds to evaluate.")
    parser.add_argument("--output_csv",  default=None,
                        help="Optional path to save results CSV.")
    parser.add_argument("--output_plot", default=None,
                        help="Optional path to save threshold curve PNG.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    weights_path = pathlib.Path(args.weights)
    img_dir      = pathlib.Path(args.imgs)
    mask_dir     = pathlib.Path(args.masks)

    # Build val set (same random_state=42 as training)
    val_df  = build_val_dataframe(img_dir, mask_dir, args.valid_split)

    # Load model
    learner = load_model(weights_path, args.img_size, args.patch_size,
                         val_df, device)

    # Evaluate
    results = evaluate_thresholds(learner, args.thresholds, device)

    # Print table
    print(f"\n{'='*75}")
    print(f"{'Threshold':>10} {'Sens':>8} {'PPV':>8} {'Spec':>8} "
          f"{'F1':>8} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
    print(f"{'-'*75}")
    for _, row in results.iterrows():
        print(f"{int(row['threshold']):>10,} "
              f"{row['sensitivity']:>8.3f} "
              f"{row['PPV']:>8.3f} "
              f"{row['specificity']:>8.3f} "
              f"{row['F1']:>8.3f} "
              f"{int(row['TP']):>4} "
              f"{int(row['FP']):>4} "
              f"{int(row['FN']):>4} "
              f"{int(row['TN']):>4}")
    print(f"{'='*75}")

    # Recommend threshold (best F1)
    best_row = results.loc[results["F1"].idxmax()]
    print(f"\n⭐ Best F1 threshold: {int(best_row['threshold']):,} px  "
          f"→  Sensitivity={best_row['sensitivity']:.3f}  "
          f"PPV={best_row['PPV']:.3f}  "
          f"F1={best_row['F1']:.3f}")

    # Clinical threshold (sensitivity ≥ 0.90)
    high_sens = results[results["sensitivity"] >= 0.90]
    if len(high_sens) > 0:
        clinical = high_sens.loc[high_sens["PPV"].idxmax()]
        print(f"🏥 Clinical threshold (sens≥0.90, best PPV): "
              f"{int(clinical['threshold']):,} px  "
              f"→  Sensitivity={clinical['sensitivity']:.3f}  "
              f"PPV={clinical['PPV']:.3f}")

    # Save CSV
    if args.output_csv:
        out_csv = pathlib.Path(args.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_csv, index=False)
        print(f"💾 Results saved → {out_csv}")

    # Save plot
    plot_path = pathlib.Path(args.output_plot) if args.output_plot \
                else weights_path.parent / "threshold_curve.png"
    plot_threshold_curve(results, plot_path)


if __name__ == "__main__":
    main()