"""
eval_abdn_threshold.py
======================
ทดสอบ sensitivity / PPV / specificity ของ abandoned lead detection
โดยใช้ทั้ง pixel count threshold และ probability threshold

Usage:
    python eval_abdn_threshold.py

Optional args:
    --weights      path to best_gen.pth or best_abdn.pth
    --imgs         image directory
    --masks        mask directory
    --img_size     512
    --valid_split  0.2
    --thresholds   100 250 500 1000 1500 2000 2800 4000
    --prob_thresholds  0.3 0.5 0.7 0.9
    --output_csv   results/threshold_eval.csv
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

# ── PyTorch 2.6+ fix ────────────────────────────────────────────────
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

CLASS_NAMES = ["background", "generator", "lead", "abandoned_lead"]
ABDN_CLASS  = 3


# ══════════════════════════════════════════════════════════════════════
# 1. Build validation dataframe
# ══════════════════════════════════════════════════════════════════════
def build_val_dataframe(img_dir, mask_dir, valid_split=0.2, random_state=42):
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
def load_model(weights_path, img_size, val_df, device):
    def get_x(r): return r["image"]
    def get_y(r): return r["mask"]

    BS_DUMMY = 4
    eval_df  = val_df.copy()
    eval_df["is_valid"] = True
    dummy = pd.concat([eval_df.iloc[[0]]] * BS_DUMMY, ignore_index=True)
    dummy["is_valid"] = False
    eval_df = pd.concat([dummy, eval_df], ignore_index=True)

    dblock = DataBlock(
        blocks    = (ImageBlock, MaskBlock(codes=CLASS_NAMES)),
        get_x     = get_x,
        get_y     = get_y,
        splitter  = ColSplitter(col="is_valid"),
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
    print(f"✅ Weights loaded: {weights_path.name}")
    return learner


# ══════════════════════════════════════════════════════════════════════
# 3. Collect per-image predictions (both argmax and softmax probs)
# ══════════════════════════════════════════════════════════════════════
def collect_predictions(learner, device):
    """
    Returns DataFrame with one row per image:
      - targ_px  : ground truth abandoned_lead pixel count
      - pred_px  : argmax predicted abandoned_lead pixel count (prob > 0.5)
      - prob_XXX : predicted pixel count at each probability threshold
    
    Softmax probability of class 3 per pixel is computed.
    A pixel counts as abandoned_lead if prob[:,3,:,:] > prob_threshold.
    This gives finer control than argmax (implicit threshold = 0.5).
    """
    PROB_THRESHOLDS = [0.3, 0.5, 0.7, 0.9]
    records = []

    with torch.no_grad():
        for xb, yb in learner.dls.valid:
            xb = xb.to(device)
            yb = yb.to(device)

            with torch.cuda.amp.autocast():
                logits = learner.model(xb)               # (B, 4, H, W)
                probs  = logits.softmax(dim=1)           # (B, 4, H, W)
                preds  = logits.argmax(dim=1)            # (B, H, W) argmax

            abdn_prob = probs[:, ABDN_CLASS, :, :]       # (B, H, W)

            for i in range(len(xb)):
                row = {
                    "targ_px": (yb[i] == ABDN_CLASS).sum().item(),
                    "pred_px": (preds[i] == ABDN_CLASS).sum().item(),  # argmax = prob>0.5
                }
                for pt in PROB_THRESHOLDS:
                    row[f"prob_{int(pt*100):02d}"] = (abdn_prob[i] > pt).sum().item()
                records.append(row)

    df = pd.DataFrame(records)
    print(f"\n📊 Images evaluated : {len(df)}")
    print(f"   GT abdn images   : {(df['targ_px'] > 0).sum()}")
    print(f"   Pred pixel stats (argmax): "
          f"min={df['pred_px'].min()}  "
          f"median={df['pred_px'].median():.0f}  "
          f"max={df['pred_px'].max()}")
    return df, PROB_THRESHOLDS


# ══════════════════════════════════════════════════════════════════════
# 4. Compute metrics for one (pixel_col, pixel_threshold) combination
# ══════════════════════════════════════════════════════════════════════
def compute_metrics(df_img, pixel_col, pixel_threshold):
    has_pred = df_img[pixel_col] > pixel_threshold
    has_targ = df_img["targ_px"] > 0

    tp = int(( has_pred &  has_targ).sum())
    fp = int(( has_pred & ~has_targ).sum())
    fn = int((~has_pred &  has_targ).sum())
    tn = int((~has_pred & ~has_targ).sum())

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1   = 2*sens*ppv / (sens+ppv) if (sens+ppv) > 0 else 0.0

    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "sensitivity": round(sens, 4),
            "PPV":         round(ppv,  4),
            "specificity": round(spec, 4),
            "F1":          round(f1,   4)}


# ══════════════════════════════════════════════════════════════════════
# 5. Evaluate all threshold combinations
# ══════════════════════════════════════════════════════════════════════
def evaluate_all(df_img, pixel_thresholds, prob_thresholds):
    """
    Evaluate every combination of:
      - prob_threshold : 0.3 / 0.5 (argmax) / 0.7 / 0.9
      - pixel_threshold: list from args
    """
    rows = []

    # argmax (equivalent to prob > 0.5)
    for pt in pixel_thresholds:
        m = compute_metrics(df_img, "pred_px", pt)
        rows.append({"prob_threshold": 0.50, "pixel_threshold": pt, **m})

    # explicit prob thresholds (skip 0.5 — already covered by argmax)
    for prob_t in prob_thresholds:
        if abs(prob_t - 0.5) < 1e-6:
            continue
        col = f"prob_{int(prob_t*100):02d}"
        if col not in df_img.columns:
            continue
        for pt in pixel_thresholds:
            m = compute_metrics(df_img, col, pt)
            rows.append({"prob_threshold": prob_t, "pixel_threshold": pt, **m})

    return pd.DataFrame(rows).sort_values(
        ["prob_threshold", "pixel_threshold"]).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
# 6. Print summary tables
# ══════════════════════════════════════════════════════════════════════
def print_results(results):
    prob_levels = results["prob_threshold"].unique()

    for prob_t in sorted(prob_levels):
        sub = results[results["prob_threshold"] == prob_t]
        label = "argmax (=0.50)" if abs(prob_t - 0.5) < 1e-6 else f"{prob_t:.2f}"
        print(f"\n{'='*78}")
        print(f"  Probability threshold = {label}")
        print(f"{'='*78}")
        print(f"{'Pixel thr':>10} {'Sens':>7} {'PPV':>7} {'Spec':>7} "
              f"{'F1':>7} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
        print(f"{'-'*78}")
        for _, row in sub.iterrows():
            print(f"{int(row['pixel_threshold']):>10,} "
                  f"{row['sensitivity']:>7.3f} "
                  f"{row['PPV']:>7.3f} "
                  f"{row['specificity']:>7.3f} "
                  f"{row['F1']:>7.3f} "
                  f"{int(row['TP']):>4} "
                  f"{int(row['FP']):>4} "
                  f"{int(row['FN']):>4} "
                  f"{int(row['TN']):>4}")

    # Best F1 overall
    best = results.loc[results["F1"].idxmax()]
    print(f"\n⭐ Best F1 overall: "
          f"prob={best['prob_threshold']:.2f}  "
          f"pixel={int(best['pixel_threshold']):,}  "
          f"→ Sens={best['sensitivity']:.3f}  "
          f"PPV={best['PPV']:.3f}  "
          f"F1={best['F1']:.3f}")

    # Clinical: sensitivity >= 0.90, best PPV
    high_sens = results[results["sensitivity"] >= 0.90]
    if len(high_sens) > 0:
        clinical = high_sens.loc[high_sens["PPV"].idxmax()]
        print(f"🏥 Clinical (sens≥0.90, best PPV): "
              f"prob={clinical['prob_threshold']:.2f}  "
              f"pixel={int(clinical['pixel_threshold']):,}  "
              f"→ Sens={clinical['sensitivity']:.3f}  "
              f"PPV={clinical['PPV']:.3f}  "
              f"F1={clinical['F1']:.3f}")
    else:
        print("⚠️  No combination achieved sensitivity ≥ 0.90")


# ══════════════════════════════════════════════════════════════════════
# 7. Plot
# ══════════════════════════════════════════════════════════════════════
def plot_results(results, output_path):
    prob_levels = sorted(results["prob_threshold"].unique())
    colors = {0.30: "orange", 0.50: "blue", 0.70: "green", 0.90: "red"}
    styles = {0.30: "--", 0.50: "-", 0.70: "-.", 0.90: ":"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Sensitivity vs pixel threshold
    ax = axes[0]
    for prob_t in prob_levels:
        sub   = results[results["prob_threshold"] == prob_t]
        label = f"prob>{prob_t:.2f}" if abs(prob_t-0.5) > 1e-6 else "argmax (prob>0.50)"
        ax.plot(sub["pixel_threshold"], sub["sensitivity"],
                marker="o", color=colors.get(prob_t, "grey"),
                linestyle=styles.get(prob_t, "-"), linewidth=2, label=label)
    ax.axhline(0.90, color="grey", linestyle=":", linewidth=1, label="Sens=0.90")
    ax.set_xlabel("Pixel threshold"); ax.set_ylabel("Sensitivity")
    ax.set_ylim(0, 1.05); ax.set_title("Sensitivity vs Pixel Threshold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Right: PPV vs pixel threshold
    ax = axes[1]
    for prob_t in prob_levels:
        sub   = results[results["prob_threshold"] == prob_t]
        label = f"prob>{prob_t:.2f}" if abs(prob_t-0.5) > 1e-6 else "argmax (prob>0.50)"
        ax.plot(sub["pixel_threshold"], sub["PPV"],
                marker="s", color=colors.get(prob_t, "grey"),
                linestyle=styles.get(prob_t, "-"), linewidth=2, label=label)
    ax.set_xlabel("Pixel threshold"); ax.set_ylabel("PPV (Precision)")
    ax.set_ylim(0, 1.05); ax.set_title("PPV vs Pixel Threshold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle("Abandoned Lead Detection — Threshold Analysis", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n📈 Plot saved → {output_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# 8. Main
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",
                        default="C:/CIEDID_data/AbdnL/models/best_gen.pth")
    parser.add_argument("--imgs",
                        default="C:/CIEDID_data/AbdnL/data")
    parser.add_argument("--masks",
                        default="C:/CIEDID_data/AbdnL/mask")
    parser.add_argument("--img_size",       type=int,   default=512)
    parser.add_argument("--valid_split",    type=float, default=0.2)
    parser.add_argument("--thresholds",     type=int,   nargs="+",
                        default=[100, 250, 500, 1000, 1500, 2000, 2800, 4000])
    parser.add_argument("--prob_thresholds",type=float, nargs="+",
                        default=[0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--output_csv",     default=None)
    parser.add_argument("--output_plot",    default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    weights_path = pathlib.Path(args.weights)
    img_dir      = pathlib.Path(args.imgs)
    mask_dir     = pathlib.Path(args.masks)

    val_df  = build_val_dataframe(img_dir, mask_dir, args.valid_split)
    learner = load_model(weights_path, args.img_size, val_df, device)

    df_img, prob_thresholds = collect_predictions(learner, device)
    results = evaluate_all(df_img, args.thresholds, args.prob_thresholds)

    print_results(results)

    if args.output_csv:
        out_csv = pathlib.Path(args.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_csv, index=False)
        print(f"💾 CSV saved → {out_csv}")

    plot_path = pathlib.Path(args.output_plot) if args.output_plot \
                else weights_path.parent / "threshold_curve.png"
    plot_results(results, plot_path)


if __name__ == "__main__":
    main()