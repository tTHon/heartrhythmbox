"""
manufHeatmaps.py
================
Heatmaps for the manufacturer classifier trained by finetuneManufV5.py.
Must sit in the same folder as finetuneManufV5.py (it imports from it).

TWO COMMANDS
------------
1) cm — confusion-matrix heatmaps from the saved out-of-fold predictions
   (reads <cv_dir>/fold_N/val_predictions_<phase>.csv; no model, no GPU needed)

   python manufHeatmaps.py cm --cv_dir C:/CIEDID_data/Manuf/models
   python manufHeatmaps.py cm --cv_dir C:/CIEDID_data/Manuf/models --per_fold

   → <cv_dir>/heatmap_confusion_pooled.png     counts | row-normalised (sensitivity)
   → <cv_dir>/heatmap_confusion_per_fold.png   (with --per_fold)

2) gradcam — Grad-CAM overlays for images of one fold's validation set
   (loads <cv_dir>/fold_N/manuf_finetuned.pkl)

   # most confident errors of fold 0 (true class vs predicted class side by side)
   python manufHeatmaps.py gradcam --cv_dir C:/CIEDID_data/Manuf/models --fold 0

   # only Abbott images that were misclassified, any fold
   python manufHeatmaps.py gradcam --fold 2 --cls Abbott --which misclassified

   # random correct predictions, or specific IDs
   python manufHeatmaps.py gradcam --fold 0 --which correct --max_n 8
   python manufHeatmaps.py gradcam --fold 0 --which ids --ids 123 456

   → <cv_dir>/fold_N/gradcam/gradcam_<which>[_<cls>].png

HOW TO READ GRAD-CAM (limits)
-----------------------------
  * Target layer = last residual stage of the ResNet-50 body (7x7 grid for a
    224x224 input), upsampled bilinearly — the maps are coarse by construction.
  * It shows where the logit of the chosen class is most sensitive, not proof
    that the network "uses" a specific anatomical or device feature.
  * "pred" panel = evidence for the predicted class; "true" panel = evidence for
    the labelled class (shown only when they differ). Red = high contribution.
  * Qualitative only; do not quote it as a validated localisation result.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ────────────────────────────────────────────────
# 1. CONFUSION-MATRIX HEATMAPS
# ────────────────────────────────────────────────

def _draw_cm(ax, cm: np.ndarray, row_labels, col_labels, title, normalise: bool):
    row_sum = cm.sum(axis=1, keepdims=True)
    data = cm / np.clip(row_sum, 1, None) if normalise else cm.astype(float)
    vmax = 1.0 if normalise else max(cm.max(), 1)
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = (f"{100 * data[i, j]:.0f}%\n({cm[i, j]}/{int(row_sum[i, 0])})"
                   if normalise else f"{cm[i, j]}")
            color = "white" if data[i, j] > 0.55 * vmax else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    return im


def cmd_cm(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import finetuneManufV5 as v5

    cv_dir = Path(a.cv_dir)
    name_to_idx = {v: k for k, v in v5.IDX_TO_CIED.items()}
    pooled = v5.pool_cv_predictions(cv_dir, a.n_splits, a.phase)

    def cm_of(df):
        yt = df["true"].map(name_to_idx).to_numpy()
        yp = df["pred"].map(name_to_idx).to_numpy()
        _, cm_df, f1 = v5.per_class_metrics(yt, yp)
        return cm_df.to_numpy()[: len(v5.PRESENT_IDX), :], f1, float((yt == yp).mean())

    cm, f1, acc = cm_of(pooled)
    rows = [f"{v5.IDX_TO_CIED[i]}" for i in v5.PRESENT_IDX]
    cols = [v5.IDX_TO_CIED[i] for i in range(v5.NUM_ORIG)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    _draw_cm(axes[0], cm, rows, cols, "Pooled out-of-fold predictions — counts", False)
    im = _draw_cm(axes[1], cm, rows, cols,
                  "Row-normalised (diagonal = sensitivity per class)", True)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(f"{a.n_splits}-fold CV, {a.phase}: n={len(pooled)}, accuracy={acc:.3f}, "
                 f"macro-F1={f1:.3f}", fontsize=11)
    fig.tight_layout()
    out = cv_dir / "heatmap_confusion_pooled.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved → {out}")

    if a.per_fold:
        fig, axes = plt.subplots(1, a.n_splits, figsize=(3.9 * a.n_splits, 4.2))
        for f, ax in zip(range(a.n_splits), np.atleast_1d(axes)):
            d = pooled[pooled["fold"] == f]
            cmf, f1f, accf = cm_of(d)
            _draw_cm(ax, cmf, rows, cols, f"fold {f}: acc {accf:.3f}, macro-F1 {f1f:.3f}", False)
        fig.tight_layout()
        out = cv_dir / "heatmap_confusion_per_fold.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"saved → {out}")


# ────────────────────────────────────────────────
# 2. GRAD-CAM
# ────────────────────────────────────────────────

class GradCAM:
    """Grad-CAM on the last residual stage of the fastai body (model[0][-1])."""

    def __init__(self, model):
        import torch
        self.torch = torch
        self.model = model.eval()
        self.layer = model[0][-1]
        self.acts = None
        self.grads = None
        self._h = self.layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        self.acts = out
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "grads", g))

    def remove(self):
        self._h.remove()

    def __call__(self, x, class_idx: int):
        """Returns (cam in [0,1] at input resolution, softmax probs over the 5 classes)."""
        import torch
        import torch.nn.functional as F

        self.model.zero_grad(set_to_none=True)
        x = x.clone().requires_grad_(True)       # guarantees a graph even if params are frozen
        with torch.enable_grad():
            logits = self.model(x)
            assert self.acts is not None and self.acts.ndim == 4, (
                "target layer did not return a 4-D feature map — check model[0][-1]")
            logits[0, class_idx].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        probs = torch.softmax(logits.detach().float(), dim=1)[0].cpu().numpy()
        return cam, probs


def cmd_gradcam(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from torchvision import transforms
    from PIL import Image
    import finetuneManufV5 as v5

    fold_dir = Path(a.cv_dir) / f"fold_{a.fold}"
    pred_csv = fold_dir / f"val_predictions_{a.phase}.csv"
    df = pd.read_csv(pred_csv)
    df["ID"] = df["ID"].astype(str)

    if a.cls:
        df = df[df["true"] == a.cls]
    if a.which == "misclassified":
        df = df[~df["correct"]].sort_values("p_pred", ascending=False)
    elif a.which == "correct":
        df = df[df["correct"]]
        df = df.sample(n=min(a.max_n, len(df)), random_state=a.seed)
    elif a.which == "ids":
        if not a.ids:
            raise SystemExit("--which ids needs --ids ID [ID ...]")
        df = df[df["ID"].isin([str(i) for i in a.ids])]
    df = df.head(a.max_n).reset_index(drop=True)
    if df.empty:
        raise SystemExit("No images match the selection.")
    print(f"{len(df)} image(s) selected from {pred_csv.name}")

    model_path = fold_dir / "manuf_finetuned.pkl"
    model, _, _ = v5.load_finetuned(str(model_path), a.model)
    model = model.to(v5.DEVICE).eval()
    cam_engine = GradCAM(model)

    # geometry identical to VAL_TF (without normalisation) for display
    disp_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((v5.IMG_SIZE + 32, v5.IMG_SIZE + 32)),
        transforms.CenterCrop(v5.IMG_SIZE),
    ])
    name_to_idx = {v: k for k, v in v5.IDX_TO_CIED.items()}
    jet = plt.get_cmap("jet")

    n = len(df)
    fig, axes = plt.subplots(n, 3, figsize=(10.5, 3.4 * n), squeeze=False)
    for r, row in df.iterrows():
        img = Image.open(row["image_path"]).convert("RGB")
        x = v5.VAL_TF(img).unsqueeze(0).to(v5.DEVICE)
        base = np.asarray(disp_tf(img)).astype(np.float32) / 255.0

        t_idx, p_idx = name_to_idx[row["true"]], name_to_idx[row["pred"]]
        targets = []
        if a.target in ("pred", "both"):
            targets.append(("pred", p_idx))
        if a.target in ("true", "both") and (t_idx != p_idx or a.target == "true"):
            targets.append(("true", t_idx))

        ax = axes[r, 0]
        ax.imshow(base, cmap="gray")
        ax.set_title(f"ID {row['ID']}\ntrue {row['true']} → pred {row['pred']} "
                     f"(p={row['p_pred']:.2f})", fontsize=9)
        ax.axis("off")

        for c in range(1, 3):
            axes[r, c].axis("off")
        for c, (kind, idx) in enumerate(targets, start=1):
            cam, probs = cam_engine(x, idx)
            overlay = np.clip(0.55 * base + 0.45 * jet(cam)[..., :3], 0, 1)
            axes[r, c].imshow(overlay)
            axes[r, c].set_title(f"Grad-CAM → {kind}: {v5.IDX_TO_CIED[idx]} "
                                 f"(p={probs[idx]:.2f})", fontsize=9)
    cam_engine.remove()

    fig.suptitle(f"Fold {a.fold} validation — {a.which}"
                 + (f", true class {a.cls}" if a.cls else "")
                 + "\nGrad-CAM, last ResNet-50 stage (coarse 7×7 map); qualitative only",
                 fontsize=10)
    fig.tight_layout()
    out_dir = fold_dir / "gradcam"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"gradcam_{a.which}{('_' + a.cls) if a.cls else ''}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved → {out}")


# ────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--cv_dir", default="C:/CIEDID_data/Manuf/models",
                        help="the --output_dir used by finetuneManufV5.py")
        sp.add_argument("--phase", default="phase2", choices=["phase1", "phase2"])
        sp.add_argument("--n_splits", type=int, default=5)

    c = sub.add_parser("cm", help="confusion-matrix heatmaps from saved CV predictions")
    common(c)
    c.add_argument("--per_fold", action="store_true", help="also draw one heatmap per fold")
    c.set_defaults(func=cmd_cm)

    g = sub.add_parser("gradcam", help="Grad-CAM overlays for one fold's validation images")
    common(g)
    g.add_argument("--fold", type=int, required=True)
    g.add_argument("--model", default="C:/CIEDID_data/pkl/classification_manuf.pkl",
                   help="original pretrained .pkl (architecture source)")
    g.add_argument("--which", default="misclassified", choices=["misclassified", "correct", "ids"])
    g.add_argument("--cls", default=None, help="keep only this TRUE class (BSX, Medtronic, Abbott, Biotronik)")
    g.add_argument("--ids", nargs="*", default=None)
    g.add_argument("--target", default="both", choices=["pred", "true", "both"])
    g.add_argument("--max_n", type=int, default=8)
    g.add_argument("--seed", type=int, default=42)
    g.set_defaults(func=cmd_gradcam)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()