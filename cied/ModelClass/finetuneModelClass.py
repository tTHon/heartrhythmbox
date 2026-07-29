"""
finetune_modelclass.py
========================
Stage IV: CIED specific-model classification, fine-tuned ResNet-50 on the
same 256x256 generator crops used for Stage III (crop_image/{ID}_crop.png).

WHY THIS IS DIFFERENT FROM finetune_manuf_v2.py:
  Our ModelClass vocabulary shares NO classes with classification_manuf.pkl's
  original vocab (that pkl predicts manufacturer, not specific model). There
  is nothing to remap and nothing to mask — the head must be freshly
  initialised at our own vocab size.

  What IS still worth reusing: the pretrained model's ENCODER (backbone).
  Per the project's established principle — transfer learning is
  layer-based, not class-based — the backbone's low/mid-level CXR features
  (edges, generator silhouette, texture) don't depend on what the head was
  originally trained to predict. This is the same logic already applied in
  finetuneCV.py (segmentation): decoder/head randomly initialised, encoder
  transferred.

  So: encoder-only transfer from --pretrained_path (defaults to
  classification_manuf.pkl), head randomly initialised at
  len(our ModelClass vocab, after rare-class grouping).

CLASS GROUPING (per REC-MURA.06 protocol, Stage IV):
  "Models with fewer than 20 images are grouped as 'others' to avoid
  overfitting." Threshold configurable via --min_class_count (default 20).
  Grouping is computed from the train/CV pool (FinalTest==0) only, then the
  same mapping is applied to the held-out test set for label consistency
  (the held-out set itself is still never used for training/threshold
  selection).

DATA CONTRACT (same as finetune_manuf_v2.py):
  - Labels: main_workbook.csv (ID, ModelClass, FinalTest, ...)
  - Images: crop_dir/{ID}_crop.png
  - FinalTest: 0 = train/CV pool, 1 = held-out pipeline test set (excluded
    from all training/CV here)
"""

import argparse
import pathlib
import platform
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from fastai.vision.all import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score

# ── Windows/Linux path-class fix (same as other scripts in the pipeline) ──
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# ── PyTorch 2.6+ fix ────────────────────────────────────────────────────
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

warnings.filterwarnings("ignore")


def parse_lr_arg(value):
    value = value.strip()
    if value.startswith("slice(") and value.endswith(")"):
        inner = value[len("slice("):-1]
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        return slice(float(parts[0]), float(parts[1]))
    return float(value)


# ==============================
# 1. LABEL / GROUPING / PATH HELPERS
# ==============================
def get_x(r): return r["image_path"]
def get_y(r): return r["ModelClass_grouped"]


def build_grouping_map(pool_df, min_count):
    """
    Classes with < min_count images in the train/CV pool get grouped into
    'Other'. Computed on the pool ONLY (never on the held-out test set),
    then applied to both — same convention Busch et al. and the protocol
    use (grouping decided from training-available data, not test data).
    """
    counts = pool_df["ModelClass"].value_counts()
    keep = set(counts[counts >= min_count].index)
    mapping = {c: (c if c in keep else "Other") for c in counts.index}
    n_grouped_classes = len(counts) - len(keep)
    n_grouped_images = int(counts[~counts.index.isin(keep)].sum())
    print(f"🗂️  Class grouping (min_count={min_count}): "
          f"{len(keep)} kept as-is, {n_grouped_classes} classes "
          f"({n_grouped_images} images) grouped into 'Other'")
    return mapping


def build_dataframe(args):
    df = pd.read_csv(args.labels_csv, usecols=["ID", "ModelClass", "FinalTest"])
    df["ID"] = df["ID"].astype(str)
    df["image_path"] = df["ID"].apply(lambda i: str((args.crop_dir / f"{i}_crop.png").resolve()))

    missing_img = ~df["image_path"].apply(lambda p: pathlib.Path(p).exists())
    if missing_img.sum():
        print(f"⚠️  {missing_img.sum()} rows missing image in {args.crop_dir} — dropped.")
    df = df.loc[~missing_img].reset_index(drop=True)

    missing_label = df["ModelClass"].isna()
    if missing_label.sum():
        print(f"⚠️  {missing_label.sum()} rows have no ModelClass label — dropped.")
        df = df.loc[~missing_label].reset_index(drop=True)

    test_df = df[df["FinalTest"] == 1].reset_index(drop=True)
    pool_df = df[df["FinalTest"] == 0].reset_index(drop=True)
    print(f"📂 Train/CV pool: {len(pool_df)}   Held-out pipeline test set (excluded): {len(test_df)}")

    # ── class grouping, computed from pool only ─────────────────────
    mapping = build_grouping_map(pool_df, args.min_class_count)
    pool_df["ModelClass_grouped"] = pool_df["ModelClass"].map(mapping)
    # apply same mapping to held-out test set; unseen classes (shouldn't
    # normally happen since pool should be a superset) fall back to 'Other'
    test_df["ModelClass_grouped"] = test_df["ModelClass"].map(lambda c: mapping.get(c, "Other"))

    test_df.to_csv(args.output_dir / "held_out_pipeline_test_set.csv", index=False)

    class_names = sorted(pool_df["ModelClass_grouped"].unique().tolist())
    print(f"Classes after grouping ({len(class_names)}): {class_names}")
    print("Pool distribution (grouped):")
    print(pool_df["ModelClass_grouped"].value_counts())

    if args.fold == -1:
        train_idx, valid_idx = train_test_split(
            pool_df.index, test_size=args.valid_split,
            stratify=pool_df["ModelClass_grouped"], random_state=42
        )
        pool_df["is_valid"] = False
        pool_df.loc[valid_idx, "is_valid"] = True
        print(f"📂 Single split: train={len(train_idx)}  val={len(valid_idx)}")
    else:
        if args.fold >= args.n_splits:
            raise ValueError(f"--fold {args.fold} out of range for --n_splits {args.n_splits}")
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
        splits = list(skf.split(pool_df, pool_df["ModelClass_grouped"]))
        train_idx, valid_idx = splits[args.fold]
        pool_df["is_valid"] = False
        pool_df.loc[pool_df.index[valid_idx], "is_valid"] = True
        print(f"📂 CV fold {args.fold+1}/{args.n_splits}: train={len(train_idx)}  val={len(valid_idx)}")

    return pool_df, class_names, mapping


def compute_class_weights(df, class_names):
    train_counts = df.loc[~df["is_valid"], "ModelClass_grouped"].value_counts()
    counts = np.array([train_counts.get(c, 0) for c in class_names], dtype=float)
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (len(class_names) * counts)
    print("Class weights (inverse frequency, from train split):")
    for c, w in zip(class_names, weights):
        print(f"  {c}: {w:.3f}")
    return torch.tensor(weights, dtype=torch.float32)


# ==============================
# 2. ENCODER-ONLY TRANSFER (layer-based, not class-based)
# ==============================
def load_pretrained_encoder(learner, path):
    """
    Load ONLY the backbone/body from a pretrained fastai classification
    Learner (e.g. classification_manuf.pkl). The head is intentionally left
    at its fresh random init because the class vocab has no overlap with
    ours. This mirrors the encoder-only transfer used for the segmentation
    model in finetuneCV.py.

    fastai cnn_learner models are nn.Sequential(body, head):
      model[0] = body/backbone (ResNet-50 conv layers)
      model[1] = head (adaptive pool + linear layers, class-count dependent)
    We only touch model[0].
    """
    device = next(learner.model.parameters()).device
    old = load_learner(path, cpu=True)
    old_body_state = old.model[0].state_dict()
    new_body_state = learner.model[0].state_dict()

    loaded = skipped = 0
    for k, v in old_body_state.items():
        if k in new_body_state and new_body_state[k].shape == v.shape:
            new_body_state[k] = v.to(device)
            loaded += 1
        else:
            skipped += 1
    learner.model[0].load_state_dict(new_body_state)
    print(f"✅ Encoder (backbone) transferred: {loaded} layers loaded, {skipped} skipped "
          f"(shape mismatch)")
    print(f"   Head (model[1]) left at fresh random init — vocab has no overlap with pretrained model.")
    return learner


# ==============================
# 3. MAIN
# ==============================
def finetune(args):
    out = pathlib.Path(args.output_dir) / (f"fold_{args.fold}" if args.fold >= 0 else "single_split")
    out.mkdir(parents=True, exist_ok=True)
    args.output_dir = pathlib.Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output dir: {out}")

    df, class_names, grouping_map = build_dataframe(args)
    weights_tensor = compute_class_weights(df, class_names)

    # ── save config + grouping map for reproducibility ──────────────
    import csv
    csv_path = out / "training_history.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["# TRAINING CONFIG", datetime.now().strftime("%Y-%m-%d %H:%M")])
        for k, v in {
            "img_size": args.img_size, "batch_size": args.batch_size,
            "epochs_head": args.epochs_head, "epochs_full": args.epochs_full,
            "backbone": "resnet50", "grad_accum": args.grad_accum,
            "lr_head": args.lr_head, "lr_full": args.lr_full,
            "pretrained_path": args.pretrained_path,
            "min_class_count": args.min_class_count,
            "classes": class_names,
        }.items():
            writer.writerow([f"# {k}", v])
        writer.writerow([])
    pd.Series(grouping_map, name="grouped_into").to_csv(out / "class_grouping_map.csv")

    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock(vocab=class_names)),
        get_x=get_x, get_y=get_y,
        splitter=ColSplitter(col='is_valid'),
        item_tfms=Resize(args.img_size, method='pad', pad_mode='zeros'),
        batch_tfms=[
            *aug_transforms(
                do_flip=True, flip_vert=False, max_rotate=15,
                min_zoom=0.9, max_zoom=1.1, max_lighting=0.15,
                max_warp=0.0, p_affine=0.75, p_lighting=0.5,
            ),
            Normalize.from_stats(*imagenet_stats),
        ],
    )
    dls = dblock.dataloaders(df, bs=args.batch_size, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    weights_tensor = weights_tensor.to(device)
    loss_func = CrossEntropyLossFlat(weight=weights_tensor)

    learner = cnn_learner(
        dls, resnet50, loss_func=loss_func,
        metrics=[accuracy, BalancedAccuracy(), F1Score(average='macro')],
        cbs=[CSVLogger(fname=str(out / "training_history.csv"), append=True)],
    ).to_fp16()

    if args.pretrained_path:
        print("\n📦 Loading domain-matched pretrained ENCODER only (head stays fresh) …")
        learner = load_pretrained_encoder(learner, args.pretrained_path)
    else:
        print("\n⚠️  No --pretrained_path given — using plain ImageNet-pretrained ResNet-50.")

    # Phase 1 — head only
    print("\n--- Phase 1: Training head (backbone frozen) ---")
    learner.freeze()
    learner.fit_one_cycle(args.epochs_head, args.lr_head,
                           cbs=GradientAccumulation(n_acc=args.grad_accum))

    # Phase 2 — full fine-tune with best-model checkpoint
    print("\n--- Phase 2: Full fine-tuning ---")
    learner.unfreeze()
    learner.path = out
    learner.model_dir = ""
    learner.fit_one_cycle(
        args.epochs_full, lr_max=args.lr_full,
        cbs=[GradientAccumulation(n_acc=args.grad_accum),
             SaveModelCallback(monitor='balanced_accuracy_score', fname='best_modelclass', with_opt=False)]
    )

    best_path = out / "best_modelclass.pth"
    if best_path.exists():
        learner.load(str(out / "best_modelclass"))
        print("🏆 Loaded best checkpoint (highest balanced accuracy).")

    # ── validation-fold report ───────────────────────────────────────
    print("\n📊 Validation fold performance:")
    preds, targs = learner.get_preds()
    pred_labels = preds.argmax(dim=1).numpy()
    targ_labels = targs.numpy()
    print(classification_report(targ_labels, pred_labels, target_names=class_names,
                                 digits=3, zero_division=0))
    cm = confusion_matrix(targ_labels, pred_labels)
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=class_names, columns=class_names))

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(class_names)), max(5, 0.5 * len(class_names))))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=90, ha='right', fontsize=7)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ── export ────────────────────────────────────────────────────
    weights_path = out / "modelclass_classifier_weights.pth"
    torch.save(learner.model.state_dict(), weights_path)
    with open(out / "class_names.txt", "w") as f:
        f.write("\n".join(class_names))
    print(f"\n✅ Weights saved → {weights_path}")
    print(f"✅ Class order saved → {out / 'class_names.txt'}")
    print(f"✅ Grouping map saved → {out / 'class_grouping_map.csv'}")


# ==============================
# 4. ENTRY POINT
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_csv", default="C:/CIEDID_data/main_workbook.csv")
    parser.add_argument("--crop_dir", default="C:/CIEDID_data/Images_crop")
    parser.add_argument("--output_dir", default="C:/CIEDID_data/ModelClass/models")
    parser.add_argument("--pretrained_path", default="C:/CIEDID_data/pkl/classification_model.pkl",
                         help="fastai Learner .pkl to transfer the ENCODER (backbone) from. "
                              "Head is always freshly initialised regardless — set to '' / None "
                              "to skip and use plain ImageNet-pretrained ResNet-50 instead.")

    parser.add_argument("--min_class_count", type=int, default=1,
                         help="Models with fewer than this many images in the train/CV pool "
                              "are grouped into 'Other' (per REC-MURA.06 Stage IV protocol).")

    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--valid_split", type=float, default=0.2)

    parser.add_argument("--epochs_head", type=int, default=8)
    parser.add_argument("--epochs_full", type=int, default=20)
    parser.add_argument("--lr_head", type=float, default=3e-3)
    parser.add_argument("--lr_full", type=parse_lr_arg, default="slice(1e-6, 1e-4)")

    parser.add_argument("--fold", type=int, default=0,
                         help="CV fold index (0-based). -1 = single 80/20 split.")
    parser.add_argument("--n_splits", type=int, default=5)

    args = parser.parse_args()
    args.labels_csv = pathlib.Path(args.labels_csv)
    args.crop_dir = pathlib.Path(args.crop_dir)
    args.output_dir = pathlib.Path(args.output_dir)
    if not args.pretrained_path:
        args.pretrained_path = None

    finetune(args)