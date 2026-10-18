"""
finetune_manuf_v4.py
=====================
Finetune classification_manuf.pkl on our CIED chest radiograph dataset.

CHANGES in V4 vs V3:
  - VAL_TF fixed to match TRAIN_TF scale: Resize(IMG_SIZE+32) -> CenterCrop(IMG_SIZE)
    (V2/V3 resized val directly to 224, i.e. ~12% different scale from the
    256 -> RandomCrop(224) used in training).
  - Checkpoint selection: macro-F1 over the 4 present classes (Vitatron
    excluded), tie-break = lower val loss. (V2/V3: val accuracy.)
  - Per-class report (precision, sensitivity, specificity, F1, support,
    Wilson 95% CI for sensitivity) + 5x5 confusion matrix printed per epoch
    summary line (macro-F1) and in full for the selected checkpoint of each
    phase; saved to <output_dir>/per_class_report_phase{1,2}[_fold{N}].csv
    and confusion_matrix_phase{1,2}[_fold{N}].csv.
  - Misclassified-image export: for the selected checkpoint of each phase,
    <output_dir>/val_predictions_<tag>.csv (every val image: ID, true, pred,
    correct, softmax probability of each of the 5 vocab classes, p_pred,
    p_true) and misclassified_val_<tag>.csv (errors only, sorted by p_pred
    descending = most confident errors first). Probabilities are the raw
    5-class softmax (Vitatron column NOT renormalised after masking).
  - NOTE: the RUN RESULTS block below was produced by V2 (acc-based
    selection, old VAL_TF). V4 numbers are NOT directly comparable.

CHANGES in V3 vs finetuneManufV2.py:
  - build_dataframe now REPORTS every dropped row instead of only a count:
      * missing crop images  -> printed (ID, Manuf, FinalTest, expected path)
                                and saved to  <output_dir>/missing_crop_images.csv
      * FinalTest not in {0,1} (e.g. NaN) -> printed and saved to
                                <output_dir>/bad_finaltest_rows.csv
        (V2 silently lost these: `== 1` / `== 0` both exclude NaN)
      * Manuf not in CIED_TO_ORIG_NAME -> IDs printed
      * orphan crops (file in crop_dir with no row in the CSV) -> printed
      * a row-accounting line: CSV rows = missing + bad FinalTest + unknown Manuf + pool + test
  - Everything else (model, vocab remap, Vitatron masking, 2-phase training)
    is UNCHANGED.

=====================================================================
RUN RESULTS — finetuneManufV2.py (single stratified split, --fold -1)
=====================================================================
Data
  - 5 rows had no crop image and were dropped.
  - Train/val pool = 1296; held-out pipeline test set = 319 (excluded).
    Manuscript cohort is 1298 pool + 324 test = 1622. 324 - 319 = 5, which
    fits the 5 missing images all being in the test set, BUT the pool is
    also 2 short (1296 vs 1298) with no warning printed. Likely cause:
    2 rows with FinalTest NaN (silently excluded by == 0 / == 1) or a
    CSV row-count mismatch. V3 now prints this explicitly — verify.
  - Split: train = 1036, val = 260 (80/20, stratified by Manuf).
  - Train distribution: BSX 499, Medtronic 469, Abbott 43, Biotronik 25
    (BSX + Medtronic = 93%). Vitatron absent -> weight 0 (masked).
  - Class weights (1/count): Biotronik 0.040, BSX 0.002, Medtronic 0.00213,
    Abbott 0.02326. (Weighted-mean CE normalises by sum of weights, so the
    absolute scale does not matter; only the ratios do.)

Phase 1 (head only, lr 1e-3, 10 ep)
  - Val acc 84.2% -> 92.7% (best at ep 10); train acc 94.7%.
  - Val loss fell steadily 0.524 -> 0.259; no overfitting yet.

Phase 2 (full unfreeze, lr 1e-5, 20 ep)
  - Best val acc 96.9% (ep 12; ep 19 tied at 96.9%). Plateau ~96.2-96.9%
    from ep 3 onward.
  - Train acc kept rising to ~98.4% while val loss rose again after its
    minimum at ep 6 (0.184 -> 0.24-0.30): mild overfitting. Accuracy gain
    after ep 3 is within noise.

Interpretation / caveats
  - 260 val images: one image = 0.38 pp, so 96.2% vs 96.9% = 2 images.
    Checkpoint choice by "best val acc" is therefore noise-driven; val loss
    (min at ep 6) or a tie-break on val loss would be more stable.
  - The saved model is the best-val-acc checkpoint, so val acc 96.9% is
    optimistically biased (selected on the same set). The unbiased estimate
    is the held-out pipeline test set, not this number.
  - Overall accuracy is dominated by BSX/Medtronic. Biotronik (~6 val
    images) and Abbott (~11 val images) are too few for the aggregate to
    show their performance; per-class sensitivity/specificity/F1 on val
    and on the test set are needed before claiming 96.9% for all four.
  - Compare with Busch et al. manufacturer accuracy 94.36% (95% CI
    90.93-96.84) — only after the held-out test evaluation, not val.

Original usage:
  python finetune_manuf_v3.py \
      --model      classification_manuf.pkl \
      --labels_csv C:/CIEDID_data/main_workbook.csv \
      --crop_dir   C:/CIEDID_data/Images_crop \
      --epochs 30 \
      --out    finetuned_manuf.pkl

Vocab เดิมของโมเดล (5 class, index 0-4):
  0: Biotronik
  1: Boston Scientific
  2: Medtronic
  3: ST. Jude Medical
  4: Vitatron   ← ไม่มีในข้อมูลเรา
CSV "BSX" → "Boston Scientific"(1); "Abbott" → "ST. Jude Medical"(3);
"Medtronic" → (2); "Biotronik" → (0)
"""

import argparse
import copy
import time
import platform
import pathlib
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from fastai.vision.all import load_learner
from PIL import Image
import pickle
import numpy as np
from tqdm import tqdm

# Windows/Linux path-class fix (pretrained .pkl may be pickled on another OS)
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# PyTorch 2.6+ fix: torch.load weights_only=False for fastai Learner pickles
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix

# ────────────────────────────────────────────────
# VOCAB — must match the pretrained model exactly
# ────────────────────────────────────────────────
ORIG_VOCAB = {
    "Biotronik":         0,
    "Boston Scientific": 1,
    "Medtronic":         2,
    "ST. Jude Medical":  3,
    "Vitatron":          4,   # absent in our data → weight=0
}
NUM_ORIG = len(ORIG_VOCAB)   # 5 — head size never changes

CIED_TO_ORIG_NAME = {
    "Biotronik": "Biotronik",
    "BSX":       "Boston Scientific",
    "Medtronic": "Medtronic",
    "Abbott":    "ST. Jude Medical",
}
OUR_CLASSES    = list(CIED_TO_ORIG_NAME.keys())
ABSENT_CLASSES = ["Vitatron"]

ENCODER_PREFIX = "0"

IMG_SIZE     = 224
BATCH_SIZE   = 16
NUM_WORKERS  = 2

P1_EPOCHS    = 10
P1_LR        = 1e-3
P2_EPOCHS    = 20
P2_LR        = 1e-5
WEIGHT_DECAY = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ────────────────────────────────────────────────
# AUGMENTATION  (CXR-specific) — unchanged
# ────────────────────────────────────────────────
TRAIN_TF = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TF = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),   # same scale as TRAIN_TF
    transforms.CenterCrop(IMG_SIZE),                      # deterministic counterpart of RandomCrop
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ────────────────────────────────────────────────
# MODEL LOADING  (head preserved, unchanged)
# ────────────────────────────────────────────────

def load_pkl(path: str):
    """classification_manuf.pkl is a fastai-exported Learner → needs load_learner()."""
    return load_learner(path, cpu=(DEVICE == "cpu"))


def extract_pytorch_model(obj) -> nn.Module:
    if isinstance(obj, nn.Module):
        return obj
    if hasattr(obj, "model"):
        return obj.model
    raise ValueError(f"ไม่รู้จัก object type: {type(obj)}")


def verify_head(model: nn.Module):
    """ตรวจสอบว่า output layer มี NUM_ORIG neurons"""
    head = model[1]
    last_linear = None
    for layer in head:
        if isinstance(layer, nn.Linear):
            last_linear = layer
    assert last_linear is not None, "ไม่พบ Linear layer ใน head"
    assert last_linear.out_features == NUM_ORIG, (
        f"output size ไม่ match: {last_linear.out_features} != {NUM_ORIG}\n"
        f"ถ้า vocab เปลี่ยน ให้แก้ ORIG_VOCAB ด้านบน"
    )
    print(f"  head verified: output={last_linear.out_features} class  "
          f"(in_features={last_linear.in_features})")


def set_grad(model: nn.Module, phase: int):
    if phase == 1:
        frozen = trainable = 0
        for name, param in model.named_parameters():
            if name.startswith(ENCODER_PREFIX + "."):
                param.requires_grad = False
                frozen += 1
            else:
                param.requires_grad = True
                trainable += 1
        print(f"  Phase 1 | frozen={frozen}  trainable={trainable} params")
    else:
        for param in model.parameters():
            param.requires_grad = True
        print(f"  Phase 2 | unfreeze ทั้งหมด")


# ────────────────────────────────────────────────
# DATASET
# ────────────────────────────────────────────────

class CIEDManufDataset(Dataset):
    """(image_path, Manuf) rows → pretrained vocab index via CIED_TO_ORIG_NAME → ORIG_VOCAB."""
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["image_path"]).convert("RGB")
        img = self.transform(img)
        orig_name = CIED_TO_ORIG_NAME[row["Manuf"]]
        label = ORIG_VOCAB[orig_name]
        return img, label


def _print_rows(title: str, sub: pd.DataFrame, cols, max_rows: int = 50):
    """Print the offending rows (capped) so nothing is dropped silently."""
    print(f"  {title}: {len(sub)}")
    for _, r in sub.head(max_rows).iterrows():
        print("     " + "  ".join(f"{c}={r[c]}" for c in cols))
    if len(sub) > max_rows:
        print(f"     ... +{len(sub) - max_rows} more (see CSV)")


def build_dataframe(args):
    """
    Load main_workbook.csv, attach image paths, report + drop problem rows,
    split off the held-out pipeline test set (FinalTest == 1), then
    stratified-split the remaining pool into train/val.
    """
    df = pd.read_csv(args.labels_csv, usecols=["ID", "Manuf", "FinalTest"])
    n_csv_rows = len(df)
    df["ID"] = df["ID"].astype(str)
    df["image_path"] = df["ID"].apply(lambda i: str((args.crop_dir / f"{i}_crop.png").resolve()))
    print(f"\n── Row accounting ──────────────────────────────")
    print(f"  CSV rows read: {n_csv_rows}")

    # ── 1. Missing crop images ───────────────────────────────────────
    missing_mask = ~df["image_path"].apply(lambda p: Path(p).exists())
    n_missing = int(missing_mask.sum())
    if n_missing:
        miss_df = df.loc[missing_mask, ["ID", "Manuf", "FinalTest", "image_path"]] \
                    .rename(columns={"image_path": "expected_path"})
        _print_rows("⚠️  Missing crop images (dropped)", miss_df,
                    ["ID", "Manuf", "FinalTest", "expected_path"])
        print(f"     by FinalTest: {miss_df['FinalTest'].value_counts(dropna=False).to_dict()}")
        miss_csv = args.output_dir / "missing_crop_images.csv"
        miss_df.to_csv(miss_csv, index=False)
        print(f"     saved → {miss_csv}")
        df = df.loc[~missing_mask].reset_index(drop=True)

    # ── 2. Orphan crops (file exists, no CSV row) ───────────────────
    csv_ids = set(pd.read_csv(args.labels_csv, usecols=["ID"])["ID"].astype(str))
    crop_ids = {p.stem[:-5] for p in args.crop_dir.glob("*_crop.png")}
    orphans = sorted(crop_ids - csv_ids)
    if orphans:
        print(f"  ⚠️  Orphan crops (in {args.crop_dir.name}, not in CSV): {len(orphans)}")
        print("     " + ", ".join(orphans[:50]) + (" ..." if len(orphans) > 50 else ""))

    # ── 3. FinalTest not in {0,1} (NaN etc.) — V2 dropped these silently
    bad_ft_mask = ~df["FinalTest"].isin([0, 1])
    n_bad_ft = int(bad_ft_mask.sum())
    if n_bad_ft:
        bad_df = df.loc[bad_ft_mask, ["ID", "Manuf", "FinalTest", "image_path"]]
        _print_rows("⚠️  FinalTest not 0/1 (dropped)", bad_df, ["ID", "Manuf", "FinalTest"])
        bad_csv = args.output_dir / "bad_finaltest_rows.csv"
        bad_df.to_csv(bad_csv, index=False)
        print(f"     saved → {bad_csv}")
        df = df.loc[~bad_ft_mask].reset_index(drop=True)

    # ── 4. Unknown manufacturer ─────────────────────────────────────
    unknown_mask = ~df["Manuf"].isin(CIED_TO_ORIG_NAME.keys())
    n_unknown = int(unknown_mask.sum())
    if n_unknown:
        unk_df = df.loc[unknown_mask, ["ID", "Manuf", "FinalTest"]]
        _print_rows("⚠️  Manuf not in CIED_TO_ORIG_NAME (dropped)", unk_df,
                    ["ID", "Manuf", "FinalTest"])
        df = df.loc[~unknown_mask].reset_index(drop=True)

    test_df = df[df["FinalTest"] == 1].reset_index(drop=True)
    pool_df = df[df["FinalTest"] == 0].reset_index(drop=True)
    test_df.to_csv(args.output_dir / "held_out_pipeline_test_set.csv", index=False)

    print(f"  = missing {n_missing} + bad FinalTest {n_bad_ft} + unknown Manuf {n_unknown} "
          f"+ pool {len(pool_df)} + test {len(test_df)} "
          f"= {n_missing + n_bad_ft + n_unknown + len(pool_df) + len(test_df)} "
          f"(CSV rows {n_csv_rows})")
    print(f"────────────────────────────────────────────────")
    print(f"📂 Train/val pool: {len(pool_df)}   Held-out pipeline test set (excluded): {len(test_df)}")

    if args.fold == -1:
        train_df, val_df = train_test_split(
            pool_df, test_size=args.valid_split,
            stratify=pool_df["Manuf"], random_state=42
        )
        print(f"📂 Single split: train={len(train_df)}  val={len(val_df)}")
    else:
        if args.fold >= args.n_splits:
            raise ValueError(f"--fold {args.fold} out of range for --n_splits {args.n_splits}")
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
        splits = list(skf.split(pool_df, pool_df["Manuf"]))
        train_idx, val_idx = splits[args.fold]
        train_df = pool_df.iloc[train_idx]
        val_df = pool_df.iloc[val_idx]
        print(f"📂 CV fold {args.fold+1}/{args.n_splits}: train={len(train_df)}  val={len(val_df)}")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def build_loaders(args):
    train_df, val_df = build_dataframe(args)

    print(f"\n  train: {len(train_df)} ภาพ  |  val: {len(val_df)} ภาพ")
    print(f"  train Manuf distribution:\n{train_df['Manuf'].value_counts()}")
    print(f"  val Manuf distribution:\n{val_df['Manuf'].value_counts()}")

    train_ds = CIEDManufDataset(train_df, TRAIN_TF)
    val_ds   = CIEDManufDataset(val_df, VAL_TF)

    # class weight — size=NUM_ORIG, Vitatron=0 (masked)
    remapped_targets = [ORIG_VOCAB[CIED_TO_ORIG_NAME[m]] for m in train_df["Manuf"]]
    class_counts = np.bincount(remapped_targets, minlength=NUM_ORIG).astype(float)
    class_weights = torch.zeros(NUM_ORIG)
    for cied_name, orig_name in CIED_TO_ORIG_NAME.items():
        idx = ORIG_VOCAB[orig_name]
        class_weights[idx] = 1.0 / class_counts[idx] if class_counts[idx] > 0 else 0.0

    print(f"\n  class counts : { {k: int(class_counts[ORIG_VOCAB[v]]) for k, v in CIED_TO_ORIG_NAME.items()} }")
    print(f"  class weights: { {k: round(class_weights[ORIG_VOCAB[v]].item(), 5) for k, v in CIED_TO_ORIG_NAME.items()} }")
    print(f"  Vitatron (absent) weight: {class_weights[ORIG_VOCAB['Vitatron']].item()}  ← masked out")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, class_weights, val_df


# ────────────────────────────────────────────────
# TRAIN / EVAL — unchanged
# ────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, is_train: bool):
    model.train() if is_train else model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels, all_probs = [], [], []

    desc = "Train" if is_train else "Valid"
    pbar = tqdm(loader, desc=f"  {desc:>5}", leave=False)

    with torch.set_grad_enabled(is_train):
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss    = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds   = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += imgs.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(torch.softmax(outputs.detach().float(), dim=1).cpu())

    return (total_loss / total, correct / total,
            torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy(),
            torch.cat(all_probs).numpy())


# Indices (in the pretrained vocab) of the classes actually present in our data.
PRESENT_IDX   = [ORIG_VOCAB[CIED_TO_ORIG_NAME[c]] for c in OUR_CLASSES]
IDX_TO_CIED   = {ORIG_VOCAB[CIED_TO_ORIG_NAME[c]]: c for c in OUR_CLASSES}
IDX_TO_CIED[ORIG_VOCAB["Vitatron"]] = "Vitatron(absent)"


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a proportion k/n (returns (nan, nan) if n == 0)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def per_class_metrics(y_true, y_pred):
    """
    Per-class precision / sensitivity / specificity / F1 from a 5x5 confusion
    matrix (full pretrained vocab, so predictions into the absent Vitatron
    column are counted as errors of the true class). Macro-F1 is the mean F1
    over the 4 PRESENT classes only.
    Returns (report_df, cm_df, macro_f1).
    """
    all_idx = list(range(NUM_ORIG))
    cm = confusion_matrix(y_true, y_pred, labels=all_idx)
    total = cm.sum()
    rows = []
    for c in PRESENT_IDX:
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        n_pos = tp + fn
        sens = tp / n_pos if n_pos else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1   = 2 * prec * sens / (prec + sens) if (prec + sens) and n_pos else 0.0
        lo, hi = wilson_ci(int(tp), int(n_pos))
        rows.append({
            "class": IDX_TO_CIED[c], "support": int(n_pos),
            "precision": prec, "sensitivity": sens,
            "sens_CI95_low": lo, "sens_CI95_high": hi,
            "specificity": spec, "F1": f1,
        })
    report = pd.DataFrame(rows)
    macro_f1 = float(report["F1"].mean())
    names = [IDX_TO_CIED[i] for i in all_idx]
    cm_df = pd.DataFrame(cm, index=[f"true_{n}" for n in names],
                         columns=[f"pred_{n}" for n in names])
    return report, cm_df, macro_f1


def print_report(report, cm_df, macro_f1, title):
    print(f"\n  ── {title} ──")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 140):
        print(report.to_string(index=False))
        print(f"\n  macro-F1 (4 present classes): {macro_f1:.3f}")
        print("\n  confusion matrix (rows=true, cols=pred):")
        print(cm_df.to_string())


def export_predictions(val_df, y_true, y_pred, probs, out_dir: Path, tag: str):
    """
    Save per-image val predictions and the misclassified subset.
    Relies on val_loader having shuffle=False, so row i of the outputs is
    row i of val_df; this is verified against the CSV Manuf labels.
    """
    assert len(val_df) == len(y_true), "val_df / predictions length mismatch"
    expected = np.array([ORIG_VOCAB[CIED_TO_ORIG_NAME[m]] for m in val_df["Manuf"]])
    assert (expected == y_true).all(), "val order mismatch — loader must not shuffle"

    out = val_df[["ID", "Manuf", "image_path"]].reset_index(drop=True).copy()
    out["true"] = [IDX_TO_CIED[i] for i in y_true]
    out["pred"] = [IDX_TO_CIED[i] for i in y_pred]
    out["correct"] = (y_true == y_pred)
    for i in range(NUM_ORIG):
        out[f"p_{IDX_TO_CIED[i]}"] = probs[:, i]
    out["p_pred"] = probs[np.arange(len(y_pred)), y_pred]
    out["p_true"] = probs[np.arange(len(y_true)), y_true]

    out.to_csv(out_dir / f"val_predictions_{tag}.csv", index=False)
    wrong = out[~out["correct"]].sort_values("p_pred", ascending=False)
    wrong.to_csv(out_dir / f"misclassified_val_{tag}.csv", index=False)

    print(f"\n  Misclassified val images: {len(wrong)}/{len(out)}")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 140):
        print(wrong[["ID", "true", "pred", "p_pred", "p_true"]].to_string(index=False))
    print(f"  saved → val_predictions_{tag}.csv, misclassified_val_{tag}.csv")


def train_phase(model, train_loader, val_loader, criterion,
                lr: float, epochs: int, phase_name: str,
                out_dir: Path, tag: str, val_df: pd.DataFrame):
    """
    Checkpoint selection: highest val macro-F1 (4 present classes);
    ties broken by LOWER val loss. Val accuracy is shown but not used.
    """
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_key   = (-1.0, float("-inf"))     # (macro_f1, -val_loss)
    best_f1    = 0.0
    best_state = None
    best_ep    = 0

    print(f"\n{'─'*70}")
    print(f"  {phase_name}  |  lr={lr}  epochs={epochs}  |  select: macro-F1 (tie: val loss)")
    print(f"{'─'*70}")
    print(f"  {'Ep':>3}  {'TrainLoss':>10}  {'TrainAcc':>9}  {'ValLoss':>9}  {'ValAcc':>8}  {'MacroF1':>8}")

    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, _, _, _ = run_epoch(model, train_loader, criterion, optimizer, True)
        vl_loss, vl_acc, vp, vt, _ = run_epoch(model, val_loader, criterion, optimizer, False)
        scheduler.step()
        _, _, f1 = per_class_metrics(vt, vp)

        key = (round(f1, 6), -vl_loss)
        is_best = key > best_key
        mark = " ★" if is_best else ""
        print(f"  {ep:>3}  {tr_loss:>10.4f}  {tr_acc:>8.1%}  "
              f"{vl_loss:>9.4f}  {vl_acc:>7.1%}  {f1:>8.3f}{mark}  ({time.time()-t0:.1f}s)")

        if is_best:
            best_key, best_f1, best_ep = key, f1, ep
            best_state = copy.deepcopy(model.state_dict())

    print(f"\n  selected checkpoint ({phase_name}): epoch {best_ep}, val macro-F1 {best_f1:.3f}")
    model.load_state_dict(best_state)

    # Full per-class report for the selected checkpoint
    _, _, vp, vt, vprob = run_epoch(model, val_loader, criterion, optimizer, False)
    report, cm_df, macro_f1 = per_class_metrics(vt, vp)
    print_report(report, cm_df, macro_f1, f"{phase_name} — selected checkpoint (val set)")
    report.to_csv(out_dir / f"per_class_report_{tag}.csv", index=False)
    cm_df.to_csv(out_dir / f"confusion_matrix_{tag}.csv")
    print(f"  saved → per_class_report_{tag}.csv, confusion_matrix_{tag}.csv")
    export_predictions(val_df, vt, vp, vprob, out_dir, tag)
    return model


# ────────────────────────────────────────────────
# SAVE / LOAD — unchanged
# ────────────────────────────────────────────────

def save_model(model: nn.Module, out_path: str):
    payload = {
        "model_state_dict": model.state_dict(),
        "orig_vocab":        ORIG_VOCAB,
        "cied_to_orig_name": CIED_TO_ORIG_NAME,
        "our_classes":       OUR_CLASSES,
        "num_orig":          NUM_ORIG,
        "img_size":          IMG_SIZE,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n✅ บันทึกโมเดลที่ {out_path}")


def load_finetuned(finetuned_pkl: str, original_pkl: str):
    """
        model, vocab, classes = load_finetuned("finetuned_manuf.pkl", "classification_manuf.pkl")
        idx_to_class = {v: k for k, v in vocab.items()}
    """
    with open(finetuned_pkl, "rb") as f:
        payload = pickle.load(f)
    base_obj = load_pkl(original_pkl)
    model    = extract_pytorch_model(base_obj)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload["orig_vocab"], payload["our_classes"]


# ────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────

def main():
    global BATCH_SIZE
    parser = argparse.ArgumentParser(
        description="Finetune manufacturer classifier (5-class head preserved, Vitatron masked)"
    )
    parser.add_argument("--model",      default="C:/CIEDID_data/pkl/classification_manuf.pkl")
    parser.add_argument("--labels_csv", default="C:/CIEDID_data/main_workbook.csv")
    parser.add_argument("--crop_dir",   default="C:/CIEDID_data/Images_crop")
    parser.add_argument("--output_dir", default="C:/CIEDID_data/Manuf/models")
    parser.add_argument("--epochs",  type=int, default=P1_EPOCHS + P2_EPOCHS)
    parser.add_argument("--p1_frac", type=float,
                        default=P1_EPOCHS / (P1_EPOCHS + P2_EPOCHS))
    parser.add_argument("--out",     default="C:/CIEDID_data/Manuf/manuf_finetuned.pkl")
    parser.add_argument("--batch",   type=int, default=BATCH_SIZE)
    parser.add_argument("--valid_split", type=float, default=0.2)
    parser.add_argument("--fold", type=int, default=-1,
                        help="CV fold index (0-based). -1 = single stratified split (default).")
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    import random
    def set_seed(seed=42):
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    args.labels_csv = Path(args.labels_csv)
    args.crop_dir   = Path(args.crop_dir)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    BATCH_SIZE = args.batch

    p1_ep = max(1, int(args.epochs * args.p1_frac))
    p2_ep = max(1, args.epochs - p1_ep)

    print(f"\n{'='*58}")
    print(f"  Finetune Manufacturer Classifier")
    print(f"  device       : {DEVICE}")
    print(f"  our classes  : {OUR_CLASSES}")
    print(f"  absent class : {ABSENT_CLASSES}  (weight=0 → masked)")
    print(f"  phase1={p1_ep}ep (head only)  phase2={p2_ep}ep (full)")
    print(f"{'='*58}")

    print(f"\n[1/5] โหลดโมเดล: {args.model}")
    base_obj = load_pkl(args.model)
    model    = extract_pytorch_model(base_obj)
    verify_head(model)
    model    = model.to(DEVICE)

    print(f"\n[2/5] โหลด dataset จาก {args.labels_csv} + {args.crop_dir}")
    train_loader, val_loader, class_weights, val_df = build_loaders(args)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    fold_tag = f"_fold{args.fold}" if args.fold >= 0 else ""
    print(f"\n[3/5] Phase 1: train head เท่านั้น")
    set_grad(model, phase=1)
    model = train_phase(model, train_loader, val_loader, criterion,
                        lr=P1_LR, epochs=p1_ep,
                        phase_name="Phase 1 (head only)",
                        out_dir=args.output_dir, tag=f"phase1{fold_tag}", val_df=val_df)

    print(f"\n[4/5] Phase 2: unfreeze ทั้งหมด")
    set_grad(model, phase=2)
    model = train_phase(model, train_loader, val_loader, criterion,
                        lr=P2_LR, epochs=p2_ep,
                        phase_name="Phase 2 (full finetune)",
                        out_dir=args.output_dir, tag=f"phase2{fold_tag}", val_df=val_df)

    print(f"\n[5/5] บันทึกโมเดล")
    save_model(model, args.out)


if __name__ == "__main__":
    main()