"""
finetune_manuf_v2.py
=====================
Finetune classification_manuf.pkl on our CIED chest radiograph dataset.

CHANGES vs fineTuneManuf.py (previous version):
  - Data source switched from ImageFolder (train/val folders per class) to
    main_workbook.csv + crop_image/{ID}_crop.png, matching how the rest of
    the pipeline (segmentation, Stage III) sources its data.
  - Held-out pipeline test set (FinalTest == 1, n=324) is split off BEFORE
    train/val splitting and is never touched here — saved to a CSV for the
    separate Stage V end-to-end evaluation, same convention as
    finetune_manuf_classifier.py.
  - Train/val split is now a stratified split by Manuf (sklearn), computed
    from the FinalTest == 0 pool only.
  - Original strategy is otherwise UNCHANGED:
      - head kept at all 5 original classes, never replaced
      - our 4 manufacturers remapped into the pretrained model's original
        vocab indices (folder/CSV names differ from pretrained names, see
        CIED_TO_ORIG_NAME below)
      - Vitatron (absent in our data) stays in the head but is masked with
        class weight = 0 in the loss
      - 2-phase training: head-only, then full unfreeze at low lr

Vocab เดิมของโมเดล (5 class, index 0-4):
  0: Biotronik
  1: Boston Scientific
  2: Medtronic
  3: ST. Jude Medical
  4: Vitatron   ← ไม่มีในข้อมูลเรา

Our CSV uses different manufacturer strings than the pretrained model's
vocab, so an extra name-remap layer (CIED_TO_ORIG_NAME) sits in front of
ORIG_VOCAB:
  CSV "BSX"       → pretrained "Boston Scientific" → idx 1
  CSV "Abbott"    → pretrained "ST. Jude Medical"   → idx 3
  CSV "Medtronic" → pretrained "Medtronic"          → idx 2
  CSV "Biotronik" → pretrained "Biotronik"          → idx 0

วิธีใช้:
  python finetune_manuf_v2.py \
      --model      classification_manuf.pkl \
      --labels_csv C:/CIEDID_data/main_workbook.csv \
      --crop_dir   C:/CIEDID_data/crop_image \
      --epochs 30 \
      --out    finetuned_manuf.pkl
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

 
# Windows/Linux path-class fix: the pretrained .pkl may have been pickled on
# a different OS than the one loading it now (e.g. saved with PosixPath on
# Linux, loaded on Windows). Without this, unpickling raises
# "cannot instantiate 'PosixPath' on your system". Same fix as finetuneCV.py
# and genCrop_batch.py.
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# PyTorch 2.6+ fix: monkey-patch torch.load to use weights_only=False for
# fastai compatibility (fastai Learner pickles need full unpickling, not the
# restricted weights-only mode that became default in torch 2.6)
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

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

# Our CSV's Manuf strings → pretrained model's vocab names
CIED_TO_ORIG_NAME = {
    "Biotronik": "Biotronik",
    "BSX":       "Boston Scientific",
    "Medtronic": "Medtronic",
    "Abbott":    "ST. Jude Medical",
}
OUR_CLASSES    = list(CIED_TO_ORIG_NAME.keys())   # CSV-side names, for reporting
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
# AUGMENTATION  (CXR-specific) — unchanged from previous version
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
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ────────────────────────────────────────────────
# MODEL LOADING  (head preserved, unchanged)
# ────────────────────────────────────────────────

def load_pkl(path: str):
    """
    classification_manuf.pkl is a fastai-exported Learner (torch.save-based,
    with persistent_id references for tensor storage) — same format as
    segmentation.pkl elsewhere in the pipeline. Plain pickle.load() cannot
    open this; fastai's load_learner() is required.
    """
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
# DATASET — CSV + crop_image/{ID}_crop.png, remapped to ORIG_VOCAB index
# ────────────────────────────────────────────────

class CIEDManufDataset(Dataset):
    """
    Reads (image_path, Manuf) rows from a dataframe already filtered to
    existing images, remaps the CSV manufacturer string to the pretrained
    model's vocab index via CIED_TO_ORIG_NAME → ORIG_VOCAB, and applies the
    given torchvision transform.
    """
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


def build_dataframe(args):
    """
    Load main_workbook.csv, attach image paths, drop rows with missing
    images/labels or manufacturers outside CIED_TO_ORIG_NAME, split off the
    held-out pipeline test set (FinalTest == 1), then stratified-split the
    remaining pool into train/val.
    """
    df = pd.read_csv(args.labels_csv, usecols=[
        "ID", "Manuf", "FinalTest"
    ])
    df["ID"] = df["ID"].astype(str)
    df["image_path"] = df["ID"].apply(lambda i: str((args.crop_dir / f"{i}_crop.png").resolve()))

    missing_img = ~df["image_path"].apply(lambda p: Path(p).exists())
    if missing_img.sum():
        print(f"⚠️  {missing_img.sum()} rows missing image in {args.crop_dir} — dropped.")
    df = df.loc[~missing_img].reset_index(drop=True)

    unknown_manuf = ~df["Manuf"].isin(CIED_TO_ORIG_NAME.keys())
    if unknown_manuf.sum():
        print(f"⚠️  {unknown_manuf.sum()} rows have a Manuf value not in "
              f"CIED_TO_ORIG_NAME ({sorted(df.loc[unknown_manuf, 'Manuf'].unique())}) — dropped.")
        df = df.loc[~unknown_manuf].reset_index(drop=True)

    test_df = df[df["FinalTest"] == 1].reset_index(drop=True)
    pool_df = df[df["FinalTest"] == 0].reset_index(drop=True)
    test_df.to_csv(args.output_dir / "held_out_pipeline_test_set.csv", index=False)
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

    train_ds = CIEDManufDataset(train_df, TRAIN_TF)
    val_ds   = CIEDManufDataset(val_df, VAL_TF)

    # class weight — size=NUM_ORIG, Vitatron=0 (same masking strategy as before)
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

    return train_loader, val_loader, class_weights


# ────────────────────────────────────────────────
# TRAIN / EVAL — unchanged from previous version
# ────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, is_train: bool):
    model.train() if is_train else model.eval()
    total_loss = correct = total = 0

    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
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

    return total_loss / total, correct / total


def train_phase(model, train_loader, val_loader, criterion,
                lr: float, epochs: int, phase_name: str):

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_state   = None

    print(f"\n{'─'*58}")
    print(f"  {phase_name}  |  lr={lr}  epochs={epochs}")
    print(f"{'─'*58}")
    print(f"  {'Ep':>3}  {'TrainLoss':>10}  {'TrainAcc':>9}  {'ValLoss':>9}  {'ValAcc':>8}")

    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer, False)
        scheduler.step()

        mark = " ★" if vl_acc > best_val_acc else ""
        print(f"  {ep:>3}  {tr_loss:>10.4f}  {tr_acc:>8.1%}  "
              f"{vl_loss:>9.4f}  {vl_acc:>7.1%}{mark}  ({time.time()-t0:.1f}s)")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = copy.deepcopy(model.state_dict())

    print(f"\n  best val acc ({phase_name}): {best_val_acc:.1%}")
    model.load_state_dict(best_state)
    return model


# ────────────────────────────────────────────────
# SAVE / LOAD — unchanged from previous version
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
    ตัวอย่าง:
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
    global BATCH_SIZE   # ต้องประกาศก่อนมีการใช้ชื่อ BATCH_SIZE ใดๆ ในฟังก์ชันนี้
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

    # 1. โหลดโมเดล
    print(f"\n[1/5] โหลดโมเดล: {args.model}")
    base_obj = load_pkl(args.model)
    model    = extract_pytorch_model(base_obj)
    verify_head(model)
    model    = model.to(DEVICE)

    # 2. โหลด data
    print(f"\n[2/5] โหลด dataset จาก {args.labels_csv} + {args.crop_dir}")
    train_loader, val_loader, class_weights = build_loaders(args)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    # 3. Phase 1
    print(f"\n[3/5] Phase 1: train head เท่านั้น")
    set_grad(model, phase=1)
    model = train_phase(model, train_loader, val_loader, criterion,
                        lr=P1_LR, epochs=p1_ep,
                        phase_name="Phase 1 (head only)")

    # 4. Phase 2
    print(f"\n[4/5] Phase 2: unfreeze ทั้งหมด")
    set_grad(model, phase=2)
    model = train_phase(model, train_loader, val_loader, criterion,
                        lr=P2_LR, epochs=p2_ep,
                        phase_name="Phase 2 (full finetune)")

    # 5. บันทึก
    print(f"\n[5/5] บันทึกโมเดล")
    save_model(model, args.out)


if __name__ == "__main__":
    main()