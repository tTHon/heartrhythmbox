"""
finetune_manuf.py
=================
Finetune classification_manuf.pkl บน chest X-ray dataset ของเรา

Vocab เดิมของโมเดล (5 class, index 0-4):
  0: Biotronik
  1: Boston Scientific
  2: Medtronic
  3: ST. Jude Medical
  4: Vitatron   ← ไม่มีในข้อมูลเรา

Strategy: คง head 5 class ไว้ทั้งหมด — ไม่ replace head
  - Weight ของ 4 class ที่เรามีนั้น pretrained มาแล้ว → เก็บไว้
  - Dataset folder ต้องชื่อตาม OUR_CLASSES ด้านล่าง
  - Loss: CrossEntropyLoss weighted, Vitatron weight=0 (masked ออก)

โครงสร้าง dataset:
  dataset/
    train/  Abbott/  Medtronic/  BostonScientific/  Biotronik/
    val/    Abbott/  Medtronic/  BostonScientific/  Biotronik/

วิธีใช้:
  python finetune_manuf.py \
      --model  classification_manuf.pkl \
      --data   dataset/ \
      --epochs 30 \
      --out    finetuned_manuf.pkl

Finetuning strategy (2-phase):
  Phase 1 (10 ep) : freeze encoder '0.*', train head '1.*' เท่านั้น
  Phase 2 (20 ep) : unfreeze ทั้งหมด, lr เล็กมาก
"""

import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pickle
import numpy as np

# ────────────────────────────────────────────────
# VOCAB — ต้องตรงกับโมเดลเดิมทุกประการ
# key = ชื่อ folder ใน dataset, value = index ใน output layer
# ────────────────────────────────────────────────
ORIG_VOCAB = {
    "Biotronik":        0,
    "BSX": 1,   # folder ชื่อ "BostonScientific" (ไม่มีช่องว่าง)
    "Medtronic":        2,
    "Abbott":           3,
    "Vitatron":         4,   # ไม่มีในข้อมูลเรา → weight=0
}
OUR_CLASSES    = ["Biotronik", "BSX", "Medtronic","Abbott"]  # ชื่อ folder ต้องตรงนี้']
ABSENT_CLASSES = ["Vitatron"]
NUM_ORIG       = len(ORIG_VOCAB)   # 5 — ขนาด head ไม่เปลี่ยน

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
# AUGMENTATION  (CXR-specific)
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
# MODEL LOADING  (ไม่ replace head)
# ────────────────────────────────────────────────

def load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


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
# DATASET  — remap ImageFolder index → ORIG_VOCAB index
# ────────────────────────────────────────────────

class RemappedDataset(torch.utils.data.Dataset):
    """
    ImageFolder ให้ label 0-3 (alphabetical)
    เรา remap → index ใน ORIG_VOCAB เพื่อให้ตรงกับ pretrained head
    เช่น Abbott (folder idx 0) → ORIG_VOCAB['Abbott'] = 3
    """
    def __init__(self, imagefolder_ds):
        self.ds = imagefolder_ds
        self.label_map = {}
        for cls, folder_idx in imagefolder_ds.class_to_idx.items():
            if cls not in ORIG_VOCAB:
                raise ValueError(
                    f"folder '{cls}' ไม่อยู่ใน ORIG_VOCAB\n"
                    f"ชื่อ folder ที่รองรับ: {list(ORIG_VOCAB.keys())}"
                )
            self.label_map[folder_idx] = ORIG_VOCAB[cls]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, folder_label = self.ds[i]
        return img, self.label_map[folder_label]


def build_loaders(data_dir: str):
    raw_train = datasets.ImageFolder(str(Path(data_dir) / "train"), transform=TRAIN_TF)
    raw_val   = datasets.ImageFolder(str(Path(data_dir) / "val"),   transform=VAL_TF)

    print(f"\n  ImageFolder classes: {raw_train.class_to_idx}")
    print(f"  train: {len(raw_train)} ภาพ  |  val: {len(raw_val)} ภาพ")

    train_ds = RemappedDataset(raw_train)
    val_ds   = RemappedDataset(raw_val)

    print(f"  label remap (folder_idx → orig_vocab_idx): {train_ds.label_map}")

    # class weight — size=NUM_ORIG, Vitatron=0
    remapped_targets  = [train_ds.label_map[t] for t in raw_train.targets]
    class_counts      = np.bincount(remapped_targets, minlength=NUM_ORIG).astype(float)
    class_weights     = torch.zeros(NUM_ORIG)
    for cls in OUR_CLASSES:
        idx = ORIG_VOCAB[cls]
        class_weights[idx] = 1.0 / class_counts[idx] if class_counts[idx] > 0 else 0.0

    print(f"\n  class counts : { {k: int(class_counts[v]) for k, v in ORIG_VOCAB.items()} }")
    print(f"  class weights: { {k: round(class_weights[v].item(), 5) for k, v in ORIG_VOCAB.items()} }")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, class_weights


# ────────────────────────────────────────────────
# TRAIN / EVAL
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
# SAVE / LOAD
# ────────────────────────────────────────────────

def save_model(model: nn.Module, out_path: str):
    payload = {
        "model_state_dict": model.state_dict(),
        "orig_vocab":       ORIG_VOCAB,
        "our_classes":      OUR_CLASSES,
        "num_orig":         NUM_ORIG,
        "img_size":         IMG_SIZE,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n✅ บันทึกโมเดลที่ {out_path}")


def load_finetuned(finetuned_pkl: str, original_pkl: str):
    """
    โหลดโมเดลที่ finetune แล้ว
    Returns: (model, orig_vocab, our_classes)

    ตัวอย่าง:
        model, vocab, classes = load_finetuned("finetuned_manuf.pkl", "classification_manuf.pkl")
        idx_to_class = {v: k for k, v in vocab.items()}
        # ใช้ idx_to_class เพื่อ decode prediction
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
    parser = argparse.ArgumentParser(
        description="Finetune manufacturer classifier (5-class head preserved, Vitatron masked)"
    )
    parser.add_argument("--model",   default="classification_manuf.pkl")
    parser.add_argument("--data",    default="dataset/",
                        help="dataset root (ต้องมี train/ และ val/)")
    parser.add_argument("--epochs",  type=int, default=P1_EPOCHS + P2_EPOCHS)
    parser.add_argument("--p1_frac", type=float,
                        default=P1_EPOCHS / (P1_EPOCHS + P2_EPOCHS))
    parser.add_argument("--out",     default="finetuned_manuf.pkl")
    parser.add_argument("--batch",   type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    global BATCH_SIZE
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
    print(f"\n[2/5] โหลด dataset: {args.data}")
    train_loader, val_loader, class_weights = build_loaders(args.data)
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