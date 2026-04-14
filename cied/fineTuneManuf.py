"""
finetune_manuf.py
=================
Finetune classification_manuf.pkl บน chest X-ray dataset ของเรา
รองรับ 4 class: Abbott, Medtronic, BostonScientific, Biotronik

โครงสร้าง dataset ที่ต้องการ (สร้างด้วย prepare_dataset.py):
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
  Phase 1 (head only)  : freeze encoder '0.*', train head '1.*'  → 10 epoch
  Phase 2 (full unfreeze): unfreeze ทั้งหมด, lr เล็กมาก           → 20 epoch
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
# CONFIG
# ────────────────────────────────────────────────
CLASSES        = ["Abbott", "Medtronic", "BostonScientific", "Biotronik"]
NUM_CLASSES    = len(CLASSES)          # 4
ENCODER_PREFIX = "0"                   # backbone prefix ใน state_dict

IMG_SIZE       = 224
BATCH_SIZE     = 16
NUM_WORKERS    = 2

# Phase 1: train head only
P1_EPOCHS      = 10
P1_LR          = 1e-3

# Phase 2: full fine-tune
P2_EPOCHS      = 20
P2_LR          = 1e-5

WEIGHT_DECAY   = 1e-4
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# ────────────────────────────────────────────────
# AUGMENTATION
# ────────────────────────────────────────────────
# CXR-specific: grayscale → replicate 3ch, mild augmentation
TRAIN_TF = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

VAL_TF = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ────────────────────────────────────────────────
# MODEL LOADING & HEAD REPLACEMENT
# ────────────────────────────────────────────────

def load_pkl(pkl_path: str):
    """โหลด fastai / custom pickle model"""
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    return obj


def extract_pytorch_model(obj) -> nn.Module:
    """
    พยายาม unwrap model ออกจาก fastai Learner หรือ wrapper
    Returns nn.Module พร้อมใช้
    """
    if isinstance(obj, nn.Module):
        return obj
    # fastai Learner
    if hasattr(obj, "model"):
        return obj.model
    raise ValueError(f"ไม่รู้จัก object type: {type(obj)}")


def replace_head(model: nn.Module, num_classes: int) -> nn.Module:
    """
    เปลี่ยน output layer ใน head (prefix '1') ให้ match num_classes ใหม่
    
    จาก state_dict:
      1.4.weight  (512, 4096)
      1.8.weight  (5, 512)    ← output เดิม 5 class → เปลี่ยนเป็น 4
    """
    head = model[1]  # Sequential head

    # หา Linear layer สุดท้าย
    last_linear_idx = None
    for idx, layer in enumerate(head):
        if isinstance(layer, nn.Linear):
            last_linear_idx = idx

    if last_linear_idx is None:
        raise RuntimeError("ไม่พบ Linear layer ใน head")

    old_linear: nn.Linear = head[last_linear_idx]
    in_features = old_linear.in_features
    print(f"  เปลี่ยน output layer: ({old_linear.out_features} → {num_classes}) | in_features={in_features}")

    head[last_linear_idx] = nn.Linear(in_features, num_classes)
    return model


def set_grad(model: nn.Module, phase: int):
    """
    phase=1 → freeze encoder '0.*', train head '1.*'
    phase=2 → unfreeze ทั้งหมด
    """
    if phase == 1:
        frozen, trainable = 0, 0
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
# DATA LOADER
# ────────────────────────────────────────────────

def build_loaders(data_dir: str):
    train_ds = datasets.ImageFolder(str(Path(data_dir) / "train"), transform=TRAIN_TF)
    val_ds   = datasets.ImageFolder(str(Path(data_dir) / "val"),   transform=VAL_TF)

    # แสดง class mapping
    print(f"\n  class mapping (ImageFolder): {train_ds.class_to_idx}")
    print(f"  train: {len(train_ds)} ภาพ  |  val: {len(val_ds)} ภาพ")

    # class weight สำหรับ imbalanced data
    targets = train_ds.targets
    class_counts = np.bincount(targets, minlength=NUM_CLASSES).astype(float)
    class_weights = torch.tensor(
        [1.0 / c if c > 0 else 0.0 for c in class_counts], dtype=torch.float
    )
    print(f"  class counts: {dict(zip(train_ds.classes, class_counts.astype(int)))}")
    print(f"  class weights (loss): {class_weights.tolist()}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, class_weights, train_ds.class_to_idx


# ────────────────────────────────────────────────
# TRAIN / EVAL LOOP
# ────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, is_train: bool, device: str):
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss    = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += imgs.size(0)

    return total_loss / total, correct / total


def train_phase(model, train_loader, val_loader, criterion,
                lr: float, epochs: int, phase_name: str, device: str):
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc  = 0.0
    best_state    = None
    print(f"\n{'─'*55}")
    print(f"  {phase_name}  |  lr={lr}  epochs={epochs}")
    print(f"{'─'*55}")
    print(f"  {'Ep':>3}  {'TrainLoss':>10}  {'TrainAcc':>9}  {'ValLoss':>9}  {'ValAcc':>8}")

    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, True,  device)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer, False, device)
        scheduler.step()

        marker = " ★" if vl_acc > best_val_acc else ""
        print(f"  {ep:>3}  {tr_loss:>10.4f}  {tr_acc:>8.1%}  {vl_loss:>9.4f}  {vl_acc:>7.1%}{marker}  ({time.time()-t0:.1f}s)")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = copy.deepcopy(model.state_dict())

    print(f"\n  best val acc ({phase_name}): {best_val_acc:.1%}")
    model.load_state_dict(best_state)
    return model


# ────────────────────────────────────────────────
# SAVE
# ────────────────────────────────────────────────

def save_model(model: nn.Module, class_to_idx: dict, out_path: str):
    """
    บันทึกเป็น pkl dict ที่มีทั้ง state_dict และ metadata
    โหลดกลับด้วย load_finetuned()
    """
    payload = {
        "model_state_dict": model.state_dict(),
        "class_to_idx":     class_to_idx,
        "num_classes":      NUM_CLASSES,
        "img_size":         IMG_SIZE,
        "classes":          CLASSES,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n✅ บันทึกโมเดลที่ {out_path}")


# ────────────────────────────────────────────────
# INFERENCE HELPER (ใช้ใน production)
# ────────────────────────────────────────────────

def load_finetuned(pkl_path: str, original_pkl: str) -> tuple[nn.Module, dict]:
    """
    โหลดโมเดลที่ finetune แล้ว
    Returns: (model, class_to_idx)
    
    ตัวอย่าง:
        model, c2i = load_finetuned("finetuned_manuf.pkl", "classification_manuf.pkl")
        idx_to_class = {v: k for k, v in c2i.items()}
    """
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)

    base_obj = load_pkl(original_pkl)
    model    = extract_pytorch_model(base_obj)
    model    = replace_head(model, payload["num_classes"])
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload["class_to_idx"]


# ────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Finetune manufacturer classifier")
    parser.add_argument("--model",   type=str, default="classification_manuf.pkl",
                        help="path to original .pkl model")
    parser.add_argument("--data",    type=str, default="dataset/",
                        help="dataset root (ต้องมี train/ และ val/)")
    parser.add_argument("--epochs",  type=int, default=P1_EPOCHS + P2_EPOCHS,
                        help=f"total epochs (default={P1_EPOCHS + P2_EPOCHS})")
    parser.add_argument("--p1_frac", type=float, default=P1_EPOCHS / (P1_EPOCHS + P2_EPOCHS),
                        help="สัดส่วน phase1 ของ total epochs (default=0.33)")
    parser.add_argument("--out",     type=str, default="finetuned_manuf.pkl",
                        help="output pkl path")
    parser.add_argument("--batch",   type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    global BATCH_SIZE
    BATCH_SIZE = args.batch

    p1_ep = max(1, int(args.epochs * args.p1_frac))
    p2_ep = max(1, args.epochs - p1_ep)

    print(f"\n{'='*55}")
    print(f"  Finetune Manufacturer Classifier")
    print(f"  device={DEVICE}  | classes={CLASSES}")
    print(f"  phase1={p1_ep}ep (head only)  phase2={p2_ep}ep (full)")
    print(f"{'='*55}")

    # ── โหลดโมเดลเดิม ──
    print(f"\n[1/5] โหลดโมเดล: {args.model}")
    base_obj = load_pkl(args.model)
    model    = extract_pytorch_model(base_obj)
    print(f"  โหลดสำเร็จ: {type(model).__name__}")

    # ── เปลี่ยน head ──
    print(f"\n[2/5] เปลี่ยน output head → {NUM_CLASSES} classes")
    model = replace_head(model, NUM_CLASSES)
    model = model.to(DEVICE)

    # ── โหลด data ──
    print(f"\n[3/5] โหลด dataset: {args.data}")
    train_loader, val_loader, class_weights, class_to_idx = build_loaders(args.data)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    # ── Phase 1: head only ──
    print(f"\n[4/5] Phase 1: train head เท่านั้น")
    set_grad(model, phase=1)
    model = train_phase(model, train_loader, val_loader, criterion,
                        lr=P1_LR, epochs=p1_ep,
                        phase_name="Phase 1 (head only)", device=DEVICE)

    # ── Phase 2: full finetune ──
    print(f"\n[5/5] Phase 2: unfreeze ทั้งหมด")
    set_grad(model, phase=2)
    model = train_phase(model, train_loader, val_loader, criterion,
                        lr=P2_LR, epochs=p2_ep,
                        phase_name="Phase 2 (full finetune)", device=DEVICE)

    # ── บันทึก ──
    save_model(model, class_to_idx, args.out)


if __name__ == "__main__":
    main()