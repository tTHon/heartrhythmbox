"""
finetune_classification.py
==========================
Finetune classification_model.pkl (ResNet50 backbone, 28-class head)
on a new 18-class dataset defined by a CSV file.

CSV format expected:
    ID, Manuf, ModelClass, PartitionFull, PartitionAbdn, PartitionManuf, ParitionMGrp
    - ID             : image filename without extension  → image path = img_dir/{ID}.png
    - ModelClass     : class label (18 classes)
    - PartitionMGrp : 0 = train, 1 = val

New classes:
    ABT_dPPM, ABT_HV, ABT_Other, BIO_dICD, BIO_Other,
    BSX_CRTD, BSX_dICD, BSX_dPPM, BSX_Other, BSX_scICD, BSX_sICD,
    MDT_CRTD, MDT_CRTP, MDT_dICD, MDT_dPPM, MDT_Leadless, MDT_sICD, MDT_sPPM

Usage:
    # Both phases
    python finetune_classification.py \
        --pkl C:/CIEDID_data/pkl/classification_model.pkl \
        --csv C:/CIEDID_data/labels.csv \
        --img_dir C:/CIEDID_data/images

    # Phase 1 only (head warmup)
    python finetune_classification.py ... --phase 1

    # Phase 2 only (full finetune, load phase1 checkpoint)
    python finetune_classification.py ... --phase 2 --resume checkpoints/phase1_best.pt
"""

import argparse
import os
import pickle

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

# ─────────────────────────────────────────────
# 1.  Class definitions
# ─────────────────────────────────────────────
NEW_CLASSES = [
    "ABT_dPPM", "ABT_HV", "ABT_Other",
    "BIO_dICD", "BIO_Other",
    "BSX_CRTD", "BSX_dICD", "BSX_dPPM", "BSX_Other", "BSX_scICD", "BSX_sICD",
    "MDT_CRTD", "MDT_CRTP", "MDT_dICD", "MDT_dPPM",
    "MDT_Leadless", "MDT_sICD", "MDT_sPPM",
]
NUM_CLASSES  = len(NEW_CLASSES)          # 18
CLASS_TO_IDX = {c: i for i, c in enumerate(NEW_CLASSES)}

# CSV columns
COL_ID    = "ID"
COL_LABEL = "ModelClass"
COL_SPLIT = "PartitionMGrp"    # 0 = train, 1 = val


# ─────────────────────────────────────────────
# 2.  Custom Dataset
# ─────────────────────────────────────────────
class CIEDDataset(Dataset):
    """
    Reads {img_dir}/{ID}.png for each row in df.
    Rows whose ModelClass is not in NEW_CLASSES are skipped with a warning.
    """
    def __init__(self, df: pd.DataFrame, img_dir: str, transform=None):
        unknown = ~df[COL_LABEL].isin(NEW_CLASSES)
        if unknown.any():
            print(f"  ⚠️  Skipping {unknown.sum()} rows with unknown ModelClass: "
                  f"{df.loc[unknown, COL_LABEL].unique().tolist()}")
            df = df[~unknown]

        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row[COL_ID]}.png")
        image    = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = CLASS_TO_IDX[row[COL_LABEL]]
        return image, label


# ─────────────────────────────────────────────
# 3.  Model
# ─────────────────────────────────────────────
class CIEDClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.backbone = backbone      # ResNet50 body (prefix "0" from pkl)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.BatchNorm1d(2048),     # ResNet50 outputs 2048 channels
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)


# ─────────────────────────────────────────────
# 4.  Load pkl → backbone
# ─────────────────────────────────────────────
def load_backbone_from_pkl(pkl_path: str) -> nn.Module:
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)


# FastAI learn.export() stores the model inside the Learner object
    # Common attribute paths: obj.model, obj[0], obj.model[0]

    raw_model = None
    for attr in ["model", "network", "module"]:
        if hasattr(obj, attr):
            raw_model = getattr(obj, attr)
            break
    if raw_model is None:
        try:
            raw_model = obj[0]
        except Exception:
            raise RuntimeError(
                "Cannot locate model inside pkl.\n"
                "Inspect with: pickle.load(open(pkl,'rb')).__dict__.keys()"
            )

    backbone = raw_model[0] if isinstance(raw_model, nn.Sequential) else raw_model
    print(f"✅ Backbone loaded — type: {type(backbone).__name__}")
    return backbone


# ─────────────────────────────────────────────
# 5.  Freeze helpers
# ─────────────────────────────────────────────
def freeze_backbone(model: CIEDClassifier):
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True
    print("🔒 Backbone frozen — training head only.")

def unfreeze_all(model: CIEDClassifier):
    for p in model.parameters():
        p.requires_grad = True
    print("🔓 All layers unfrozen — full fine-tuning.")


# ─────────────────────────────────────────────
# 6.  Data loaders from CSV
# ─────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406] # Standard ImageNet normalization
IMAGENET_STD  = [0.229, 0.224, 0.225] # Standard ImageNet normalization

def get_dataloaders(csv_path: str, img_dir: str,
                    batch_size: int = 32, img_size: int = 224):
    df = pd.read_csv(csv_path)

    for col in [COL_ID, COL_LABEL, COL_SPLIT]:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: '{col}'")

    train_df = df[df[COL_SPLIT] == 0].copy()
    val_df   = df[df[COL_SPLIT] == 1].copy()

    print(f"\n📊 CSV loaded — train: {len(train_df)} rows | val: {len(val_df)} rows")
    print("  Train class counts:")
    print(train_df[COL_LABEL].value_counts().to_string())
    print("  Val class counts:")
    print(val_df[COL_LABEL].value_counts().to_string())

    train_tf = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = CIEDDataset(train_df, img_dir, transform=train_tf)
    val_ds   = CIEDDataset(val_df,   img_dir, transform=val_tf)

    print(f"  Dataset sizes after label filtering — train: {len(train_ds)} | val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    return train_loader, val_loader


# ─────────────────────────────────────────────
# 7.  Training loop
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(imgs)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast():
            logits = model(imgs)
            loss   = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total

def save_checkpoint(model, optimizer, epoch, val_acc, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":     epoch,
        "val_acc":   val_acc,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "classes":   NEW_CLASSES,
    }, path)
    print(f"   💾 Saved → {path}")


# ─────────────────────────────────────────────
# 8.  Two-phase runner
# ─────────────────────────────────────────────
def run_phase(model, train_loader, val_loader, device,
              epochs, lr, phase_name, ckpt_path):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable, lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    scaler    = GradScaler()
    best_acc  = 0.0

    print(f"\n{'='*60}")
    print(f"  {phase_name}  |  {epochs} epochs  |  lr={lr}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion,
                                          optimizer, scaler, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        flag = " ★" if vl_acc > best_acc else ""
        print(f"  Epoch {epoch:03d}/{epochs}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}{flag}")

        if vl_acc > best_acc:
            best_acc = vl_acc
            save_checkpoint(model, optimizer, epoch, vl_acc, ckpt_path)

    print(f"\n  ✅ {phase_name} done. Best val_acc = {best_acc:.4f}")


# ─────────────────────────────────────────────
# 9.  Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl",        required=True,  help="Path to classification_model.pkl")
    parser.add_argument("--csv",        required=True,  help="Path to CSV file")
    parser.add_argument("--img_dir",    required=True,  help="Directory containing {ID}.png images")
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--img_size",   type=int,   default=224)
    parser.add_argument("--phase",      type=int,   default=0,
                        help="0=both phases, 1=head warmup only, 2=full finetune only")
    parser.add_argument("--resume",     default=None,
                        help="Checkpoint .pt to load (required when --phase 2)")
    parser.add_argument("--epochs_p1",  type=int,   default=10)
    parser.add_argument("--epochs_p2",  type=int,   default=30)
    parser.add_argument("--lr_p1",      type=float, default=1e-3)
    parser.add_argument("--lr_p2",      type=float, default=1e-4)
    parser.add_argument("--ckpt_dir",   default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # Load model
    backbone = load_backbone_from_pkl(args.pkl)
    model    = CIEDClassifier(backbone, num_classes=NUM_CLASSES).to(device)
    print(f"📦 Model ready — output classes: {NUM_CLASSES}")

    # Data
    train_loader, val_loader = get_dataloaders(
        args.csv, args.img_dir, args.batch_size, args.img_size
    )

    p1_ckpt = os.path.join(args.ckpt_dir, "phase1_best.pt")
    p2_ckpt = os.path.join(args.ckpt_dir, "phase2_best.pt")

    # Resume from checkpoint
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"▶️  Resumed from {args.resume}  "
              f"(epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")

    # Phase 1 — head warmup
    if args.phase in (0, 1):
        freeze_backbone(model)
        run_phase(model, train_loader, val_loader, device,
                  epochs=args.epochs_p1, lr=args.lr_p1,
                  phase_name="PHASE 1 — Head warmup",
                  ckpt_path=p1_ckpt)

    # Phase 2 — full finetune
    if args.phase in (0, 2):
        if args.phase == 0:
            ckpt = torch.load(p1_ckpt, map_location=device)
            model.load_state_dict(ckpt["model"])
            print("▶️  Loaded phase-1 best before phase 2")
        unfreeze_all(model)
        run_phase(model, train_loader, val_loader, device,
                  epochs=args.epochs_p2, lr=args.lr_p2,
                  phase_name="PHASE 2 — Full finetune",
                  ckpt_path=p2_ckpt)

    final = p2_ckpt if args.phase != 1 else p1_ckpt
    print(f"\n🎉 Done! Best model → {final}")
    print(f"   Classes: {NEW_CLASSES}")


if __name__ == "__main__":
    main()