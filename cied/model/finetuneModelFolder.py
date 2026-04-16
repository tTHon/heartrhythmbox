"""
finetune_classification.py
==========================
Finetune classification_model.pkl (ResNet50 backbone, 28-class head)
on a new 18-class ImageFolder dataset.

New classes:
    ABT_dPPM, ABT_HV, ABT_Other, BIO_dICD, BIO_Other,
    BSX_CRTD, BSX_dICD, BSX_dPPM, BSX_Other, BSX_scICD, BSX_sICD,
    MDT_CRTD, MDT_CRTP, MDT_dICD, MDT_dPPM, MDT_Leadless, MDT_sICD, MDT_sPPM

Dataset layout expected (ImageFolder style):
    data_root/
        train/
            ABT_dPPM/  img1.jpg ...
            ABT_HV/    ...
            ...
        val/
            ABT_dPPM/  ...
            ...

Usage:
    python finetune_classification.py --data_root /path/to/data --pkl /path/to/classification_model.pkl
    
    # Phase 1 only (head warmup):
    python finetune_classification.py --data_root ... --pkl ... --phase 1

    # Phase 2 only (full finetune, requires phase1 checkpoint):
    python finetune_classification.py --data_root ... --pkl ... --phase 2 --resume checkpoints/phase1_best.pt
"""

import argparse
import os
import pickle
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
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
NUM_CLASSES = len(NEW_CLASSES)   # 18

# Keys that belong to the backbone (prefix "0") vs head (prefix "1")
ENCODER_PREFIXES = ("0",)        # ResNet50 backbone lives under prefix "0"


# ─────────────────────────────────────────────
# 2.  Model wrapper
# ─────────────────────────────────────────────
class CIEDClassifier(nn.Module):
    """
    Wraps the FastAI-exported backbone and attaches a fresh 18-class head.
    The backbone is a Sequential stored under key "0" in the pkl state dict.
    The original head (key "1") had shape (28, 512); we replace it with (18, 512).
    """
    def __init__(self, backbone: nn.Module, num_classes: int = 18):
        super().__init__()
        self.backbone = backbone          # prefix "0" — ResNet50 body
        # Mirror the original head structure: AdaptivePool → Flatten → BN(4096)
        # → Linear(4096,512) → BN(512) → Linear(512, num_classes)
        # We keep everything except the final linear which changes 28 → num_classes.
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.BatchNorm1d(2048),          # ResNet50 last channel = 2048
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
# 3.  Load pkl and extract backbone
# ─────────────────────────────────────────────
def load_backbone_from_pkl(pkl_path: str) -> nn.Module:
    """
    Loads classification_model.pkl and returns the backbone nn.Module
    with pretrained weights (prefix "0" state dict keys).
    """
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
        # Try indexing — some FastAI exports wrap in a list/tuple
        try:
            raw_model = obj[0]
        except Exception:
            raise RuntimeError(
                "Cannot locate model inside pkl. "
                "Inspect with: pickle.load(open(pkl,'rb')).__dict__.keys()"
            )

    # raw_model is typically nn.Sequential([body, head])
    # body is at index 0 (ENCODER_PREFIXES = "0")
    if isinstance(raw_model, nn.Sequential):
        backbone = raw_model[0]
    else:
        backbone = raw_model

    print(f"✅ Backbone loaded. Type: {type(backbone).__name__}")
    return backbone


# ─────────────────────────────────────────────
# 4.  Freezing helpers
# ─────────────────────────────────────────────
def freeze_backbone(model: CIEDClassifier):
    """Freeze all backbone weights; only head trains."""
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True
    print("🔒 Backbone frozen — training head only.")


def unfreeze_all(model: CIEDClassifier):
    """Unfreeze everything for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    print("🔓 All layers unfrozen — full fine-tuning.")


def unfreeze_last_n_backbone_layers(model: CIEDClassifier, n: int = 2):
    """
    Unfreeze only the last n top-level children of the backbone.
    Useful for a middle-ground strategy (layer4 + head, etc.).
    """
    children = list(model.backbone.children())
    for child in children[:-n]:
        for param in child.parameters():
            param.requires_grad = False
    for child in children[-n:]:
        for param in child.parameters():
            param.requires_grad = True
    for param in model.head.parameters():
        param.requires_grad = True
    print(f"🔓 Last {n} backbone layer(s) + head unfrozen.")


# ─────────────────────────────────────────────
# 5.  Data loaders
# ─────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_dataloaders(data_root: str, batch_size: int = 32, img_size: int = 224):
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

    train_ds = datasets.ImageFolder(os.path.join(data_root, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(data_root, "val"),   transform=val_tf)

    # Verify class alignment
    detected = sorted(train_ds.classes)
    expected = sorted(NEW_CLASSES)
    if detected != expected:
        print("⚠️  WARNING: folder class names don't match NEW_CLASSES!")
        print(f"   Detected : {detected}")
        print(f"   Expected : {expected}")
    else:
        print(f"✅ Classes aligned: {train_ds.classes}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    return train_loader, val_loader


# ─────────────────────────────────────────────
# 6.  Training loop
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
    print(f"   💾 Saved checkpoint → {path}")


# ─────────────────────────────────────────────
# 7.  Two-phase training
# ─────────────────────────────────────────────
def run_phase(model, train_loader, val_loader, device,
              epochs, lr, phase_name, ckpt_path,
              weight_decay=1e-4):

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    scaler    = GradScaler()

    best_acc = 0.0
    print(f"\n{'='*55}")
    print(f"  {phase_name}  |  {epochs} epochs  |  lr={lr}")
    print(f"{'='*55}")

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
    return ckpt_path


# ─────────────────────────────────────────────
# 8.  Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Finetune CIED classifier on new 18-class data")
    parser.add_argument("--pkl",        required=True,  help="Path to classification_model.pkl")
    parser.add_argument("--data_root",  required=True,  help="Root dir with train/ and val/ subfolders")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size",   type=int, default=224)
    parser.add_argument("--phase",      type=int, default=0,
                        help="0=both phases, 1=head warmup only, 2=full finetune only")
    parser.add_argument("--resume",     default=None,
                        help="Checkpoint .pt to resume from (required for --phase 2)")
    parser.add_argument("--epochs_p1",  type=int, default=10,
                        help="Epochs for phase 1 (head warmup)")
    parser.add_argument("--epochs_p2",  type=int, default=30,
                        help="Epochs for phase 2 (full finetune)")
    parser.add_argument("--lr_p1",      type=float, default=1e-3,
                        help="Learning rate for phase 1")
    parser.add_argument("--lr_p2",      type=float, default=1e-4,
                        help="Learning rate for phase 2")
    parser.add_argument("--ckpt_dir",   default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # ── Load model ──────────────────────────────
    backbone = load_backbone_from_pkl(args.pkl)
    model    = CIEDClassifier(backbone, num_classes=NUM_CLASSES).to(device)
    print(f"📦 Model ready. Output classes: {NUM_CLASSES}")

    # ── Data ────────────────────────────────────
    train_loader, val_loader = get_dataloaders(
        args.data_root, args.batch_size, args.img_size
    )

    p1_ckpt = os.path.join(args.ckpt_dir, "phase1_best.pt")
    p2_ckpt = os.path.join(args.ckpt_dir, "phase2_best.pt")

    # ── Resume if requested ──────────────────────
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"▶️  Resumed from {args.resume}  (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")

    # ── Phase 1: train head only ─────────────────
    if args.phase in (0, 1):
        freeze_backbone(model)
        run_phase(model, train_loader, val_loader, device,
                  epochs=args.epochs_p1, lr=args.lr_p1,
                  phase_name="PHASE 1 — Head warmup",
                  ckpt_path=p1_ckpt)

    # ── Phase 2: full finetune ───────────────────
    if args.phase in (0, 2):
        # Load best phase-1 checkpoint before unfreezing
        if args.phase == 0:
            ckpt = torch.load(p1_ckpt, map_location=device)
            model.load_state_dict(ckpt["model"])
            print(f"▶️  Loaded phase-1 best before phase 2")
        unfreeze_all(model)
        # Optional: use differential LRs (backbone slower than head)
        # Uncomment below and comment out the unfreeze_all call above for
        # a gentler approach that reduces catastrophic forgetting:
        #
        # unfreeze_last_n_backbone_layers(model, n=2)
        # optimizer groups would need backbone_lr = args.lr_p2 / 10
        #
        run_phase(model, train_loader, val_loader, device,
                  epochs=args.epochs_p2, lr=args.lr_p2,
                  phase_name="PHASE 2 — Full finetune",
                  ckpt_path=p2_ckpt)

    print("\n🎉 Training complete!")
    print(f"   Best model saved at: {p2_ckpt if args.phase != 1 else p1_ckpt}")
    print(f"   Classes: {NEW_CLASSES}")


if __name__ == "__main__":
    main()