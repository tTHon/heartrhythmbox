# This script fine-tunes a pre-trained model on a custom dataset defined by a CSV file.
# The CSV should have columns: ID, Manuf, PartitionManuf (0 for train, 1 for val).
# The script handles class mapping, optional custom normalization stats, and a two-phase training approach.
# This script used custom normalization stats calculated from the training data, which can be enabled with the --calc_stats flag.

"""
finetune_manuf.py
=================
Finetune classification_manuf.pkl on our dataset

--- FastAI Model Inspection ---
✅ Found Vocab in dls.vocab:
  Index 0: Biotronik
  Index 1: Boston Scientific
  Index 2: Medtronic
  Index 3: St. Jude Medical
  Index 4: Vitatron

Strategy: keep all head 5 class outputs, 
but mask out Vitatron (index 4) during training 
by setting its weight to 0 in the loss function.

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
from torch.cuda.amp import GradScaler, autocast

# ─────────────────────────────────────────────
# 1. Config & Global Constants
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model's original vocab: 0:Biotronik, 1:Boston, 2:Medtronic, 3:St. Jude, 4:Vitatron
OUR_CLASSES = ["Biotronik", "Boston Scientific", "Medtronic", "Abbott"]
# Map CSV "Manuf" strings to the model's expected indices
# Note: Abbott in your data corresponds to "St. Jude Medical" (Index 3) in the pkl
MANUF_MAP = {
    "Biotronik": 0,
    "Boston Scientific": 1,
    "Medtronic": 2,
    "Abbott": 3
}
ABSENT_CLASSES = [4] # Vitatron

# Standard ImageNet Fallbacks
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────
# 2. Custom CSV Dataset
# ─────────────────────────────────────────────
class ManufDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['ID']}.png")
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        # Map manufacturer string to integer index
        label = MANUF_MAP[row['Manuf']]
        return image, label

# ─────────────────────────────────────────────
# 3. Normalization Stats Calculation
# ─────────────────────────────────────────────
@torch.no_grad()
def calculate_stats(loader):
    print("  📊 Calculating custom mean/std from CSV data...")
    sum_ = torch.zeros(3)
    sum_sq = torch.zeros(3)
    count = 0
    for images, _ in loader:
        b, c, h, w = images.shape
        num_pixels = b * h * w
        sum_ += torch.sum(images, dim=[0, 2, 3])
        sum_sq += torch.sum(images**2, dim=[0, 2, 3])
        count += num_pixels
    mean = sum_ / count
    std = torch.sqrt((sum_sq / count) - (mean**2))
    return mean.tolist(), std.tolist()

# ─────────────────────────────────────────────
# 4. Helpers (from original fineTuneManuf.py)
# ─────────────────────────────────────────────
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def extract_pytorch_model(obj):
    if hasattr(obj, 'model'): return obj.model
    if isinstance(obj, list): return obj[0]
    return obj

def set_grad(model, phase=1):
    if phase == 1:
        for p in model[0].parameters(): p.requires_grad = False
        for p in model[1].parameters(): p.requires_grad = True
        print("  [Phase 1] Frozen backbone, training head only.")
    else:
        for p in model.parameters(): p.requires_grad = True
        print("  [Phase 2] Unfrozen all layers.")

# ─────────────────────────────────────────────
# 5. Build Dataloaders
# ─────────────────────────────────────────────
def build_loaders(csv_path, img_dir, batch_size, calc_custom_stats=False):
    df = pd.read_csv(csv_path)
    train_df = df[df['PartitionManuf'] == 0]
    val_df = df[df['PartitionManuf'] == 1]

    mean, std = IMAGENET_MEAN, IMAGENET_STD

    if calc_custom_stats:
        temp_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        temp_ds = ManufDataset(train_df, img_dir, temp_tf)
        temp_loader = DataLoader(temp_ds, batch_size=batch_size, num_workers=4)
        mean, std = calculate_stats(temp_loader)
        print(f"  ✅ Custom Stats: Mean={mean}, Std={std}")

    tfms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }

    train_ds = ManufDataset(train_df, img_dir, tfms['train'])
    val_ds = ManufDataset(val_df, img_dir, tfms['val'])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Weight handling: Mask out index 4 (Vitatron)
    weights = torch.ones(5)
    weights[4] = 0.0 
    
    return train_loader, val_loader, weights

# ─────────────────────────────────────────────
# 6. Main execution
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    required=True, help="Path to .pkl")
    parser.add_argument("--csv",      required=True, help="Path to CSV file")
    parser.add_argument("--img_dir",  required=True, help="Folder with id.png")
    parser.add_argument("--epochs",   type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--calc_stats", action="store_true", help="Use custom mean/std")
    args = parser.parse_args()

    # Load Model
    base_obj = load_pkl(args.model)
    model = extract_pytorch_model(base_obj).to(DEVICE)

    # Build Data
    train_loader, val_loader, class_weights = build_loaders(
        args.csv, args.img_dir, args.batch_size, args.calc_stats
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    # Phase 1: Warmup
    set_grad(model, phase=1)
    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    # ... [Training Loop logic similar to original script] ...

    # Phase 2: Full Finetune
    set_grad(model, phase=2)
    optimizer = Adam(model.parameters(), lr=1e-4)
    # ... [Training Loop logic similar to original script] ...

if __name__ == "__main__":
    main()