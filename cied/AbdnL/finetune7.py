# ==============================
# PATCH-BASED FINETUNE VERSION
# This version extracts random patches from the images during training, with a bias towards areas containing abandoned leads. This allows us to train on smaller patches (e.g. 256x256) instead of the full 512x512 images, which can help with memory and speed while still learning relevant features.
# not working
# ==============================

import sys, pathlib, platform, argparse, os, random
import numpy as np, pandas as pd, torch, warnings
warnings.filterwarnings("ignore")

os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts=false"

if not hasattr(np, 'int'):
    np.int = int

import fastai.callback.fp16
if not hasattr(fastai.callback.fp16, 'AMPMode'):
    class AMPMode:
        def __init__(self, *args, **kwargs): pass
    fastai.callback.fp16.AMPMode = AMPMode
    sys.modules['fastai.callback.fp16'].AMPMode = AMPMode

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

from fastai.vision.all import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==============================
# CONFIG
# ==============================
BASE_DIR = pathlib.Path("C:/CIEDID_data")
MODEL_PATH = pathlib.Path("C:/CIEDID_data/pkl/segmentation.pkl")
NEW_IMGS = pathlib.Path("C:/CIEDID_data/AbdnL/data")
NEW_MASKS = pathlib.Path("C:/CIEDID_data/AbdnL/mask")
OUTPUT_DIR = pathlib.Path("C:/CIEDID_data/AbdnL/models")

CLASS_NAMES = ["background","generator","lead","abandoned_lead"]
N_OUT_NEW = 4
SEED = 42

# ==============================
# PATCH LOGIC
# ==============================
def get_patch_coords(mask, patch_size=256):
    h, w = mask.shape
    ys, xs = np.where(mask > 0)

    if len(ys) > 0 and random.random() < 0.7:
        idx = random.randint(0, len(ys)-1)
        cy, cx = ys[idx], xs[idx]
    else:
        cy = random.randint(0, h-1)
        cx = random.randint(0, w-1)

    y1 = max(0, cy - patch_size//2)
    x1 = max(0, cx - patch_size//2)
    y2 = min(h, y1 + patch_size)
    x2 = min(w, x1 + patch_size)

    if (y2 - y1) < patch_size:
        y1 = max(0, y2 - patch_size)
    if (x2 - x1) < patch_size:
        x1 = max(0, x2 - patch_size)

    return y1, y2, x1, x2

from fastai.data.transforms import Transform

class PatchTransform(Transform):
    def __init__(self, patch_size=256):
        self.patch_size = patch_size

    def encodes(self, x:tuple):
        img, mask = x   # ✅ ตอนนี้จะเป็น tuple แล้ว

        img_np  = np.array(img)
        mask_np = np.array(mask)

        y1, y2, x1, x2 = get_patch_coords(mask_np, self.patch_size)

        img_patch  = img_np[y1:y2, x1:x2]
        mask_patch = mask_np[y1:y2, x1:x2]

        return PILImage.create(img_patch), PILMask.create(mask_patch)

# ==============================
# DATAFRAME
# ==============================
def build_dataframe(args):
    rows = []
    exts = {".jpg",".png",".jpeg",".bmp"}

    for img in args.new_imgs.iterdir():
        if img.suffix.lower() not in exts: continue

        mask = args.new_masks / f"{img.stem}_mask.png"
        if not mask.exists():
            mask = args.new_masks / f"{img.stem}.png"
        if not mask.exists(): continue

        arr = np.array(PILImage.create(mask))

        rows.append({
            "image": str(img.resolve()),
            "mask": str(mask.resolve()),
            "has_abandoned": 3 in np.unique(arr)
        })

    df = pd.DataFrame(rows)

    train_idx, valid_idx = train_test_split(
        df.index,
        test_size=args.valid_split,
        stratify=df["has_abandoned"],
        random_state=SEED
    )

    df["is_valid"] = False
    df.loc[valid_idx, "is_valid"] = True

    return df

# ==============================
# LOSS
# ==============================
class WeightedSegLoss(Module):
    def __init__(self, class_weights, alpha=0.2, axis=1):
        self.alpha = alpha
        self.axis = axis
        self.weights = class_weights
        self.ce = CrossEntropyLossFlat(axis=axis, weight=class_weights)

    def forward(self, pred, targ):
        ce_loss = self.ce(pred, targ)

        pred_soft = pred.softmax(dim=self.axis)
        dice_loss = 0.
        w_sum = 0.
        eps = 1e-6

        for c in range(pred_soft.shape[1]):
            w = self.weights[c].item()
            p = pred_soft[:, c]
            t = (targ == c).float()
            inter = (p*t).sum()
            union = p.sum() + t.sum()
            dice_c = 1 - (2*inter+eps)/(union+eps)
            dice_loss += w*dice_c
            w_sum += w

        dice_loss = dice_loss/(w_sum+eps)
        return self.alpha*ce_loss + (1-self.alpha)*dice_loss

# ==============================
# METRIC
# ==============================
def dice_abandoned(inp, targ, eps=1e-6):
    pred = inp.argmax(dim=1)
    p = (pred==3).float()
    t = (targ==3).float()
    inter = (p*t).sum()
    union = p.sum() + t.sum()
    return (2*inter+eps)/(union+eps)

# ==============================
# TRAIN
# ==============================
def finetune(args):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_dataframe(args)

    def get_x(r): return r["image"]
    def get_y(r): return r["mask"]

    dblock = DataBlock(
        blocks=(ImageBlock, MaskBlock(codes=CLASS_NAMES)),
        get_x=get_x,
        get_y=get_y,
        splitter=ColSplitter('is_valid'),

        item_tfms=[
            PatchTransform(patch_size=args.patch_size),
            Resize(args.patch_size)
        ],

        batch_tfms=[
            *aug_transforms(size=args.patch_size,
                            max_rotate=15,
                            max_zoom=1.2,
                            max_lighting=0.2,
                            max_warp=0.1),
        ]
    )

    dls = dblock.dataloaders(df, bs=args.batch_size, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = torch.tensor([1.0, 15.0, 15.0, 100.0]).to(device)

    learn = unet_learner(
        dls,
        resnet50,
        n_out=N_OUT_NEW,
        loss_func=WeightedSegLoss(weights, alpha=0.2),
        metrics=[DiceMulti(), dice_abandoned]
    ).to_fp16()

    learn.freeze_to(-2)
    learn.fit_one_cycle(args.epochs_head, 1e-3)

    learn.show_results(max_n=4)

    learn.unfreeze()
    learn.fit_one_cycle(args.epochs_full, lr_max=slice(1e-6,1e-4))

    learn.export(OUTPUT_DIR/"seg_patch.pkl")

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=str(MODEL_PATH))
    parser.add_argument("--new_imgs", default=str(NEW_IMGS))
    parser.add_argument("--new_masks", default=str(NEW_MASKS))
    parser.add_argument("--epochs_head", type=int, default=5)
    parser.add_argument("--epochs_full", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--valid_split", type=float, default=0.2)

    args = parser.parse_args()
    args.new_imgs = pathlib.Path(args.new_imgs)
    args.new_masks = pathlib.Path(args.new_masks)

    finetune(args)