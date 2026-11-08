# Stage IV: ModelClass prediction on the held-out crops (FinalTest == 1).
# Mirrors finetuneModelClass.py (fastai): Resize(256, pad, zeros) -> ImageNet Normalize, cnn_learner(resnet50).
# Weights: final_epoch_weights.pth = FINAL-epoch weights of each fold (the reported model).
# Class names/order are read from class_names.txt written at training time (no manual list needed).
# PRIMARY result = average of the softmax of all 5 fold models (column modelclass_pred); per-fold calls are saved too.
import pathlib, platform
from pathlib import Path
import numpy as np, pandas as pd, torch
from fastai.vision.all import *

MODEL_DIR = "C:/CIEDID_data/ModelClass/epoch_25_lrslice(1e-5, 1e-3)/models_encoder_s42"
FOLDS     = [0, 1, 2, 3, 4]                       # use [k] for a single fold
WORKBOOK  = "C:/CIEDID_data/main_workbook.csv"
CROP_DIR  = "C:/CIEDID_data/Images_crop"
OUT       = "C:/CIEDID_data/finalTest/pred_modelclass.csv"

if platform.system() == "Windows": pathlib.PosixPath = pathlib.WindowsPath
else: pathlib.WindowsPath = pathlib.PosixPath
_ol = torch.load
torch.load = lambda *a, **k: _ol(*a, **{**k, "weights_only": False})

# class order must be identical across folds
cls_sets = [Path(f"{MODEL_DIR}/fold_{k}/class_names.txt").read_text().split("\n") for k in FOLDS]
CLASSES = [c.strip() for c in cls_sets[0] if c.strip()]
for k, cs in zip(FOLDS, cls_sets):
    assert [c.strip() for c in cs if c.strip()] == CLASSES, f"class_names differ in fold {k}"
print(f"{len(CLASSES)} classes: {CLASSES}")
no_w = [f"{MODEL_DIR}/fold_{k}/final_epoch_weights.pth" for k in FOLDS
        if not Path(f"{MODEL_DIR}/fold_{k}/final_epoch_weights.pth").exists()]
assert not no_w, f"weight files not found: {no_w}"

wb = pd.read_csv(WORKBOOK, usecols=range(13), dtype={"ID": str}, low_memory=False)
ids = wb.loc[wb.FinalTest == 1, "ID"].str.strip().tolist()
paths = [str((Path(CROP_DIR) / f"{i}_crop.png").resolve()) for i in ids]
missing = [p for p in paths if not Path(p).exists()]
assert not missing, f"{len(missing)} crops missing, e.g. {missing[:3]}"

# test crops go in the validation split (no augmentation, deterministic centred padding); dummy labels are unused
test = pd.DataFrame({"ID": ids, "image_path": paths, "label": CLASSES[0], "is_valid": True})
dummy = pd.concat([test.iloc[[0]]] * 4, ignore_index=True); dummy["is_valid"] = False
dblock = DataBlock(blocks=(ImageBlock, CategoryBlock(vocab=CLASSES)),
                   get_x=lambda r: r["image_path"], get_y=lambda r: r["label"],
                   splitter=ColSplitter(col="is_valid"),
                   item_tfms=Resize(256, method="pad", pad_mode="zeros"),
                   batch_tfms=[Normalize.from_stats(*imagenet_stats)])
dls = dblock.dataloaders(pd.concat([dummy, test], ignore_index=True), bs=16, num_workers=0)
assert dls.valid_ds.items["ID"].tolist() == ids, "validation order != workbook order"
learn = cnn_learner(dls, resnet50, pretrained=False).to_fp16()

fold_probs = []
for k in FOLDS:
    w = f"{MODEL_DIR}/fold_{k}/final_epoch_weights.pth"
    learn.model.load_state_dict(torch.load(w, map_location=dls.device))   # strict: any mismatch fails loudly
    probs, _ = learn.get_preds(dl=dls.valid)
    fold_probs.append(probs.float().numpy())
    print("done:", w)

p = np.mean(fold_probs, axis=0)
out = pd.DataFrame({"ID": ids, "modelclass_pred": [CLASSES[i] for i in p.argmax(1)]})
for k, fp in zip(FOLDS, fold_probs):
    out[f"modelclass_pred_fold{k}"] = [CLASSES[i] for i in fp.argmax(1)]
for j, c in enumerate(CLASSES):
    out[f"p_{c}"] = p[:, j]
out.to_csv(OUT, index=False)
print(len(out), "predictions ->", OUT)