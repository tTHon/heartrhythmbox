# Stage III: manufacturer prediction on the held-out crops (FinalTest == 1).
# Mirrors finetuneManufCV.py: Grayscale(3) -> Resize(256,256) -> CenterCrop(224) -> ImageNet Normalize.
# Weights: final-epoch Phase-2 state_dicts (final_epoch_weights_phase2.pth), NOT manuf_finetuned.pkl
#   (the .pkl is the best checkpoint picked on its own validation fold = optimistic).
# PRIMARY result = average of the softmax of all 5 fold models (column manuf_pred).
# Per-fold predictions are saved too, but the primary choice is fixed here, before looking at the test set.
import pathlib, platform
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from torchvision import transforms
from fastai.vision.all import load_learner, create_cnn_model, resnet50

MODEL_DIR = "C:/CIEDID_data/Manuf/models_encoder_newhead_s42"
FOLDS     = [0, 1, 2, 3, 4]                       # use [k] for a single fold
WEIGHTS   = [f"{MODEL_DIR}/fold_{k}/final_epoch_weights_phase2.pth" for k in FOLDS]
BASE_PKL  = "C:/CIEDID_data/pkl/classification_manuf.pkl"   # only used if the architecture fallback is needed
WORKBOOK  = "C:/CIEDID_data/main_workbook.csv"
CROP_DIR  = "C:/CIEDID_data/Images_crop"
OUT       = "C:/CIEDID_data/finalTest/pred_manuf.csv"

if platform.system() == "Windows": pathlib.PosixPath = pathlib.WindowsPath
else: pathlib.WindowsPath = pathlib.PosixPath
_ol = torch.load
torch.load = lambda *a, **k: _ol(*a, **{**k, "weights_only": False})
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TF = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                         transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
NAMES  = ["Biotronik", "Boston Scientific", "Medtronic", "ST. Jude Medical", "Vitatron"]   # pretrained index order
TO_CSV = ["Biotronik", "BSX", "Medtronic", "Abbott", "Vitatron"]
VIT = 4

def load_model(path):
    sd = torch.load(path, map_location=DEVICE)
    try:
        m = create_cnn_model(resnet50, 5, pretrained=False)
        m.load_state_dict(sd)
    except RuntimeError:                           # head differs from fastai default -> use the pretrained pkl's architecture
        m = load_learner(BASE_PKL, cpu=(DEVICE == "cpu")).model
        m.load_state_dict(sd)
    return m.to(DEVICE).eval()

wb = pd.read_csv(WORKBOOK, usecols=range(13), dtype={"ID": str}, low_memory=False)
ids = wb.loc[wb.FinalTest == 1, "ID"].str.strip().tolist()
paths = [Path(CROP_DIR) / f"{i}_crop.png" for i in ids]
missing = [str(p) for p in paths if not p.exists()]
assert not missing, f"{len(missing)} crops missing, e.g. {missing[:3]}"
no_w = [w for w in WEIGHTS if not Path(w).exists()]
assert not no_w, f"weight files not found: {no_w}"
print(f"{len(ids)} test crops | models: {len(WEIGHTS)}")

imgs = [TF(Image.open(p).convert("RGB")) for p in paths]       # deterministic preprocessing, done once
fold_probs = []
for w in WEIGHTS:
    model, pr = load_model(w), []
    with torch.no_grad():
        for i in range(0, len(imgs), 32):
            pr.append(torch.softmax(model(torch.stack(imgs[i:i + 32]).to(DEVICE)), 1).cpu().numpy())
    fold_probs.append(np.concatenate(pr))
    print("done:", w)

def call(p):                                       # Vitatron is absent in our data: mask, then argmax
    q = p.copy(); q[:, VIT] = -1
    return q.argmax(1)

p = np.mean(fold_probs, axis=0)
print("Vitatron was the raw argmax in", int((p.argmax(1) == VIT).sum()), "images (masked)")
out = pd.DataFrame({"ID": ids, "manuf_pred": [TO_CSV[i] for i in call(p)]})
for k, fp in zip(FOLDS, fold_probs):
    out[f"manuf_pred_fold{k}"] = [TO_CSV[i] for i in call(fp)]
for j, c in enumerate(TO_CSV):
    out[f"p_{c}"] = p[:, j]
out.to_csv(OUT, index=False)
print(len(out), "predictions ->", OUT)