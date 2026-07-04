import pathlib
import platform
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from fastai.vision.all import *
from tqdm import tqdm
from PIL import Image

# ==========================================================
# 1. CONFIGURATION
# ==========================================================
FILE_WEIGHTS  = 'C:/CIEDID_data/AbdnL/models/best/best_abdn.pth'
DIR_IMAGES    = 'C:/CIEDID_data/AbdnL/data'
DIR_MASKS     = 'C:/CIEDID_data/AbdnL/mask'
CSV_OUTPUT    = 'C:/CIEDID_data/AbdnL/best/grid_search_results.csv'

IMG_Size = 640
CLASS_NAMES = ["Background", "Generator", "Lead", "Abdn_Lead"]
ABDN_CLASS_IDX = 3

# ==========================================================
# 2. PREPARATION
# ==========================================================
if platform.system() == 'Windows': pathlib.PosixPath = pathlib.WindowsPath
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_files = get_image_files(DIR_IMAGES)

# สร้าง Model Structure (ResNet50)
dls_dummy = SegmentationDataLoaders.from_label_func(
    pathlib.Path("."), bs=1, fnames=img_files[:1], 
    label_func=lambda x: x, codes=CLASS_NAMES, 
    item_tfms=Resize(IMG_Size, method='pad', pad_mode='zeros')
)
learn = unet_learner(dls_dummy, resnet50, n_out=len(CLASS_NAMES))
learn.model.load_state_dict(torch.load(FILE_WEIGHTS, map_location=device))
learn.model.to(device).eval()

# Pipeline การเตรียมภาพ (ตรงตาม finetuneCombinedLoss.py)
timg_pipe = Pipeline([PILImage.create, Resize(IMG_Size, method='pad', pad_mode='zeros'), ToTensor(), IntToFloatTensor()])
mean_tensor = torch.tensor([0.502668]*3, device=device).view(3, 1, 1)
std_tensor  = torch.tensor([0.240966]*3, device=device).view(3, 1, 1)

# ==========================================================
# 3. GRID SEARCH
# ==========================================================
PROB_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.70, 0.8, 0.9, 1.0]
PIXEL_MIN_SPACE = [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000,4100,4200,4300,4400,4500,4600,4700,4800,4900,5000]

grid_results = []
all_probs = []
all_gts = []

print("Extracting probabilities...")
for img_path in tqdm(img_files):
    mask_path = pathlib.Path(DIR_MASKS) / f"{img_path.stem}_mask.png"
    if not mask_path.exists(): mask_path = pathlib.Path(DIR_MASKS) / f"{img_path.stem}.png"
    if not mask_path.exists(): continue

    # โหลด Mask แบบ Grayscale + Nearest Resize (เหมือน Fastai MaskBlock)
    gt_mask = Image.open(mask_path).convert('L').resize((IMG_Size, IMG_Size), resample=Image.NEAREST)
    gt_has_abdn = (ABDN_CLASS_IDX in np.unique(np.array(gt_mask)))
    
    # Inference
    timg = timg_pipe(img_path).to(device)
    timg = (timg - mean_tensor) / std_tensor
    with torch.no_grad():
        output = learn.model(timg.unsqueeze(0))
        prob_map = F.softmax(output, dim=1)[0, ABDN_CLASS_IDX].cpu().numpy()
    
    all_probs.append(prob_map)
    all_gts.append(gt_has_abdn)

print("Searching optimal parameters...")
for prob_th in PROB_THRESHOLDS:
    for pix_min in PIXEL_MIN_SPACE:
        TP, FP, TN, FN = 0, 0, 0, 0
        for i in range(len(all_probs)):
            pred_has_abdn = (all_probs[i] > prob_th).sum() >= pix_min
            gt_has_abdn = all_gts[i]
            
            if pred_has_abdn and gt_has_abdn: TP += 1
            elif pred_has_abdn and not gt_has_abdn: FP += 1
            elif not pred_has_abdn and not gt_has_abdn: TN += 1
            else: FN += 1

        sens = TP/(TP+FN) if (TP+FN)>0 else 0
        spec = TN/(TN+FP) if (TN+FP)>0 else 0
        prec = TP/(TP+FP) if (TP+FP)>0 else 0
        f1 = 2*(prec*sens)/(prec+sens) if (prec+sens)>0 else 0
        grid_results.append({
            'Prob_Th': prob_th, 
            'Pix_Min': pix_min, 
            'Sensitivity': sens, 
            'Specificity': spec,
            'F1': f1})

df = pd.DataFrame(grid_results).sort_values(by=['Sensitivity', 'Specificity'], ascending=[False, False])
df.to_csv(CSV_OUTPUT, index=False)
print(f"Done! Results saved to {CSV_OUTPUT}")