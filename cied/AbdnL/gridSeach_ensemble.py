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
#FILE_WEIGHTS  = 'C:/CIEDID_data/AbdnL/models/fold_0/seg_abdnL_weights.pth'
DIR_IMAGES    = 'C:/CIEDID_data/AbdnL/data'
DIR_MASKS     = 'C:/CIEDID_data/AbdnL/mask'
CSV_OUTPUT    = 'C:/CIEDID_data/AbdnL/grid_search_results.csv'

# --- ส่วนการโหลดโมเดล (ปรับปรุง) ---
model_paths = [f'C:/CIEDID_data/AbdnL/models/fold_{i}/best_gen.pth' for i in range(5)]
models = []

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

for p in model_paths:
    # สร้าง Learner โครงสร้างเดิม
    learn = unet_learner(dls_dummy, resnet50, n_out=4)
    learn.model.load_state_dict(torch.load(p, map_location=device))
    learn.model.to(device).eval()
    models.append(learn.model)


# Pipeline การเตรียมภาพ (ตรงตาม finetuneCombinedLoss.py)
timg_pipe = Pipeline([PILImage.create, Resize(IMG_Size, method='pad', pad_mode='zeros'), ToTensor(), IntToFloatTensor()])
mean_tensor = torch.tensor([0.502668]*3, device=device).view(3, 1, 1)
std_tensor  = torch.tensor([0.240966]*3, device=device).view(3, 1, 1)

# ==========================================================
# 3. GRID SEARCH
# ==========================================================
PROB_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
PIXEL_MIN_SPACE = [500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000]

grid_results = []
all_probs = []
all_gts = []

print("Extracting probabilities...")
all_probs = []
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
        # สร้าง Tensor ว่างบน GPU เพื่อสะสมผลลัพธ์
        ensemble_prob = torch.zeros((IMG_Size, IMG_Size), device=device)
        for m in models:
            output = m(timg.unsqueeze(0))
            ensemble_prob += F.softmax(output, dim=1)[0, ABDN_CLASS_IDX]
        
        # หารเฉลี่ยแล้วค่อยย้ายไป CPU ครั้งเดียว
        ensemble_prob = (ensemble_prob / len(models)).cpu().numpy()
        
    all_probs.append(ensemble_prob)
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