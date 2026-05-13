import torch
import pathlib
import numpy as np
import pandas as pd
import torch.nn.functional as F
from fastai.vision.all import *
from sklearn.metrics import classification_report
from PIL import Image
from tqdm import tqdm

# ==========================================================
# CONFIGURATION (ใช้ค่าจาก Grid Search ที่ดีที่สุด)
# ==========================================================
path_img_folder = "C:/CIEDID_data/AbdnL/data"
path_mask_folder = "C:/CIEDID_data/AbdnL/mask"
path_weights = "C:/CIEDID_data/AbdnL/models/fold_0/best_gen.pth"
threshold = 0.5  
pixel_min = 1500 
IMG_Size = 512
CLASS_NAMES = ["Background", "Generator", "Lead", "Abdn_Lead"]
ABDN_CLASS_IDX = 3

if os.name == 'nt': pathlib.PosixPath = pathlib.WindowsPath
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load Model
img_files = get_image_files(path_img_folder)
dls = SegmentationDataLoaders.from_label_func(pathlib.Path("."), bs=1, fnames=img_files[:1], label_func=lambda x:x, codes=CLASS_NAMES, item_tfms=Resize(IMG_Size, method='pad'))
learn = unet_learner(dls, resnet50, n_out=4)
learn.model.load_state_dict(torch.load(path_weights, map_location=device))
learn.model.to(device)  # ย้ายโมเดลไปที่ GPU (หรือ CPU ตามที่ระบุใน device)
learn.model.eval()

# 2. Preprocessing
timg_pipe = Pipeline([PILImage.create, Resize(IMG_Size, method='pad', pad_mode='zeros'), ToTensor(), IntToFloatTensor()])
mean = torch.tensor([0.502668]*3, device=device).view(3, 1, 1)
std = torch.tensor([0.240966]*3, device=device).view(3, 1, 1)

results_case, all_y_true, all_y_pred = [], [], []

print("Running Inference...")
for img_path in tqdm(img_files):
    mask_path = pathlib.Path(path_mask_folder) / f"{img_path.stem}_mask.png"
    if not mask_path.exists(): mask_path = pathlib.Path(path_mask_folder) / f"{img_path.stem}.png"
    if not mask_path.exists(): continue

    # Ground Truth: บังคับ Grayscale และ Nearest Resize (แก้ Error Inconsistent samples)
    gt_mask = Image.open(mask_path).convert('L').resize((IMG_Size, IMG_Size), resample=Image.NEAREST)
    y_true_arr = np.array(gt_mask)
    gt_exists = (ABDN_CLASS_IDX in np.unique(y_true_arr))

    # Prediction
    timg = timg_pipe(img_path).to(device)
    timg = (timg - mean) / std
    with torch.no_grad():
        output = learn.model(timg.unsqueeze(0))
        probs = F.softmax(output, dim=1)[0].cpu()
        pred_mask = torch.argmax(probs, dim=0).numpy()

    # Case-level Logic
    abdn_prob_map = probs[ABDN_CLASS_IDX].numpy()
    pred_exists = (abdn_prob_map > threshold).sum() >= pixel_min
    
    # Pixel-level Logic (Sync กับ Case-level)
    final_pixel_mask = pred_mask.copy()
    if not pred_exists:
        final_pixel_mask[final_pixel_mask == ABDN_CLASS_IDX] = 0
    else:
        final_pixel_mask[(final_pixel_mask == ABDN_CLASS_IDX) & (abdn_prob_map < threshold)] = 0

    results_case.append({'Actual': gt_exists, 'Predicted': pred_exists})
    all_y_true.extend(y_true_arr.flatten())
    all_y_pred.extend(final_pixel_mask.flatten())

# Reporting
print("\nCase-level Report:")
print(classification_report([x['Actual'] for x in results_case], [x['Predicted'] for x in results_case], zero_division=0))

print("\nPixel-level Report:")
print(classification_report(all_y_true, all_y_pred, target_names=CLASS_NAMES, labels=[0,1,2,3], zero_division=0))