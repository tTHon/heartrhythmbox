import torch
import pathlib
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from fastai.vision.all import *
from PIL import Image
import scipy.ndimage as ndimage
from matplotlib.lines import Line2D

# ==========================================================
# 0. DETERMINISTIC SETTINGS
# ==========================================================
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ==========================================================
# 1. SETTINGS & PARAMETERS
# ==========================================================
path_img_folder = "C:/CIEDID_data/AbdnL/test_data"
path_weights    = "C:/CIEDID_data/AbdnL/models/best/best_abdn.pth"
IMG_Size = 640

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================================
# 2. LOAD MODEL
# ==========================================================
img_files = sorted(get_image_files(path_img_folder))

dls = SegmentationDataLoaders.from_label_func(
    pathlib.Path("."), bs=1, fnames=img_files[:1], 
    label_func=lambda x: x, codes=["Background","Generator","Lead","Abdn_Lead"], 
    item_tfms=Resize(IMG_Size, method='pad')
)
learn = unet_learner(dls, resnet50, n_out=4)
learn.model.load_state_dict(torch.load(path_weights, map_location=device))
learn.model.to(device).eval()

timg_pipe = Pipeline([PILImage.create, Resize(IMG_Size, method='pad', pad_mode='zeros'), ToTensor(), IntToFloatTensor()])
mean = torch.tensor([0.5150052309036255]*3, device=device).view(3, 1, 1)
std = torch.tensor([0.23788487911224365]*3, device=device).view(3, 1, 1)

# ==========================================================
# 3. HELPER FUNCTIONS FOR CROP & BLEND
# ==========================================================
def auto_crop_center(img, threshold=0.05):
    """ ตรวจหาขอบดำที่แท้จริงและครอปให้ภาพ X-ray อยู่กึ่งกลางอย่างสมดุล """
    gray = np.mean(img, axis=2) if img.ndim == 3 else img
    row_mask = np.mean(gray, axis=1) > threshold
    col_mask = np.mean(gray, axis=0) > threshold
    
    if not np.any(row_mask) or not np.any(col_mask):
        return 0, img.shape[0], 0, img.shape[1]
        
    r_start, r_end = np.where(row_mask)[0][0], np.where(row_mask)[0][-1]
    c_start, c_end = np.where(col_mask)[0][0], np.where(col_mask)[0][-1]
    
    pad = 5
    r_start = max(0, r_start - pad)
    r_end = min(img.shape[0], r_end + pad)
    c_start = max(0, c_start - pad)
    c_end = min(img.shape[1], c_end + pad)
    
    return r_start, r_end, c_start, c_end

# ==========================================================
# 4. PROCESSING SELECTED IMAGES (SIDE-BY-SIDE VIEW)
# ==========================================================
selected_filenames = ["a_x1.png"]
selected_imgs = [p for p in img_files if p.name in selected_filenames]

missing = set(selected_filenames) - {p.name for p in selected_imgs}
if missing:
    raise FileNotFoundError(f"ไม่พบไฟล์เหล่านี้ใน {path_img_folder}: {missing}")

# สร้าง 1 แถว 2 คอลัมน์ เพื่อวางภาพ Original ไว้ซ้าย และ Heatmap ไว้ขวา
fig, axes = plt.subplots(1, 2, figsize=(16,9))
plt.rcParams.update({'font.size': 18, 'font.family': 'inter'})

for img_path in selected_imgs:
    timg = timg_pipe(img_path).to(device)
    timg_norm = (timg - mean) / std
    
    with torch.no_grad():
        output = learn.model(timg_norm.unsqueeze(0))
        probs = F.softmax(output, dim=1)[0].cpu().numpy()
    
    gen_prob  = probs[1]
    lead_prob = probs[2]
    abdn_prob = probs[3]
    
    raw_img = timg.permute(1, 2, 0).cpu().numpy()
    raw_img = np.clip(raw_img, 0, 1)
    
    # 1. ทำ Direct Alpha Blending สำหรับภาพฝั่งขวา (Heatmap)
    color_gen  = np.array([0.0, 1.0, 0.0])  
    color_lead = np.array([0.0, 0.6, 1.0])  
    color_abdn = np.array([1.0, 0.0, 0.0])  
    
    max_alpha = 0.5 
    blended_img = raw_img.copy()
    
    for prob, color in zip([gen_prob, lead_prob, abdn_prob], [color_gen, color_lead, color_abdn]):
        alpha = np.expand_dims(prob * max_alpha, axis=-1)
        blended_img = (1 - alpha) * blended_img + alpha * color

    blended_img = np.clip(blended_img, 0, 1)
    
    # 2. หาพิกัดการครอปจัดกึ่งกลาง (Auto-Centering)
    r_start, r_end, c_start, c_end = auto_crop_center(raw_img)
    
    # ครอปทั้งคู่เพื่อให้ขนาดเฟรมและสัดส่วนตรงกันเป๊ะ
    cropped_original = raw_img[r_start:r_end, c_start:c_end]
    cropped_heatmap  = blended_img[r_start:r_end, c_start:c_end]
    
    # --- ฝั่งซ้าย: Original Image ---
    axes[0].imshow(cropped_original)
    axes[0].set_title("Original Image", fontsize=16, fontweight='bold', pad=10)
    axes[0].axis('off')
    
    # --- ฝั่งขวา: Bright Heatmap Overlay ---
    axes[1].imshow(cropped_heatmap)
    axes[1].set_title("Model-Predicted Heatmap", fontsize=16, fontweight='bold', pad=10)
    axes[1].axis('off')

# สร้าง Legend อธิบายสี
custom_lines = [Line2D([0], [0], color=[0, 1, 0], lw=5),
                Line2D([0], [0], color=[0, 0.6, 1], lw=5),
                Line2D([0], [0], color=[1, 0, 0], lw=5)]

fig.legend(custom_lines,
           ['Generator', 'Active Lead', 'Abandoned Lead'],
           loc='upper center', ncol=3, fontsize=16, frameon=True, framealpha=0.9, edgecolor='darkgray')

# คำอธิบายรายละเอียดด้านล่าง
fig.text(0.5, 0.05,
         "Color opacity indicates model-predicted probability: "
         "darker/more saturated = higher probability, faint = lower probability.",
         ha='center', fontsize=14, style='italic', color='#555555')

plt.tight_layout(rect=[0, 0.05, 1, 0.92])
plt.savefig("cied/AbdnL/heatmap_overlay.png", dpi=300, bbox_inches='tight')