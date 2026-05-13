import torch
import pathlib
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from fastai.vision.all import *
from PIL import Image

# ==========================================================
# 1. SETTINGS (ใช้ค่าเดียวกับที่คุณตั้งไว้)
# ==========================================================
path_img_folder = "C:/CIEDID_data/AbdnL/data"
path_weights = "C:/CIEDID_data/AbdnL/models/fold_0/best_gen.pth"
IMG_Size = 512
ABDN_CLASS_IDX = 3

# พารามิเตอร์ที่คุณกำหนด
threshold = 0.5   
pixel_min = 1500  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================================
# 2. LOAD MODEL & PIPELINE
# ==========================================================
img_files = get_image_files(path_img_folder)
# โหลดโมเดล (โครงสร้างเดียวกับตอนเทรน)
dls = SegmentationDataLoaders.from_label_func(
    pathlib.Path("."), bs=1, fnames=img_files[:1], 
    label_func=lambda x: x, codes=["B","G","L","A"], 
    item_tfms=Resize(IMG_Size, method='pad')
)
learn = unet_learner(dls, resnet50, n_out=4)
learn.model.load_state_dict(torch.load(path_weights, map_location=device))
learn.model.to(device).eval()

timg_pipe = Pipeline([PILImage.create, Resize(IMG_Size, method='pad', pad_mode='zeros'), ToTensor(), IntToFloatTensor()])
mean = torch.tensor([0.502668]*3, device=device).view(3, 1, 1)
std = torch.tensor([0.240966]*3, device=device).view(3, 1, 1)

# ==========================================================
# 3. VISUALIZATION & DETECTION LOGIC
# ==========================================================
random_imgs = random.sample(img_files, 3)
fig, axes = plt.subplots(3, 2, figsize=(14, 18))

for i, img_path in enumerate(random_imgs):
    # --- Inference ---
    timg = timg_pipe(img_path).to(device)
    timg_norm = (timg - mean) / std
    
    with torch.no_grad():
        output = learn.model(timg_norm.unsqueeze(0))
        probs = F.softmax(output, dim=1)[0].cpu()
        abdn_prob_map = probs[ABDN_CLASS_IDX].numpy()
    
    # --- Detection Decision ---
    # นับพิกเซลที่ความมั่นใจ > threshold
    pixel_count = (abdn_prob_map > threshold).sum()
    is_detected = pixel_count >= pixel_min
    
    # --- Plotting ---
    # 1. รูปต้นฉบับ
    raw_img = timg.permute(1, 2, 0).cpu().numpy()
    axes[i, 0].imshow(raw_img)
    axes[i, 0].set_title(f"Image: {img_path.name}")
    axes[i, 0].axis('off')
    
    # 2. Probability Heatmap พร้อมสถานะการตัดสินใจ
    im = axes[i, 1].imshow(abdn_prob_map, cmap='jet', vmin=0, vmax=1)
    
    # แสดงผลข้อความบนรูป
    status_text = f"DETECTED" if is_detected else "NOT DETECTED"
    color = 'red' if is_detected else 'white'
    
    axes[i, 1].text(20, 45, f"Status: {status_text}\nPixels > {threshold}: {pixel_count:,}\nRequired: {pixel_min:,}", 
                  color='white', fontsize=12, fontweight='bold',
                  bbox=dict(facecolor=color, alpha=0.7, edgecolor='none'))
    
    axes[i, 1].set_title(f"Model Perspective (Abdn_Lead)")
    axes[i, 1].axis('off')
    plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()