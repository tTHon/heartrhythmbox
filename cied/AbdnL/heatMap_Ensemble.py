import torch
import pathlib
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from fastai.vision.all import *
from PIL import Image

# ==========================================================
# 1. SETTINGS & PARAMETERS
# ==========================================================
path_img_folder = "C:/CIEDID_data/AbdnL/data4Test"
# เปลี่ยนเป็นรายชื่อ Path ของทั้ง 5 Folds
model_paths = [f'C:/CIEDID_data/AbdnL/models/fold_{i}/best_gen.pth' for i in range(5)]

IMG_Size = 640
threshold = 0.5    # ปรับตามผลที่ดีที่สุดจาก Grid Search
pixel_min = 2250   # ปรับตามผลที่ดีที่สุดจาก Grid Search

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================================
# 2. LOAD MODELS (Ensemble)
# ==========================================================
img_files = get_image_files(path_img_folder)
dls_dummy = SegmentationDataLoaders.from_label_func(
    pathlib.Path("."), bs=1, fnames=img_files[:1], 
    label_func=lambda x: x, codes=["Background","Generator","Lead","Abdn_Lead"], 
    item_tfms=Resize(IMG_Size, method='pad')
)

models = []
print(f"Loading 5 models for Ensemble on {device}...")
for p in model_paths:
    learn = unet_learner(dls_dummy, resnet50, n_out=4)
    learn.model.load_state_dict(torch.load(p, map_location=device))
    learn.model.to(device).eval()
    models.append(learn.model)

timg_pipe = Pipeline([PILImage.create, Resize(IMG_Size, method='pad', pad_mode='zeros'), ToTensor(), IntToFloatTensor()])
mean = torch.tensor([0.502668]*3, device=device).view(3, 1, 1)
std = torch.tensor([0.240966]*3, device=device).view(3, 1, 1)

# ==========================================================
# 3. VISUALIZATION FUNCTION
# ==========================================================
def get_overlay(prob_map, color_rgb):
    """ สร้างสี Gradient บางๆ ตามค่า Probability """
    overlay = np.zeros((*prob_map.shape, 3))
    for i in range(3):
        overlay[..., i] = color_rgb[i]
    # ปรับความเข้ม (Alpha) ตามค่า Prob ตรงๆ เพื่อทำ Gradient
    return overlay, prob_map * 0.3 

# สุ่มรูปมาแสดง
num_samples = min(len(img_files), 3)
random_imgs = random.sample(img_files, num_samples)
fig, axes = plt.subplots(1, num_samples, figsize=(20, 8))
if num_samples == 1: axes = [axes] # Handle single plot

for i, img_path in enumerate(random_imgs):
    timg = timg_pipe(img_path).to(device)
    timg_norm = (timg - mean) / std
    
    with torch.no_grad():
        # --- ENSEMBLE LOGIC ---
        # สร้าง Tensor ว่างสำหรับเก็บผลรวมทุกคลาส (4 คลาส) บน GPU
        accumulated_probs = torch.zeros((4, IMG_Size, IMG_Size), device=device)
        
        for m in models:
            output = m(timg_norm.unsqueeze(0))
            accumulated_probs += F.softmax(output, dim=1)[0]
        
        # หาค่าเฉลี่ย
        avg_probs = (accumulated_probs / len(models)).cpu().numpy()
    
    # แยก Probability ของแต่ละคลาสจากค่าเฉลี่ย
    gen_prob  = avg_probs[1]
    lead_prob = avg_probs[2]
    abdn_prob = avg_probs[3]
    
    # ตรวจสอบเงื่อนไขตามพารามิเตอร์ที่กำหนด
    pixel_count = (abdn_prob > threshold).sum()
    detected = pixel_count >= pixel_min
    
    # เตรียมภาพ Background
    raw_img = timg.permute(1, 2, 0).cpu().numpy()
    axes[i].imshow(raw_img)
    
    # สร้างและซ้อน Overlay (เขียว, ฟ้า, แดง)
    for p_map, color in [(gen_prob, [0, 1, 0]), (lead_prob, [0, 0.6, 1]), (abdn_prob, [1, 0, 0])]:
        ov, alpha = get_overlay(p_map, color)
        axes[i].imshow(ov, alpha=alpha)
    
    # แสดงรายละเอียด
    status_text = "FOUND" if detected else "NOT FOUND"
    res_color = 'red' if detected else 'gray'
    axes[i].set_title(f"{img_path.name}\nAbdn Lead: {status_text} ({pixel_count:,} px)", 
                      fontsize=10, color=res_color, fontweight='bold')
    axes[i].axis('off')

# Legend
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=[0, 1, 0], lw=4),
                Line2D([0], [0], color=[0, 0.6, 1], lw=4),
                Line2D([0], [0], color=[1, 0, 0], lw=4)]
fig.legend(custom_lines, ['Generator', 'Normal Lead', 'Abandoned Lead'], loc='lower center', ncol=3)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()