import torch
import pathlib
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from fastai.vision.all import *
from PIL import Image
import scipy.ndimage as ndimage

# ==========================================================
# 1. SETTINGS & PARAMETERS
# ==========================================================
path_img_folder = "C:/CIEDID_data/AbdnL/test_data"
path_weights    = "C:/CIEDID_data/AbdnL/models/best/best_abdn.pth"
IMG_Size = 640
threshold = 0.8   
pixel_min = 600

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================================
# 2. LOAD MODEL
# ==========================================================
img_files = get_image_files(path_img_folder)
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
# 3. VISUALIZATION FUNCTION (Superimposed Gradient)
# ==========================================================
def get_overlay(img, prob_map, color_rgb):
    """ สร้างสี Gradient บางๆ ตามค่า Probability """
    overlay = np.zeros((*prob_map.shape, 3))
    for i in range(3):
        overlay[..., i] = color_rgb[i]
    # ปรับความเข้มตามความมั่นใจ (Gradient) และทำให้บางลง (Alpha 0.4)
    return overlay, prob_map * 0.3 

# ==========================================================
# ระบุชื่อไฟล์ภาพที่ต้องการทำ heatmap (แทนการสุ่ม)
# ใส่แค่ชื่อไฟล์ (พร้อมนามสกุล) ตามที่อยู่ใน path_img_folder
# ==========================================================
selected_filenames = [
    "a_x1.png",
]

selected_imgs = [p for p in img_files if p.name in selected_filenames]

missing = set(selected_filenames) - {p.name for p in selected_imgs}
if missing:
    raise FileNotFoundError(
        f"ไม่พบไฟล์เหล่านี้ใน {path_img_folder}: {missing}\n"
        f"เช็คว่าพิมพ์ชื่อไฟล์ถูกต้อง (รวมนามสกุล .png/.jpg) และมีไฟล์อยู่จริงในโฟลเดอร์"
    )

random_imgs = selected_imgs  # ใช้ตัวแปรชื่อเดิมเพื่อไม่ต้องแก้โค้ดส่วนล่าง
fig, axes = plt.subplots(1, len(random_imgs), figsize=(20, 8))
if len(random_imgs) == 1:
    axes = [axes]  # ทำให้ index ได้เหมือนกันแม้มีภาพเดียว

for i, img_path in enumerate(random_imgs):
    timg = timg_pipe(img_path).to(device)
    timg_norm = (timg - mean) / std
    
    with torch.no_grad():
        output = learn.model(timg_norm.unsqueeze(0))
        probs = F.softmax(output, dim=1)[0].cpu().numpy()
    
    # ดึง Probability ของแต่ละคลาส
    gen_prob  = probs[1]
    lead_prob = probs[2]
    abdn_prob = probs[3]
    
    # ตรวจสอบเงื่อนไข Abandoned Lead
    pixel_count = (abdn_prob > threshold).sum()
    detected = pixel_count > pixel_min
    
    # แปลงภาพต้นฉบับเพื่อแสดงผล
    raw_img = timg.permute(1, 2, 0).cpu().numpy()
    
    # แสดงภาพต้นฉบับ
    axes[i].imshow(raw_img)
    
    # สร้าง Overlay แต่ละคลาส (Gradient บางๆ)
    # Generator = เขียว, Lead = ฟ้า, Abdn = แดง
    gen_ov, gen_alpha   = get_overlay(raw_img, gen_prob, [0, 1, 0])
    lead_ov, lead_alpha = get_overlay(raw_img, lead_prob, [0, 0.6, 1])
    abdn_ov, abdn_alpha = get_overlay(raw_img, abdn_prob, [1, 0, 0])
    
    # ซ้อนทับลงบนภาพ
    axes[i].imshow(gen_ov, alpha=gen_alpha)
    axes[i].imshow(lead_ov, alpha=lead_alpha)
    axes[i].imshow(abdn_ov, alpha=abdn_alpha)

    # วาดเส้นขอบ (contour) ที่ threshold จริงที่ใช้ตัดสินใจ abandoned lead
    # เพื่อให้เห็นชัดว่า "เกินเส้นนี้ = นับเป็น abandoned lead" ไม่ใช่แค่ gradient ลอยๆ
    if abdn_prob.max() > threshold:
        axes[i].contour(abdn_prob, levels=[threshold], colors='yellow',
                         linewidths=1.5, linestyles='solid')
    
    # ตกแต่ง Text
    status_text = "FOUND" if detected else "NOT FOUND"
    res_color = 'red' if detected else 'gray'
    axes[i].set_title(f"{img_path.name}\nAbdn Lead: {status_text} ({pixel_count:,} px)", 
                      fontsize=10, color=res_color, fontweight='bold')
    axes[i].axis('off')

# สร้าง Legend อธิบายสี
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=[0, 1, 0], lw=4),
                Line2D([0], [0], color=[0, 0.6, 1], lw=4),
                Line2D([0], [0], color=[1, 0, 0], lw=4),
                Line2D([0], [0], color='yellow', lw=1.5)]
fig.legend(custom_lines,
           ['Generator', 'Normal Lead', 'Abandoned Lead',
            f'Abandoned-lead decision boundary (prob = {threshold})'],
           loc='lower center', ncol=4, fontsize=9)

# เพิ่มคำอธิบายว่า opacity = ความมั่นใจของโมเดล (probability)
fig.text(0.5, 0.005,
         "Color opacity indicates model confidence (softmax probability): "
         "darker/more saturated = higher probability, faint = lower probability.",
         ha='center', fontsize=8.5, style='italic', color='dimgray')

plt.tight_layout(rect=[0, 0.08, 1, 0.95])
plt.show()