import torch
import pathlib
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from fastai.vision.all import *
from PIL import Image

# ==========================================================
# 1. SETTINGS
# ==========================================================
path_img_folder = "C:/CIEDID_data/AbdnL/data"
path_weights = "C:/CIEDID_data/AbdnL/models/fold_0/best_gen.pth"
IMG_Size = 512
Gen_CLASS_IDX = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================================
# 2. LOAD MODEL & PREPARE IMAGE
# ==========================================================
# สร้าง Learner แบบเดียวกับที่ใช้เทรน
img_files = get_image_files(path_img_folder)
dls = SegmentationDataLoaders.from_label_func(
    pathlib.Path("."), bs=1, fnames=img_files[:1], 
    label_func=lambda x: x, codes=["B","G","L","A"], 
    item_tfms=Resize(IMG_Size, method='pad')
)
learn = unet_learner(dls, resnet50, n_out=4)
learn.model.load_state_dict(torch.load(path_weights, map_location=device))
learn.model.to(device).eval()

# Preprocessing Pipeline (ต้องตรงกับตอนเทรน)
timg_pipe = Pipeline([PILImage.create, Resize(IMG_Size, method='pad', pad_mode='zeros'), ToTensor(), IntToFloatTensor()])
mean = torch.tensor([0.502668]*3, device=device).view(3, 1, 1)
std = torch.tensor([0.240966]*3, device=device).view(3, 1, 1)

# ==========================================================
# 3. VISUALIZATION (Random 3 Images)
# ==========================================================
random_imgs = random.sample(img_files, 3)
fig, axes = plt.subplots(3, 2, figsize=(12, 15))

for i, img_path in enumerate(random_imgs):
    # 1. เตรียมรูป
    timg = timg_pipe(img_path).to(device)
    timg_norm = (timg - mean) / std
    
    # 2. Inference
    with torch.no_grad():
        output = learn.model(timg_norm.unsqueeze(0))
        probs = F.softmax(output, dim=1)[0].cpu()
        # ดึงเฉพาะ Probability Map ของคลาส Abandoned Lead
        gen_prob_map = probs[Gen_CLASS_IDX].numpy()
    
    # 3. Plot รูปต้นฉบับ
    # แปลง tensor กลับเป็น numpy เพื่อโชว์รูป
    raw_img = timg.permute(1, 2, 0).cpu().numpy()
    axes[i, 0].imshow(raw_img)
    axes[i, 0].set_title(f"Original Image: {img_path.name}")
    axes[i, 0].axis('off')
    
    # 4. Plot Heatmap (สิ่งที่โมเดลเห็น)
    # ใช้สี 'hot' หรือ 'jet' เพื่อเน้นจุดที่มีความมั่นใจสูง
    im = axes[i, 1].imshow(gen_prob_map, cmap='jet', vmin=0, vmax=1)
    axes[i, 1].set_title(f"Model Confidence (Generator)")
    axes[i, 1].axis('off')
    plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()