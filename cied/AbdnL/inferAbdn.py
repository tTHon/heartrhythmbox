# inferAbdn.py
# This script performs inference on a single X-ray image using a pre-trained UNet model for segmentation.
# This script loads the pretrained .pth.

import torch
import pathlib
import cv2
import numpy as np
from fastai.vision.all import *

# 1. ตั้งค่าพื้นฐาน
class_names = ["background", "generator", "lead", "abandoned_lead"]
path_weights = "C:/CIEDID_data/AbdnL/models/best_seg.pth"
path_img = "C:/CIEDID_data/AbdnL/data/n_x12.png" # เปลี่ยนเป็นรูปที่อยากลอง

# 2. สร้างโครงสร้างโมเดล (ต้องเหมือนตอนเทรนเป๊ะๆ)
# ใช้ ResNet50 และ n_out=4 ตามที่คุณเทรนไว้
dls = SegmentationDataLoaders.from_label_func(
    pathlib.Path("."), bs=1, fnames=[path_img], 
    label_func=lambda x: x, codes=class_names, item_tfms=Resize(512)
)
learn = unet_learner(dls, resnet50, n_out=4)

# 3. โหลดค่าน้ำหนักจากไฟล์ .pth
# ใช้ map_location เพื่อให้รันได้ทั้งเครื่องที่มี GPU และไม่มี
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learn.model.load_state_dict(torch.load(path_weights, map_location=device))
learn.model.eval()
print("✅ Loaded best_seg.pth successfully!")

# 4. ทำนายผล (Inference)
img = PILImage.create(path_img)
with torch.no_grad():
    # predict จะคืนค่าออกมา 3 อย่าง: mask ที่เป็นเลข class, แผนที่ความน่าจะเป็น
    mask, _, probs = learn.predict(img)

# --- ส่วนที่แก้ไข: การใช้ Probability Threshold ---
threshold = 0.7  # ปรับได้ตามต้องการ (เช่น 0.7 หรือ 0.85)

# 1. ดึง mask พื้นฐานที่โมเดลเลือกคลาสที่เด่นที่สุดมาให้ (มีทั้ง 0, 1, 2, 3)
final_mask = mask.numpy().copy()

# 2. กรองเฉพาะ Class 3 (Abandoned Lead) ด้วย Threshold
# ถ้าความมั่นใจใน Class 3 ไม่ถึง 0.85 ให้สั่ง "ลบ" เฉพาะสีแดงทิ้ง
# โดยที่ Class 1 (Generator) และ 2 (Lead) จะยังอยู่ที่เดิมไม่โดนลบ
abdn_probs = probs[3].numpy()
low_confidence_abdn = (final_mask == 3) & (abdn_probs < threshold)
final_mask[low_confidence_abdn] = 0

# 3. (เสริม) ถ้าอยากให้คลาสอื่นๆ มั่นใจมากขึ้นด้วย 
# สามารถสั่งให้แสดงเฉพาะ Class 1 ที่มั่นใจ > 0.5 ได้เช่นกัน (ป้องกันสีฟ้าหาย)
#generator_probs = probs[1].numpy()
#final_mask[(final_mask == 1) & (generator_probs < 0.5)] = 0
# ------------------------------------------------------------------

# 5. แสดงผลลัพธ์
# ใส่หลังจากการทำ final_mask เสร็จ
found_classes = np.unique(final_mask)
print(f"Classes found in this image: {found_classes}")
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img)
ax[0].set_title("Original X-ray")
ax[0].axis("off")

# ใช้ vmin=0, vmax=3 เพื่อให้สี Class 3 (Abandoned Lead) เป็นสีแดงเสมอ
ax[1].imshow(final_mask, vmin=0, vmax=3, cmap='jet') 
ax[1].set_title(f"Filtered Mask (Confidence > {threshold})")
ax[1].axis("off")

plt.show()