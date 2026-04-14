import torch
import pathlib
import cv2
import numpy as np
from fastai.vision.all import *

# 1. ตั้งค่าพื้นฐาน
class_names = ["background", "generator", "lead", "abandoned_lead"]
path_weights = "C:/CIEDID_data/AbdnL/models/best_seg.pth"
path_img = "C:/CIEDID_data/AbdnL/data/a_20.png" # เปลี่ยนเป็นรูปที่อยากลอง

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

# 5. แสดงผลลัพธ์
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img)
ax[0].set_title("Original X-ray")
ax[0].axis("off")

# ใช้ vmin=0, vmax=3 เพื่อให้สี Class 3 (Abandoned Lead) เป็นสีแดงเสมอ
ax[1].imshow(mask, vmin=0, vmax=3, cmap='jet') 
ax[1].set_title("Predicted Mask (Red = Abandoned Lead)")
ax[1].axis("off")

plt.show()