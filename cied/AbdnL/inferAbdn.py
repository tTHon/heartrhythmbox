# inferAbdn.py
# This script performs inference on a single X-ray image using a pre-trained UNet model for segmentation.
# This script loads the pretrained .pth.
# assume all images were CLAHE'd

import torch
import pathlib
import cv2
import numpy as np
from fastai.vision.all import *

# 1. ตั้งค่าพื้นฐาน
class_names = ["background", "generator", "lead", "abandoned_lead"]
path_weights = "C:/CIEDID_data/AbdnL/models/best_seg.pth"
path_img = "cied\Dataset\micraWAbdn.png" # เปลี่ยนเป็นรูปที่อยากลอง
IMG_Size = 512  # ขนาดที่โมเดลใช้ตอนเทรน (ต้องเหมือนกันเป๊ะๆ)

# 2. สร้างโครงสร้างโมเดล (ต้องเหมือนตอนเทรนเป๊ะๆ)
# ใช้ ResNet50 และ n_out=4 ตามที่คุณเทรนไว้
dls = SegmentationDataLoaders.from_label_func(
    pathlib.Path("."), bs=1, fnames=[path_img], 
    label_func=lambda x: x, codes=class_names, item_tfms=Resize(IMG_Size, method='pad', pad_mode='zeros')
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
abdn_threshold = 0.5  # ปรับได้ตามต้องการ (เช่น 0.7 หรือ 0.85)
generator_threshold = 0.5  # สำหรับ Class 1 (Generator) ถ้าต้องการกรองด้วย

# 1. ดึง mask พื้นฐานที่โมเดลเลือกคลาสที่เด่นที่สุดมาให้ (มีทั้ง 0, 1, 2, 3)
final_mask = mask.numpy().copy()

# 2. กรองเฉพาะ Class 3 (Abandoned Lead) ด้วย Threshold
# ถ้าความมั่นใจใน Class 3 ไม่ถึง threshold ให้สั่ง "ลบ" เฉพาะสีแดงทิ้ง
# โดยที่ Class 1 (Generator) และ 2 (Lead) จะยังอยู่ที่เดิมไม่โดนลบ
abdn_probs = probs[3].numpy()
low_confidence_abdn = (final_mask == 3) & (abdn_probs < abdn_threshold)
final_mask[low_confidence_abdn] = 0
has_abdn = abdn_probs.max() > abdn_threshold

# กรองด้วย Threshold 
valid_abdn_pixels = (final_mask == 3) & (abdn_probs > abdn_threshold)
pixel_count = valid_abdn_pixels.sum().item()


# 3. (เสริม) ถ้าอยากให้คลาสอื่นๆ มั่นใจมากขึ้นด้วย 
# สามารถสั่งให้แสดงเฉพาะ Class 1 ที่มั่นใจ > 0.5 ได้เช่นกัน (ป้องกันสีฟ้าหาย)
generator_probs = probs[1].numpy()
final_mask[(final_mask == 1) & (generator_probs < generator_threshold)] = 0
valid_generator_pixels = (final_mask == 1) & (generator_probs > generator_threshold)
generator_pixel_count = valid_generator_pixels.sum().item()
# ------------------------------------------------------------------

# 5. แสดงผลลัพธ์
# ใส่หลังจากการทำ final_mask เสร็จ
found_classes = np.unique(final_mask)
print(f"Classes found in this image: {found_classes}")
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# 1. เตรียมภาพต้นฉบับให้เป็น 512 พร้อม Padding เหมือนที่โมเดลใช้
img_padded = learn.dls.after_item(img) # FastAI จะทำ Resize(512, method='pad') ให้
if isinstance(img_padded, torch.Tensor):
    # .permute(1, 2, 0) คือการย้ายแกนที่ 0 (Channel) ไปไว้ลำดับสุดท้าย
    img_padded = img_padded.permute(1, 2, 0).cpu().numpy()


# สร้าง Figure สำหรับเปรียบเทียบ Original และ Mask
fig, ax = plt.subplots(1, 2, figsize=(15, 7))

# ฝั่งซ้าย: ภาพต้นฉบับที่ทำ Padding แล้ว
ax[0].imshow(img_padded)
ax[0].set_title("Original X-ray (Padded 512x512)")
ax[0].axis("off")

# ฝั่งขวา: Mask ที่ผ่านการกรองแล้ว
ax[1].imshow(final_mask, vmin=0, vmax=3, cmap='jet')
ax[1].axis("off")

# เพิ่มรายละเอียดพิกเซลและ Threshold ตรง Title ของฝั่ง Mask
detail_text = (
    f"Abdn Filtered (Conf > {abdn_threshold}): {pixel_count} px\n"
    f"Gen Filtered (Conf > {generator_threshold}): {generator_pixel_count} px"
)
ax[1].set_title(detail_text, fontsize=10, color='red' if has_abdn else 'black')

plt.tight_layout()
plt.show()

# --- ส่วน Overlay (แสดงการซ้อนทับภาพ) ---
plt.figure(figsize=(10, 10))
plt.imshow(img_padded)

# ทำ Mask ให้โปร่งใสเฉพาะส่วนที่เป็น Background (0)
masked_final = np.ma.masked_where(final_mask == 0, final_mask)
plt.imshow(masked_final, cmap='jet', alpha=0.4, vmin=0, vmax=3)

plt.title(f"Overlay Analysis\n{detail_text}")
plt.axis("off")
plt.show()