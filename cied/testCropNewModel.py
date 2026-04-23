import torch
import pathlib
import numpy as np
import skimage.measure
import matplotlib.pyplot as plt
from fastai.vision.all import *
import scipy.ndimage as ndimage
import cv2

# ==========================================================
# 1. CONFIGURATION & PARAMETERS
# ==========================================================
# ใส่ Path ของคุณที่นี่
path_weights = "C:/CIEDID_data/AbdnL/models/best_seg.pth"
path_img = "C:/CIEDID_data/AbdnL/data/90.png" 

IMG_Size = 512       # ปรับตามตัวแปร IMG_Size ที่คุณต้องการ
Crop_border = 0.05    # สัดส่วนขอบที่เพิ่มรอบ Generator
class_names = ["background", "generator", "lead", "abandoned_lead"]

# ==========================================================
# 2. HELPER FUNCTIONS
# ==========================================================
def fix_bbox(bbox_org, img_shape, border, minsize=160):
    minr, minc, maxr, maxc = bbox_org
    # เพิ่ม Border รอบวัตถุตาม % ที่กำหนด
    dr, dc = int((maxr-minr)*border), int((maxc-minc)*border)
    minr, minc, maxr, maxc = minr-dr, minc-dc, maxr+dr, maxc+dc
    
    # ปรับให้เป็นจัตุรัสและไม่เกินขอบภาพ
    h, w = maxr-minr, maxc-minc
    max_side = max(h, w, minsize)
    dr, dc = max_side-h, max_side-w
    minr, maxr = max(0, minr - dr//2), min(img_shape[0], maxr + dr//2)
    minc, maxc = max(0, minc - dc//2), min(img_shape[1], maxc + dc//2)
    return int(minr), int(minc), int(maxr), int(maxc)

def apply_clahe(img_input):

    """ตรวจสอบว่าภาพควรทำ CLAHE หรือไม่ โดยดูจากค่าสถิติความสว่าง
    - std_threshold: ถ้า Std ต่ำกว่านี้ แสดงว่าภาพแบน (Low Contrast) ควรทำ CLAHE
    - mean_range: ถ้า Mean อยู่นอกช่วงนี้ แสดงว่าภาพมืดหรือสว่างไป ควรทำ CLAHE
    """
    # 1. อ่านภาพแบบ Grayscale
    img = cv2.imread(str(img_input), 0)

    # 2. คำนวณค่าสถิติพื้นฐาน
    mean = np.mean(img)
    std = np.std(img)
    
    # 3. ตัดสินใจ (Logic)
    # ถ้า Standard Deviation ต่ำ (ภาพมัว/แบน) หรือ Mean ไม่อยู่ในจุดที่เหมาะสม
    std_threshold=40.0
    mean_range=(80, 170)
    apply = False
    if std < std_threshold:
        apply = True
    if mean < mean_range[0] or mean > mean_range[1]:
        apply = True
    
    img_clahe = img_input  # Default: original image
    if apply:
        print(f"Applying CLAHE: mean={mean:.2f}, std={std:.2f} (Low Contrast or Unbalanced Brightness)")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_clahe = clahe.apply(img)

    return img_clahe

# ==========================================================
# 3. LOAD MODEL & PREDICT
# ==========================================================
dls = SegmentationDataLoaders.from_label_func(
    pathlib.Path("."), bs=1, fnames=[path_img], 
    label_func=lambda x: x, codes=class_names, 
    item_tfms=Resize(IMG_Size, method='pad', pad_mode='zeros') # รักษา Aspect Ratio
)
learn = unet_learner(dls, resnet50, n_out=4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learn.model.load_state_dict(torch.load(path_weights, map_location=device))
learn.model.eval()

raw_img = PILImage.create(apply_clahe(path_img))
with torch.no_grad():
    mask, _, probs = learn.predict(raw_img)

mask_np = mask.numpy()

# ==========================================================
# 4. MORPHOLOGICAL PROCESSING (Dilation, Erosion, Fill Holes)
# ==========================================================
# ดึงเฉพาะคลาส Generator (1)
gen_mask = np.where(mask_np == 1, 1, 0).astype(np.uint8)
if gen_mask.ndim > 2: gen_mask = gen_mask.squeeze()

struct = ndimage.generate_binary_structure(2, 2) 

# ทำ Dilation และ Erosion อย่างละ 3 รอบตามโค้ดล่าสุดของคุณ
gen_mask_processed = ndimage.binary_dilation(gen_mask, structure=struct, iterations=3).astype(np.uint8)
gen_mask_processed = ndimage.binary_erosion(gen_mask_processed, structure=struct, iterations=3).astype(np.uint8)
gen_mask_processed = ndimage.binary_fill_holes(gen_mask_processed).astype(np.uint8)

# ==========================================================
# 5. OBJECT SELECTION & COORDINATE MAPPING
# ==========================================================
labeled = skimage.measure.label(gen_mask_processed)
props = skimage.measure.regionprops(labeled)

# กรองด้วย Area Threshold ที่สัมพันธ์กับ IMG_Size
MIN_GEN_AREA = (IMG_Size * IMG_Size) * 0.0038 
large_props = [p for p in props if p.area > MIN_GEN_AREA]

if len(large_props) > 0:
    main_obj = large_props[np.argmax([p.area for p in large_props])]
    
    # คำนวณพิกัดกลับไปยังรูปต้นฉบับ (Center Padding Logic)
    orig_w, orig_h = raw_img.size
    scale = max(orig_w, orig_h) / IMG_Size
    pad_y = (IMG_Size - (orig_h / scale)) / 2 if orig_h < orig_w else 0
    pad_x = (IMG_Size - (orig_w / scale)) / 2 if orig_w < orig_h else 0

    minr, minc, maxr, maxc = main_obj.bbox
    real_bbox = (
        int((minr - pad_y) * scale), int((minc - pad_x) * scale),
        int((maxr - pad_y) * scale), int((maxc - pad_x) * scale)
    )

    # ตัดรูปจาก Original Image
    raw_np = np.array(raw_img)
    f_minr, f_minc, f_maxr, f_maxc = fix_bbox(real_bbox, raw_np.shape, Crop_border)
    crop_arr = raw_np[f_minr:f_maxr, f_minc:f_maxc]
else:
    print("❌ ไม่พบ Generator ที่ขนาดใหญ่พอ")
    crop_arr = None

# ==========================================================
# 6. VISUALIZATION
# ==========================================================
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# 1. แสดงภาพที่โมเดลเห็น (Padded) และ Mask ที่ประมวลผลแล้ว
img_padded = learn.dls.after_item(raw_img)
if isinstance(img_padded, torch.Tensor):
    img_padded = img_padded.permute(1, 2, 0).cpu().numpy()

ax[0].imshow(img_padded)
# Overlay Mask ที่ผ่าน Dilation/Erosion แล้ว
masked_gen = np.ma.masked_where(gen_mask_processed == 0, gen_mask_processed)
ax[0].imshow(masked_gen, cmap='cool', alpha=0.6)
ax[0].set_title(f"Processed Mask Overlay\n(Size: {IMG_Size}x{IMG_Size})")
ax[0].axis("off")

# 2. แสดงเฉพาะ Mask ที่ได้เพื่อดูความสมบูรณ์
ax[1].imshow(gen_mask_processed, cmap='gray')
ax[1].set_title("Binary Mask after\nDilate -> Erode -> Fill")
ax[1].axis("off")

# 3. แสดงผลลัพธ์การ Crop จากภาพต้นฉบับ
if crop_arr is not None:
    ax[2].imshow(crop_arr)
    ax[2].set_title(f"Final Crop Result\n(From Original Resolution)")
else:
    ax[2].text(0.5, 0.5, 'No Object Detected', ha='center')
ax[2].axis("off")

plt.tight_layout()
plt.show()