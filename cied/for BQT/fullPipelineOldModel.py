import sys
import pathlib
import platform
import numpy as np
from PIL import Image

# ==========================================================
# STEP 1: COMPATIBILITY PATCHES (MUST BE AT THE VERY TOP)
# ==========================================================

# --- 1. Patch for NumPy 'int' attribute error ---
if not hasattr(np, 'int'):
    np.int = int

# 1. Patch for 'AMPMode' (Fixes: "AMPMode() takes no arguments")
import fastai.callback.fp16
if not hasattr(fastai.callback.fp16, 'AMPMode'):
    class AMPMode:
        def __init__(self, *args, **kwargs): 
            pass # Accept and ignore any saved training arguments
    
    fastai.callback.fp16.AMPMode = AMPMode
    sys.modules['fastai.callback.fp16'].AMPMode = AMPMode

# 2. Patch for Windows/Linux Path compatibility (Fixes: "PosixPath" error)
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# Now it is safe to import fastai
from fastai.vision.all import *

#full process, old model, one file at a time.

import pathlib
import platform
import numpy as np
import skimage.measure
from fastai.vision.all import *

# 1. หลอกระบบเรื่อง Path สำหรับข้าม OS
plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

# --- ฟังก์ชันช่วยจัดการรูปภาพ (จาก Notebook 1) ---
def resize_img(img, small_ax=1024):
    scale_f = small_ax / min(img.size)
    return img.resize((int(np.floor(img.size[0] * scale_f)), int(np.floor(img.size[1] * scale_f))))

def fix_bbox(bbox_org, img_shape, minsize=160):
    minr, minc, maxr, maxc = bbox_org
    # เพิ่ม Margin รอบตัวเครื่อง 5%
    dr, dc = int((maxr-minr)*0.05), int((maxc-minc)*0.05)
    minr, minc, maxr, maxc = minr-dr, minc-dc, maxr+dr, maxc+dc
    # ปรับให้เป็นสี่เหลี่ยมจัตุรัสและไม่เกินขอบเขตภาพ
    h, w = maxr-minr, maxc-minc
    max_side = max(h, w, minsize)
    dr, dc = max_side-h, max_side-w
    minr, maxr = max(0, minr - dr//2), min(img_shape[0], maxr + dr//2)
    minc, maxc = max(0, minc - dc//2), min(img_shape[1], maxc + dc//2)
    return int(minr), int(minc), int(maxr), int(maxc)

# 2. กำหนดชื่อไฟล์โมเดลทั้ง 3 ตัว
file_seg   = 'cied/segmentation.pkl'  # เพิ่มโมเดลตัวที่ 1
file_manuf = 'cied/classification_manuf.pkl'
file_model = 'cied/classification_model.pkl'

try:
    print("Loading all 3 models...")
    learn_seg   = load_learner(file_seg)
    learn_manuf = load_learner(file_manuf)
    learn_model = load_learner(file_model)

    # 4. โหลดรูป X-ray ต้นฉบับ
    img_path = 'cied/Dataset/A3 full.jpg'
    raw_img = PILImage.create(img_path)
    
    # --- STEP 1: SEGMENTATION ---
    print("Step 1: Segmenting device...")
    # ปรับขนาดภาพเป็น 1024 ก่อนหาตำแหน่ง (ตามที่โมเดลถูกเทรนมา)
    img_1024 = resize_img(raw_img, 1024)
    img_1024_np = np.array(img_1024)
    
    # Predict Mask
    _, mask, _ = learn_seg.predict(img_1024_np)
    mask_np = np.array(mask)
    mask_np[mask_np != 1] = 0 # เลือกเฉพาะ class ตัวเครื่อง
    
    # หาวัตถุที่ใหญ่ที่สุด
    labeled = skimage.measure.label(mask_np)
    props = skimage.measure.regionprops(labeled)
    
    if len(props) == 0:
        print("Error: ไม่พบอุปกรณ์ในรูปภาพ")
    else:
        # --- STEP 2: CROPPING ---
        print("Step 2: Cropping device...")
        main_obj = props[np.argmax([p.area for p in props])]
        minr, minc, maxr, maxc = fix_bbox(main_obj.bbox, img_1024_np.shape)
        
        # ตัดรูป (Crop) และปรับเป็น 256x256 เพื่อส่งต่อ
        crop_arr = img_1024_np[minr:maxr, minc:maxc]
        crop_img = PILImage.create(crop_arr).resize((256, 256))
        
        # --- STEP 3: CLASSIFICATION ---
        print("Step 3: Classifying...")
        # ทายยี่ห้อ
        manuf_name, _, manuf_probs = learn_manuf.predict(crop_img)
        manuf_conf = manuf_probs.max() * 100

        # ทายรุ่น
        model_name, _, model_probs = learn_model.predict(crop_img)
        model_conf = model_probs.max() * 100

        # 6. แสดงผลลัพธ์
        print("-" * 40)
        print(f"IMAGE: {img_path}")
        print(f"DETECTED BBOX : {minr, minc, maxr, maxc}")
        print(f"MANUFACTURER  : {manuf_name} ({manuf_conf:.2f}%)")
        print(f"MODEL GROUP   : {model_name} ({model_conf:.2f}%)")
        print("-" * 40)
        
        # (Option) บันทึกรูปที่ตัดออกมาดู
        crop_img.save('cied/Dataset/debug_crop.png')

except FileNotFoundError:
    print("Error: หาไฟล์โมเดลไม่พบ ตรวจสอบ Path ของไฟล์")
except Exception as e:
    print(f"An error occurred: {e}")
