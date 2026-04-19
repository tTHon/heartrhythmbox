import sys
import pathlib
import platform
import os
import numpy as np
import skimage.measure
from PIL import Image
from fastai.vision.all import *

# ==========================================================
# STEP 1: COMPATIBILITY PATCHES (แก้ไข Bug ของ Library & Windows)
# ==========================================================
def apply_patches():
    # 1. Patch สำหรับ NumPy 2.0+
    if not hasattr(np, 'int'): np.int = int
    
    # 2. Patch สำหรับ FastAI AMPMode (Version Mismatch)
    import fastai.callback.fp16
    if not hasattr(fastai.callback.fp16, 'AMPMode'):
        class AMPMode:
            def __init__(self, *args, **kwargs): pass 
        fastai.callback.fp16.AMPMode = AMPMode
        sys.modules['fastai.callback.fp16'].AMPMode = AMPMode

    # 3. Patch สำหรับการโหลดไฟล์ .pkl ข้ามระบบปฏิบัติการ
    if platform.system() == 'Windows':
        pathlib.PosixPath = pathlib.WindowsPath
    else:
        pathlib.WindowsPath = pathlib.PosixPath

# ==========================================================
# STEP 2: HELPER FUNCTIONS (ฟังก์ชันช่วยประมวลผลภาพ)
# ==========================================================
def resize_img(img, small_ax=1024):
    scale_f = small_ax / min(img.size)
    return img.resize((int(np.floor(img.size[0] * scale_f)), int(np.floor(img.size[1] * scale_f))))

def fix_bbox(bbox_org, img_shape, minsize=160):
    minr, minc, maxr, maxc = bbox_org
    # Margin 5% รอบตัวเครื่อง
    dr, dc = int((maxr-minr)*0.05), int((maxc-minc)*0.05)
    minr, minc, maxr, maxc = minr-dr, minc-dc, maxr+dr, maxc+dc
    # ปรับเป็นสี่เหลี่ยมจัตุรัส
    h, w = maxr-minr, maxc-minc
    max_side = max(h, w, minsize)
    dr, dc = max_side-h, max_side-w
    minr, maxr = max(0, minr - dr//2), min(img_shape[0], maxr + dr//2)
    minc, maxc = max(0, minc - dc//2), min(img_shape[1], maxc + dc//2)
    return int(minr), int(minc), int(maxr), int(maxc)

# ==========================================================
# STEP 3: MAIN PIPELINE
# ==========================================================
def run_pipeline():
    # --- CONFIGURATION ---
    file_seg    = 'cied/segmentation.pkl'
    file_manuf  = 'cied/classification_manuf.pkl'
    file_model  = 'cied/classification_model.pkl'
    img_input   = 'cied/Dataset/A3 full.jpg'
    temp_crop   = 'cied/Dataset/temp_crop_process.jpg' # ไฟล์ชั่วคราวแก้ Bug .read()

    try:
        apply_patches()
        print("\n" + "="*55)
        print(" CIED AUTOMATED SEGMENTATION & CLASSIFICATION SYSTEM")
        print("="*55)
        
        # 1. LOAD MODELS
        print("1. กำลังโหลดโมเดล (Loading Learners)...")
        learn_seg   = load_learner(file_seg, cpu=True)
        learn_manuf = load_learner(file_manuf, cpu=True)
        learn_model = load_learner(file_model, cpu=True)

        # 2. SEGMENTATION
        print("2. กำลังวิเคราะห์ตำแหน่งอุปกรณ์ (Segmenting)...")
        raw_img = PILImage.create(img_input)
        img_1024 = resize_img(raw_img, 1024)
        img_1024_np = np.array(img_1024)
        
        _, mask, _ = learn_seg.predict(img_1024_np)
        mask_np = np.array(mask)
        mask_np[mask_np != 1] = 0 
        
        labeled = skimage.measure.label(mask_np)
        props = skimage.measure.regionprops(labeled)
        
        if len(props) == 0:
            print("❌ ไม่พบอุปกรณ์ในรูปภาพ กรุณาตรวจสอบความคมชัดของภาพ X-ray")
            return

        # 3. CROPPING
        print("3. กำลังตัดและเตรียมรูปภาพ (Cropping & Saving Temp)...")
        main_obj = props[np.argmax([p.area for p in props])]
        minr, minc, maxr, maxc = fix_bbox(main_obj.bbox, img_1024_np.shape)
        
        crop_arr = img_1024_np[minr:maxr, minc:maxc]
        crop_img = PILImage.create(crop_arr).resize((256, 256))
        
        # บันทึกลงดิสก์เพื่อแก้ Bug AttributeError: 'Image' object has no attribute 'read'
        crop_img.save(temp_crop)
        print(f"✅ บันทึกรูปประมวลผลชั่วคราวที่: {temp_crop}")

        # 4. CLASSIFICATION
        print("4. กำลังจำแนกผู้ผลิตและกลุ่มรุ่น (Classifying)...")
        
        # ส่ง Path ของไฟล์ชั่วคราวให้โมเดลแทนการส่ง Image Object
        manuf_res, _, manuf_probs = learn_manuf.predict(temp_crop)
        model_res, _, model_probs = learn_model.predict(temp_crop)

        # 5. DISPLAY RESULTS
        manuf_conf = manuf_probs.max().item() * 100
        model_conf = model_probs.max().item() * 100

        print("\n" + "╔" + "═"*53 + "╗")
        print(f"║ ผลการวิเคราะห์: {pathlib.Path(img_input).name:<32} ║")
        print("╠" + "═"*53 + "╣")
        print(f"║ ผู้ผลิต (Manufacturer) : {str(manuf_res):<20} ({manuf_conf:>6.2f}%) ║")
        print(f"║ กลุ่มรุ่น (Model Group) : {str(model_res):<20} ({model_conf:>6.2f}%) ║")
        print("╚" + "═"*53 + "╝")

        # (Optional) ลบไฟล์ชั่วคราว
        # if os.path.exists(temp_crop): os.remove(temp_crop)

    except Exception as e:
        print(f"\n❌ [CRITICAL ERROR]: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_pipeline()