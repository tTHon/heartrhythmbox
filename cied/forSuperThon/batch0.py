import sys
import pathlib
import platform
import os
import numpy as np
import skimage.measure
from PIL import Image
from fastai.vision.all import *
import time

# ==========================================================
# STEP 1: COMPATIBILITY PATCHES (ตามไฟล์ต้นฉบับของคุณ)
# ==========================================================
def apply_patches():
    if not hasattr(np, 'int'): np.int = int
    import fastai.callback.fp16
    if not hasattr(fastai.callback.fp16, 'AMPMode'):
        class AMPMode:
            def __init__(self, *args, **kwargs): pass 
        fastai.callback.fp16.AMPMode = AMPMode
        sys.modules['fastai.callback.fp16'].AMPMode = AMPMode
    if platform.system() == 'Windows':
        pathlib.PosixPath = pathlib.WindowsPath
    else:
        pathlib.WindowsPath = pathlib.PosixPath

# ==========================================================
# STEP 2: HELPER FUNCTIONS (ยึดตามไฟล์ที่คุณให้ 100%)
# ==========================================================
def resize_img(img, small_ax=1024):
    scale_f = small_ax / min(img.size)
    return img.resize((int(np.floor(img.size[0] * scale_f)), int(np.floor(img.size[1] * scale_f))))

def fix_bbox(bbox_org, img_shape, minsize=160):
    minr, minc, maxr, maxc = bbox_org
    dr, dc = int((maxr-minr)*0.05), int((maxc-minc)*0.05)
    minr, minc, maxr, maxc = minr-dr, minc-dc, maxr+dr, maxc+dc
    h, w = maxr-minr, maxc-minc
    max_side = max(h, w, minsize)
    dr, dc = max_side-h, max_side-w
    minr, maxr = max(0, minr - dr//2), min(img_shape[0], maxr + dr//2)
    minc, maxc = max(0, minc - dc//2), min(img_shape[1], maxc + dc//2)
    return int(minr), int(minc), int(maxr), int(maxc)

# ==========================================================
# STEP 3: BATCH PIPELINE (ครอบทับโครงสร้างเดิมของคุณ)
# ==========================================================
def run_batch_pipeline(input_folder, output_folder):
    file_seg    = 'cied/segmentation.pkl'
    file_manuf  = 'cied/classification_manuf.pkl'
    file_model  = 'cied/classification_model.pkl'
    temp_crop   = 'cied/Dataset/temp_batch_crop.jpg'

    try:
        apply_patches()
        in_dir, out_dir = Path(input_folder), Path(output_folder)
        out_dir.mkdir(exist_ok=True, parents=True)

        print("\n" + "="*55)
        print(" CIED BATCH SYSTEM (Using your original Logic)")
        print("="*55)

        # ตรวจสอบ Device
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            device_type = "GPU"
        else:
            device_name = "CPU Standard"
            device_type = "CPU"

        print("\n" + "="*65)
        print(f" CIED BATCH SYSTEM - RUNNING ON: [{device_type}]")
        print(f" Device Name: {device_name}")
        print("="*65)
        
        # 1. LOAD MODELS
        start_load = time.time()
        print("1. กำลังโหลดโมเดล ...")
        learn_seg   = load_learner(file_seg)
        learn_manuf = load_learner(file_manuf)
        learn_model = load_learner(file_model)

        # 2. GET FILES
        files = get_image_files(in_dir)
        print(f"ตรวจพบรูปภาพทั้งหมด: {len(files)} รูป")
        
        total_start_time = time.time()
        success_count = 0

        for i, fname in enumerate(files):
            img_start_time = time.time() # เริ่มจับเวลาต่อรูป
            print(f"--- [{i+1}/{len(files)}] {fname.name} ---")
            
            # 2.1 SEGMENTATION (Logic ตามไฟล์คุณ)
            raw_img = PILImage.create(fname)
            img_1024 = resize_img(raw_img, 1024)
            img_1024_np = np.array(img_1024)
            
            _, mask, _ = learn_seg.predict(img_1024_np)
            mask_np = np.array(mask)
            mask_np[mask_np != 1] = 0 
            
            labeled = skimage.measure.label(mask_np)
            props = skimage.measure.regionprops(labeled)
            
            if len(props) == 0:
                print(f"⚠️ ไม่พบอุปกรณ์ในรูป: {fname.name}")
                continue

            # 2.2 CROPPING (Logic ตามไฟล์คุณ)
            main_obj = props[np.argmax([p.area for p in props])]
            minr, minc, maxr, maxc = fix_bbox(main_obj.bbox, img_1024_np.shape)
            
            crop_arr = img_1024_np[minr:maxr, minc:maxc]
            crop_img = PILImage.create(crop_arr).resize((256, 256))
            
            # บันทึกเป็น Temp เพื่อแก้ Error 'Image' object has no attribute 'read'
            crop_img.save(temp_crop)
            # เซฟไฟล์ลง Results เพื่อเก็บผลลัพธ์
            crop_img.save(out_dir / f"crop_{fname.name}")

            # 2.3 CLASSIFICATION (Logic ตามไฟล์คุณ)
            manuf_res, _, manuf_probs = learn_manuf.predict(temp_crop)
            model_res, _, model_probs = learn_model.predict(temp_crop)

            img_end_time = time.time() # จบเวลาต่อรูป
            duration = img_end_time - img_start_time

            manuf_conf = manuf_probs.max().item() * 100
            model_conf = model_probs.max().item() * 100

            print(f"   Manufacturer: {manuf_res} ({manuf_conf:.1f}%)")
            print(f"   Model Group : {model_res} ({model_conf:.1f}%)")
            success_count += 1
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time

        print(f"\n✅ กระบวนการเสร็จสิ้น!")
        print(f"   จำนวนรูปภาพที่ประมวลผลสำเร็จ: {success_count}/{len(files)}")
        print(f"   เวลาทั้งหมด: {total_duration:.2f} วินาที")

    except Exception as e:
        print(f"\n❌ [ERROR]: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # ใส่ Path โฟลเดอร์ Input และ Output
    INPUT_PATH = 'cied/Dataset/test'
    OUTPUT_PATH = 'cied/Dataset/results'
    
    run_batch_pipeline(INPUT_PATH, OUTPUT_PATH)