import sys
import pathlib
import platform
import os
import numpy as np
import skimage.measure
from PIL import Image
from fastai.vision.all import *
import time
import csv
import warnings

# ==========================================================
# STEP 1: SETUP & PATCHES (แก้ไข Bug ของ Library)
# ==========================================================
def apply_patches():
    # ปิด Warning เพื่อความสะอาดของหน้าจอ
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Patch สำหรับ NumPy 2.0+
    if not hasattr(np, 'int'): np.int = int
    
    # Patch สำหรับ FastAI AMPMode
    import fastai.callback.fp16
    if not hasattr(fastai.callback.fp16, 'AMPMode'):
        class AMPMode:
            def __init__(self, *args, **kwargs): pass 
        fastai.callback.fp16.AMPMode = AMPMode
        sys.modules['fastai.callback.fp16'].AMPMode = AMPMode

    # Patch สำหรับการโหลดไฟล์ .pkl ข้าม OS
    if platform.system() == 'Windows':
        pathlib.PosixPath = pathlib.WindowsPath
    else:
        pathlib.WindowsPath = pathlib.PosixPath

# ==========================================================
# STEP 2: HELPER FUNCTIONS (ยึดตาม Logic เดิมที่คุณต้องการ)
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
# STEP 3: MAIN BATCH PIPELINE
# ==========================================================
def run_batch_pipeline(input_folder, output_folder):
    # --- CONFIGURATION ---
    file_seg    = 'cied/segmentation.pkl'
    file_manuf  = 'cied/classification_manuf.pkl'
    file_model  = 'cied/classification_model.pkl'
    temp_crop   = 'cied/Dataset/temp_batch_crop.jpg'
    csv_output  = Path(output_folder) / 'classification_results.csv'

    try:
        apply_patches()
        in_dir, out_dir = Path(input_folder), Path(output_folder)
        out_dir.mkdir(exist_ok=True, parents=True)

        # เช็คการใช้งาน GPU/CPU
        device_status = "GPU" if torch.cuda.is_available() else "CPU"
        print("\n" + "="*65)
        print(f" CIED BATCH SYSTEM - RUNNING ON: [{device_status}]")
        if torch.cuda.is_available():
            print(f" Device: {torch.cuda.get_device_name(0)}")
        print("="*65)
        
        # 1. LOAD MODELS
        start_load = time.time()
        print("1. กำลังโหลดโมเดล (Loading Learners)...")
        # โหลดลง Device อัตโนมัติ (ไม่ใส่ cpu=True)
        learn_seg   = load_learner(file_seg)
        learn_manuf = load_learner(file_manuf)
        learn_model = load_learner(file_model)
        print(f"✅ โหลดโมเดลเสร็จสิ้น (ใช้เวลา: {time.time() - start_load:.2f} วินาที)")

        # บังคับย้ายลง GPU (ใช้ Try-Except เพื่อป้องกันการ Crash ของ DEBUG)
        try:
            learn_seg.model.to('cuda')
            learn_manuf.model.to('cuda')
            learn_model.model.to('cuda')
            print("🚀 บังคับโมเดลลง GPU สำเร็จ!")
        except:
            # ถ้าเรียก .model ไม่ได้ ให้ใช้คำสั่ง .to('cuda') กับตัว Learner ตรงๆ
            learn_seg.to('cuda')
            learn_manuf.to('cuda')
            learn_model.to('cuda')
            print("🚀 ย้าย Learner ลง GPU สำเร็จ!")

        print(f"✅ โหลดและเตรียม VRAM สำเร็จ (ใช้เวลา: {time.time() - start_load:.2f} วินาที)")

        # 2. GET FILES
        files = get_image_files(in_dir)
        print(f"ตรวจพบรูปภาพทั้งหมด: {len(files)} รูป")
        print("-" * 65)

        results_list = []
        total_start_time = time.time()
        success_count = 0

        # 3. START PROCESSING LOOP
        for i, fname in enumerate(files):
            img_start_time = time.time()
            print(f"[{i+1}/{len(files)}] {fname.name} ", end="")
            
            # --- SEGMENTATION ---
            raw_img = PILImage.create(fname)
            img_1024 = resize_img(raw_img, 1024)
            img_1024_np = np.array(img_1024)
            
            _, mask, _ = learn_seg.predict(img_1024_np)
            mask_np = np.array(mask)
            mask_np[mask_np != 1] = 0 
            
            labeled = skimage.measure.label(mask_np)
            props = skimage.measure.regionprops(labeled)
            
            if len(props) == 0:
                print(f"\n⚠️ ไม่พบอุปกรณ์ในรูป")
                continue

            # --- CROPPING ---
            main_obj = props[np.argmax([p.area for p in props])]
            minr, minc, maxr, maxc = fix_bbox(main_obj.bbox, img_1024_np.shape)
            
            crop_arr = img_1024_np[minr:maxr, minc:maxc]
            crop_img = PILImage.create(crop_arr).resize((256, 256))
            
            # Save Temp for Classification
            crop_img.save(temp_crop)
            # Save for Review
            crop_img.save(out_dir / f"crop_{fname.name}")

            # --- CLASSIFICATION ---
            manuf_res, _, manuf_probs = learn_manuf.predict(temp_crop)
            model_res, _, model_probs = learn_model.predict(temp_crop)

            # --- TIMER & DATA ---
            duration = time.time() - img_start_time
            manuf_conf = manuf_probs.max().item() * 100
            model_conf = model_probs.max().item() * 100
            
            print(f"| {manuf_res} ({manuf_conf:.1f}%) | {duration:.2f} วินาที")
            
            # เก็บข้อมูลเพื่อลง CSV
            results_list.append([
                fname.name, manuf_res, f"{manuf_conf:.2f}", 
                model_res, f"{model_conf:.2f}", f"{duration:.2f}"
            ])
            success_count += 1

        # 4. EXPORT TO CSV
        with open(csv_output, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['Filename', 'Manufacturer', 'Conf_Manuf(%)', 'Model_Group', 'Conf_Model(%)', 'Processing_Time(s)'])
            writer.writerows(results_list)

        # 5. FINAL SUMMARY
        total_duration = time.time() - total_start_time
        print("-" * 65)
        print(f" สรุปการทำงาน (Summary)")
        print(f" > จำนวนที่สำเร็จ: {success_count} / {len(files)} รูป")
        print(f" > เวลารวมทั้งหมด: {total_duration:.2f} วินาที")
        if success_count > 0:
            print(f" > เฉลี่ยต่อรูป   : {total_duration / success_count:.2f} วินาที")
        print(f" > บันทึก CSV ไปที่: {csv_output}")
        print("="*65)

    except Exception as e:
        print(f"\n❌ [CRITICAL ERROR]: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # ปรับ Path ของคุณตรงนี้
    INPUT_PATH  = 'cied/Dataset/test'
    OUTPUT_PATH = 'cied/Dataset/results'
    
    run_batch_pipeline(INPUT_PATH, OUTPUT_PATH)