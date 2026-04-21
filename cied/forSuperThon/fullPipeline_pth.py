import sys
import pathlib
import platform
import os
import torch
import numpy as np
import skimage.measure
from PIL import Image
from fastai.vision.all import *
import scipy.ndimage as ndiamge

# ==========================================================
# STEP 1: COMPATIBILITY PATCHES
# ==========================================================
def apply_patches():
    if not hasattr(np, 'int'): np.int = int
    
    if platform.system() == 'Windows':
        pathlib.PosixPath = pathlib.WindowsPath
    else:
        pathlib.WindowsPath = pathlib.PosixPath

# ==========================================================
# STEP 2: HELPER FUNCTIONS
# ==========================================================
def resize_img(img, small_ax=512):
    scale_f = small_ax / min(img.size)
    return img.resize((int(np.floor(img.size[0] * scale_f)), int(np.floor(img.size[1] * scale_f))))

def fix_bbox(bbox_org, img_shape, minsize=160):
    minr, minc, maxr, maxc = bbox_org
    dr, dc = int((maxr-minr)*0.01), int((maxc-minc)*0.01)
    minr, minc, maxr, maxc = minr-dr, minc-dc, maxr+dr, maxc+dc
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
    file_seg_weights = 'C:/CIEDID_data/AbdnL/models/best_seg.pth' 
    file_manuf       = 'C:/CIEDID_data/pkl/classification_manuf.pkl'
    file_model       = 'C:/CIEDID_data/pkl/classification_model.pkl'
    img_input        = 'cied/Dataset/A3 full.jpg'
    temp_crop        = 'cied/Dataset/temp_crop_pth.jpg'

    # Detection Thresholds
    PIXEL_MIN = 50
    PROB_THRESHOLD = 0.5
    CLASS_NAMES = ["background", "generator", "lead", "abandoned_lead"] 

    try:
        apply_patches()
        print("\n" + "="*55)
        print(" CIED AUTOMATED SEGMENTATION & CLASSIFICATION SYSTEM")
        print("="*55)
        
        # 1. LOAD MODELS
        print("1. กำลังโหลดโมเดล (Loading Weights & Learners)...")
        
        # Reconstruct architecture as per finetuneYN_custom.py
        dls_dummy = SegmentationDataLoaders.from_label_func(
            pathlib.Path("."), bs=1, fnames=[img_input], 
            label_func=lambda x: x, codes=CLASS_NAMES, item_tfms=Resize(512)
        )

        learn_seg = unet_learner(dls_dummy, resnet50, n_out=4)
        
        # Load state_dict
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        learn_seg.model.load_state_dict(torch.load(file_seg_weights, map_location=device))
        learn_seg.model.eval() # ตั้งค่าเป็น evaluation mode เพื่อความแม่นยำ
        
        learn_manuf = load_learner(file_manuf, cpu=True)
        learn_model = load_learner(file_model, cpu=True)

        # 2. SEGMENTATION & DETECTION
        print("2. กำลังวิเคราะห์ตำแหน่งอุปกรณ์ (Segmenting)...")
        raw_img = PILImage.create(img_input)
        
        # ใช้โมเดล Predict โดยตรงจาก PILImage (FastAI จะจัดการ Resize 512 ให้เองตาม dls)
        with torch.no_grad():
            mask, _, probs = learn_seg.predict(raw_img)
        
        mask_np = mask.numpy()
        
        # DEBUG CHECK
        unique, counts = np.unique(mask_np, return_counts=True)
        print(f"DEBUG - Classes found: {dict(zip(unique, counts))}")
        
        # --- Abandoned Lead Detection ---
        abdn_class_idx = 3
        # เพิ่ม .numpy() เพื่อแปลง Tensor ให้เป็น NumPy Array ก่อนนำไปคำนวณต่อ
        abdn_probs = probs[abdn_class_idx].numpy()

        # กรองด้วย Threshold 0.7
        valid_abdn_pixels = (mask_np == abdn_class_idx) & (abdn_probs > PROB_THRESHOLD)
        pixel_count = valid_abdn_pixels.sum().item()
        
        has_abdn = False
        abdn_final_prob = 0.0
        
        # ตรวจสอบเกณฑ์ Pixel_min 80
        if pixel_count >= PIXEL_MIN:
            has_abdn = True
            abdn_final_prob = abdn_probs[valid_abdn_pixels].max().item() * 100

        # Extract Generator (Class 1)
        gen_mask = np.where(mask_np == 1, 1, 0)

        # closing and combine the pointes nearby to get better connected components
        gen_mask = ndimage.binary_closing(gen_mask, iterations=15).astype(np.uint8)
        gen_mask = ndimage.binary_fill_holes(gen_mask).astype(np.uint8)

        labeled = skimage.measure.label(gen_mask)
        props = skimage.measure.regionprops(labeled)
        
        if len(props) == 0:
            print("❌ ไม่พบตัวเครื่อง (Generator) ในรูปภาพ")
            return

        # 3. CROPPING
        print("3. กำลังตัดและเตรียมรูปภาพ (Cropping)...")
        main_obj = props[np.argmax([p.area for p in props])]
        
        # ขยายพิกัดกลับไปที่ขนาดภาพต้นฉบับ หรือ Resize ภาพก่อนตัด
        img_512 = raw_img.resize((512, 512))
        img_512_np = np.array(img_512)
        minr, minc, maxr, maxc = fix_bbox(main_obj.bbox, img_512_np.shape)
        
        crop_arr = img_512_np[minr:maxr, minc:maxc]
        crop_img = PILImage.create(crop_arr).resize((256, 256))
        crop_img.save(temp_crop)

        # 4. CLASSIFICATION
        print("4. กำลังจำแนกผู้ผลิตและกลุ่มรุ่น (Classifying)...")
        manuf_res, _, manuf_probs = learn_manuf.predict(temp_crop)
        model_res, _, model_probs = learn_model.predict(temp_crop)

        # 5. DISPLAY RESULTS
        manuf_conf = manuf_probs.max().item() * 100
        model_conf = model_probs.max().item() * 100
        
        abdn_status = f"DETECTED 🔴 ({abdn_final_prob:>6.2f}%)" if has_abdn else f"None (Px: {pixel_count})"

        print("\n" + "╔" + "═"*53 + "╗")
        print(f"║ ผลการวิเคราะห์: {pathlib.Path(img_input).name:<32} ║")
        print("╠" + "═"*53 + "╣")
        print(f"║ ผู้ผลิต (Manufacturer) : {str(manuf_res):<20} ({manuf_conf:>6.2f}%) ║")
        print(f"║ กลุ่มรุ่น (Model Group) : {str(model_res):<20} ({model_conf:>6.2f}%) ║")
        print(f"║ Abandoned Lead         : {abdn_status:<32} ║")
        print("╚" + "═"*53 + "╝")

        #if os.path.exists(temp_crop): os.remove(temp_crop)

    except Exception as e:
        print(f"\n❌ [CRITICAL ERROR]: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_pipeline()