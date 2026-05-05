# CIED Automated Segmentation & Classification Pipeline
# This script performs the following steps:
# 1. Loads pre-trained segmentation and classification models.
# 2. Segments the input image to detect the generator and abandoned lead. + CLAHE
# 3. Crops the image around the detected generator.
# 4. Classifies the manufacturer and model group of the generator.
# Parameters to adjust: file paths, thresholds, and image processing settings. There are in STEP 3. 

import pathlib
import platform
import torch
import numpy as np
import skimage.measure
from fastai.vision.all import *
import scipy.ndimage as ndimage
import cv2

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

def fix_bbox(bbox_org, img_shape, border, minsize=160):
    minr, minc, maxr, maxc = bbox_org
    dr, dc = int((maxr-minr)*border), int((maxc-minc)*border)
    minr, minc, maxr, maxc = minr-dr, minc-dc, maxr+dr, maxc+dc
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
    else:        
        print(f"No CLAHE needed: mean={mean:.2f}, std={std:.2f} (Good Contrast and Balanced Brightness)")

    return img_clahe

# ==========================================================
# STEP 3: MAIN PIPELINE
# ==========================================================
def run_pipeline():
    # --- CONFIGURATION ---
    file_seg_weights = 'C:/CIEDID_data/AbdnL/models/best_abdn.pth' 
    file_manuf       = 'C:/CIEDID_data/pkl/classification_manuf.pkl'
    file_model       = 'C:/CIEDID_data/pkl/classification_model.pkl'
    img_input        = 'C:/CIEDID_data/AbdnL/data/1094.png' # เปลี่ยนเป็นรูปที่อยากลอง
    temp_crop        = 'cied/Dataset/temp_crop_pth.jpg'

    # Detection Thresholds for Abandoned Lead
    PIXEL_MIN = 1000
    PROB_THRESHOLD = 0.05
    # 4 classes by training: 0=background, 1=generator, 2=lead, 3=abandoned_lead
    CLASS_NAMES = ["background", "generator", "lead", "abandoned_lead"] 

    # Image processing parameters
    IMG_Size = 512
    Crop_border = 0.1

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
            label_func=lambda x: x, codes=CLASS_NAMES, 
            item_tfms=Resize(IMG_Size, method='pad', pad_mode='zeros') # use padding to maintain aspect ratio
        )

        learn_seg = unet_learner(dls_dummy, resnet50, n_out=4)
        
        # Load state_dict
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        learn_seg.model.load_state_dict(torch.load(file_seg_weights, map_location=device))
        learn_seg.model.eval() # ตั้งค่าเป็น evaluation mode เพื่อความแม่นยำ
        
        learn_manuf = load_learner(file_manuf, cpu=True)
        learn_model = load_learner(file_model, cpu=True)

        # 2. SEGMENTATION & DETECTION
        print("CLAHE processing applied to enhance image contrast before segmentation.")
        raw_img = apply_clahe(img_input) # เพิ่ม CLAHE เพื่อปรับปรุงความคมชัดก่อนส่งเข้าโมเดล
        raw_img = PILImage.create(raw_img) # แปลงกลับเป็น PILImage หลังจาก CLAHE แล้ว
        
        print("2. กำลังวิเคราะห์ตำแหน่งอุปกรณ์ (Segmenting)...")
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

        # กรองด้วย Threshold 
        valid_abdn_pixels = (mask_np == abdn_class_idx) & (abdn_probs > PROB_THRESHOLD)
        pixel_count = valid_abdn_pixels.sum().item()
        
        has_abdn = False
        abdn_final_prob = 0.0
        # ตรวจสอบเกณฑ์ Pixel_min 
        if pixel_count >= PIXEL_MIN:
            has_abdn = True
            abdn_final_prob = abdn_probs[valid_abdn_pixels].max().item() * 100

        # Extract Generator (Class 1)
        gen_mask = np.where(mask_np == 1, 1, 0)

        # closing and combine the pointes nearby to get better connected components
        #gen_mask = ndimage.binary_closing(gen_mask, iterations=10).astype(np.uint8)
        #gen_mask = ndimage.binary_fill_holes(gen_mask).astype(np.uint8)

        # dilatation and fill holes to get better connected components
        # 1. สร้าง Structure สำหรับภาพ 2D (Connectivity 2 คือเชื่อมแนวทแยงด้วย)
        struct = ndimage.generate_binary_structure(2, 2) 

        # 2. ตรวจสอบให้แน่ใจว่า gen_mask ไม่มีมิติส่วนเกิน (เช่น 1, 512, 512)
        if gen_mask.ndim > 2:
            gen_mask = gen_mask.squeeze() # ยุบมิติที่เกินออกให้เหลือแค่ (H, W)

        # ทำ Dilation และ Erosion อย่างละ 3 รอบตามโค้ดล่าสุดของคุณ
        iterations = max(1, int((IMG_Size / 512) * 3))  # ปรับจำนวนรอบตามขนาดภาพ
        gen_mask_processed = ndimage.binary_dilation(gen_mask, structure=struct, iterations=iterations).astype(np.uint8)
        gen_mask_processed = ndimage.binary_erosion(gen_mask_processed, structure=struct, iterations=iterations).astype(np.uint8)
        gen_mask_processed = ndimage.binary_fill_holes(gen_mask_processed).astype(np.uint8)
        # Convex hull — ทำหลัง fill holes
        # เหมาะกับ generator เพราะเป็น convex object จริงๆ
        # ช่วยให้ bbox สมมาตรและ crop ได้สวยกว่า
        from skimage.morphology import convex_hull_image
        if gen_mask_processed.sum() > 0:   # มี object อยู่ถึงทำ
            gen_mask_processed = convex_hull_image(gen_mask_processed).astype(np.uint8)

        labeled = skimage.measure.label(gen_mask)
        props = skimage.measure.regionprops(labeled)
        
        if len(props) == 0:
            print("❌ ไม่พบตัวเครื่อง (Generator) ในรูปภาพ")
            return

        # 3. CROPPING
        print("3. กำลังตัดและเตรียมรูปภาพ (Cropping)...")

        # กรองเอาเฉพาะชิ้นส่วนที่มีขนาดใหญ่กว่า 1,000 พิกเซล
        MIN_GEN_AREA = (IMG_Size * IMG_Size) * 0.0038 # ประมาณ 1,000 พิกเซลที่ขนาด 512
        large_props = [p for p in props if p.area > MIN_GEN_AREA]

        if len(large_props) == 0:
            print(f"❌ ไม่พบวัตถุที่มีขนาดใหญ่พอจะเป็น Generator (Area < {MIN_GEN_AREA})")
            # อาจจะให้ return หรือลองใช้ชิ้นที่ใหญ่ที่สุดที่มีดูถ้าต้องการ
            return 

        # เลือกชิ้นที่ใหญ่ที่สุดจากบรรดาชิ้นที่ผ่านเกณฑ์
        main_obj = large_props[np.argmax([p.area for p in large_props])]
        
        print(f"✅ พบ Generator พื้นที่: {main_obj.area} พิกเซล")
        #main_obj = props[np.argmax([p.area for p in props])]
        
        # ขยายพิกัดกลับไปที่ขนาดภาพต้นฉบับ หรือ Resize ภาพก่อนตัด
        #img_512 = raw_img.resize((512, 512))
        #img_512_np = np.array(img_512)
        #minr, minc, maxr, maxc = fix_bbox(main_obj.bbox, img_512_np.shape)
        #crop_arr = img_512_np[minr:maxr, minc:maxc]
        
        # 1. หาอัตราส่วนการย่อ (Scaling Factor)
        orig_w, orig_h = raw_img.size
        scale = max(orig_w, orig_h) / IMG_Size

        # 2. ปรับพิกัดจาก Mask (512) กลับไปเป็นขนาดต้นฉบับ
        # (สมมติว่าโมเดลทำ Padding ไว้ตรงกลาง)
        pad_y = (IMG_Size - (orig_h / scale)) / 2 if orig_h < orig_w else 0
        pad_x = (IMG_Size - (orig_w / scale)) / 2 if orig_w < orig_h else 0

        minr, minc, maxr, maxc = main_obj.bbox
        real_minr = int((minr - pad_y) * scale)
        real_maxr = int((maxr - pad_y) * scale)
        real_minc = int((minc - pad_x) * scale)
        real_maxc = int((maxc - pad_x) * scale)

        # 3. ส่งเข้า fix_bbox โดยใช้ขนาดภาพต้นฉบับ
        raw_np = np.array(raw_img)
        final_minr, final_minc, final_maxr, final_maxc = fix_bbox(
            (real_minr, real_minc, real_maxr, real_maxc), raw_np.shape, Crop_border
        )

        # 4. ตัดจากภาพต้นฉบับเพื่อให้ได้ความละเอียดสูงสุดก่อนส่งไป Classify
        crop_arr = raw_np[final_minr:final_maxr, final_minc:final_maxc]

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