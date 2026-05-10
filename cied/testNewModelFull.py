# CIED Automated Segmentation & Classification Pipeline
# FINAL VERSION: Precision Coordinate Mapping (Original Image Crop)

import pathlib
import platform
import torch
import torch.nn.functional as F
import numpy as np
import skimage.measure
from fastai.vision.all import *
import scipy.ndimage as ndimage
import cv2
import os

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

# ==========================================================
# STEP 3: MAIN PIPELINE
# ==========================================================
def run_pipeline():
    # --- CONFIGURATION ---
    file_seg_weights = 'C:/CIEDID_data/AbdnL/models/best_gen.pth' 
    file_manuf       = 'C:/CIEDID_data/pkl/classification_manuf.pkl'
    file_model       = 'C:/CIEDID_data/pkl/classification_model.pkl'
    img_input        = 'C:/CIEDID_data/AbdnL/data/101.png' 
    temp_crop        = 'cied/Dataset/temp_crop_pth.jpg'

    PIXEL_MIN = 1500
    PROB_THRESHOLD = 0.4
    CLASS_NAMES = ["background", "generator", "lead", "abandoned_lead"] 
    IMG_Size = 512
    Crop_border = 0.15

    try:
        apply_patches()
        print("\n" + "="*55)
        print(" CIED AUTOMATED SEGMENTATION & CLASSIFICATION SYSTEM")
        print("="*55)
        
        # 1. LOAD MODELS
        print("1. กำลังโหลดโมเดล...")
        dls_dummy = SegmentationDataLoaders.from_label_func(
            pathlib.Path("."), bs=1, fnames=[img_input], 
            label_func=lambda x: x, codes=CLASS_NAMES, 
            item_tfms=Resize(IMG_Size, method='pad', pad_mode='zeros') 
        )
        learn_seg = unet_learner(dls_dummy, resnet50, n_out=4)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        learn_seg.model.load_state_dict(torch.load(file_seg_weights, map_location=device))
        learn_seg.model.to(device).eval()
        
        learn_manuf = load_learner(file_manuf, cpu=True)
        learn_model = load_learner(file_model, cpu=True)

        # 2. SEGMENTATION
        print("2. กำลังวิเคราะห์ตำแหน่ง (Segmenting)...")
        raw_img = Image.open(img_input).convert('RGB')
        orig_w, orig_h = raw_img.size
        
        # เตรียมภาพเข้าโมเดล
        timg_pipe = Pipeline([PILImage.create, Resize(IMG_Size, method='pad', pad_mode='zeros'), ToTensor(), IntToFloatTensor()])
        timg_512 = timg_pipe(raw_img).to(device)

        # 🔥 เพิ่ม 3 บรรทัดนี้ เพื่อทำ Normalize ให้ตรงกับตอน Train
        mean_tensor = torch.tensor([0.502668, 0.502668, 0.502668], device=device).view(3, 1, 1)
        std_tensor = torch.tensor([0.240966, 0.240966, 0.240966], device=device).view(3, 1, 1)
        timg_512 = (timg_512 - mean_tensor) / std_tensor
        
        with torch.no_grad():
            output = learn_seg.model(timg_512.unsqueeze(0))
            mask_np = output.argmax(dim=1)[0].cpu().numpy()
            probs_tensor = F.softmax(output, dim=1)[0].cpu()

        # --- Generator Mask Processing ---
        gen_mask = np.where(mask_np == 1, 1, 0).astype(np.uint8)
        # Dilate เล็กน้อยเพื่อเชื่อมช่องว่าง
        gen_mask = ndimage.binary_dilation(gen_mask, iterations=2).astype(np.uint8)
        gen_mask = ndimage.binary_fill_holes(gen_mask).astype(np.uint8)
        
        props = skimage.measure.regionprops(skimage.measure.label(gen_mask))
        if len(props) == 0:
            print("❌ ไม่พบตัวเครื่อง")
            return
        main_obj = sorted(props, key=lambda x: x.area, reverse=True)[0]

        # 3. PRECISION CROPPING (Map back to Original)
        print("3. กำลังตัดรูปจากภาพต้นฉบับ (Precision Crop)...")
        
        # คำนวณอัตราส่วนการ Scale และ Padding ที่ FastAI ใช้
        ratio = IMG_Size / max(orig_h, orig_w)
        new_h, new_w = int(orig_h * ratio), int(orig_w * ratio)
        pad_y = (IMG_Size - new_h) // 2
        pad_x = (IMG_Size - new_w) // 2

        # ดึงพิกัดจาก Mask (512)
        m_minr, m_minc, m_maxr, m_maxc = main_obj.bbox
        
        # แปลงกลับเป็นพิกัดบนรูปต้นฉบับ (หักลบ Pad และหารด้วย Ratio)
        orig_minr = max(0, int((m_minr - pad_y) / ratio))
        orig_maxr = min(orig_h, int((m_maxr - pad_y) / ratio))
        orig_minc = max(0, int((m_minc - pad_x) / ratio))
        orig_maxc = min(orig_w, int((m_maxc - pad_x) / ratio))

        # ส่งเข้า fix_bbox เพื่อเผื่อ Border 15% บนรูปต้นฉบับจริง
        f_minr, f_minc, f_maxr, f_maxc = fix_bbox(
            (orig_minr, orig_minc, orig_maxr, orig_maxc), (orig_h, orig_w), Crop_border
        )

        # ตัดจากภาพ Original (raw_img) เพื่อความชัดสูงสุด
        raw_np = np.array(raw_img)
        crop_arr = raw_np[f_minr:f_maxr, f_minc:f_maxc]
        
        # บันทึกเป็น 256x256 สำหรับ Classification
        crop_img = PILImage.create(crop_arr).resize((256, 256))
        os.makedirs(os.path.dirname(temp_crop), exist_ok=True)
        crop_img.save(temp_crop)

        # 4. CLASSIFICATION & RESULT
        manuf_res, _, manuf_probs = learn_manuf.predict(temp_crop)
        model_res, _, model_probs = learn_model.predict(temp_crop)

        # Abandoned Lead Status
        abdn_probs = probs_tensor[3].numpy()
        has_abdn = ((mask_np == 3) & (abdn_probs > PROB_THRESHOLD)).sum() >= PIXEL_MIN
        
        print("\n" + "╔" + "═"*53 + "╗")
        print(f"║ ผลการวิเคราะห์: {pathlib.Path(img_input).name:<32} ║")
        print("╠" + "═"*53 + "╣")
        print(f"║ Manuf  : {str(manuf_res):<20} ({manuf_probs.max()*100:>6.2f}%) ║")
        print(f"║ Model  : {str(model_res):<20} ({model_probs.max()*100:>6.2f}%) ║")
        print(f"║ Abdn L : {'DETECTED 🔴' if has_abdn else 'None':<32} ║")
        print("╚" + "═"*53 + "╝")

    except Exception as e:
        print(f"\n❌ [ERROR]: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_pipeline()