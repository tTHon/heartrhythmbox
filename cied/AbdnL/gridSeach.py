# grid_search_abdn.py
# สคริปต์สำหรับหาค่า PIXEL_MIN และ PROB_THRESHOLD ที่ดีที่สุดสำหรับการตรวจจับ Abandoned Lead

import pathlib
import platform
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from fastai.vision.all import *
import os
from tqdm import tqdm

# ==========================================================
# 1. COMPATIBILITY PATCHES
# ==========================================================
def apply_patches():
    if not hasattr(np, 'int'): np.int = int
    if platform.system() == 'Windows':
        pathlib.PosixPath = pathlib.WindowsPath
    else:
        pathlib.WindowsPath = pathlib.PosixPath

# ==========================================================
# 2. CONFIGURATION & GRID SEARCH SPACE
# ==========================================================
# ตั้งค่าพาธของข้อมูล (ควรใช้รูปในชุด Validation Set เพื่อไม่ให้ Overfit)
FILE_WEIGHTS  = 'C:/CIEDID_data/AbdnL/models/best_abdn.pth' # แนะนำให้ใช้ best_abdn
DIR_IMAGES    = 'C:/CIEDID_data/AbdnL/data'
DIR_MASKS     = 'C:/CIEDID_data/AbdnL/mask'
CSV_OUTPUT    = 'C:/CIEDID_data/AbdnL/grid_search_results.csv'

IMG_Size = 512
CLASS_NAMES = ["background", "generator", "lead", "abandoned_lead"]
ABDN_CLASS_IDX = 3

# 🔥 กำหนดช่วงตัวเลขที่ต้องการค้นหา (Grid Search Space)
# คุณสามารถเพิ่ม/ลดตัวเลขในลิสต์นี้ได้ตามต้องการ
GRID_PROB_THRESHOLDS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50 ]
GRID_PIXEL_MINS      = [50, 100, 200, 300, 500, 800, 1000, 1100, 1200, 1300, 1400, 1500]

# ==========================================================
# 3. MAIN SCRIPT
# ==========================================================
def run_grid_search():
    apply_patches()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print(" 🔍 STARTING GRID SEARCH FOR ABANDONED LEAD DETECTION")
    print("="*60)

    # 1. เตรียมโมเดล (ใช้วิธี Raw PyTorch Inference แบบเดียวกับที่เวิร์ค)
    print("1. กำลังโหลดโมเดล...")
    # สร้าง Dummy DLS เพื่อปั้นโครงสร้าง U-Net
    img_files = get_image_files(DIR_IMAGES)
    if len(img_files) == 0:
        print(f"❌ ไม่พบรูปภาพใน {DIR_IMAGES}")
        return

    dls_dummy = SegmentationDataLoaders.from_label_func(
        pathlib.Path("."), bs=1, fnames=img_files[:2], 
        label_func=lambda x: x, codes=CLASS_NAMES, 
        item_tfms=Resize(IMG_Size, method='pad', pad_mode='zeros') 
    )
    learn_seg = unet_learner(dls_dummy, resnet50, n_out=4)
    learn_seg.model.load_state_dict(torch.load(FILE_WEIGHTS, map_location=device))
    learn_seg.model.to(device).eval()

    # เตรียม Pipeline และค่า Normalize ให้ตรงกับตอน Train
    timg_pipe = Pipeline([PILImage.create, Resize(IMG_Size, method='pad', pad_mode='zeros'), ToTensor(), IntToFloatTensor()])
    mean_tensor = torch.tensor([0.502668, 0.502668, 0.502668], device=device).view(3, 1, 1)
    std_tensor  = torch.tensor([0.240966, 0.240966, 0.240966], device=device).view(3, 1, 1)

    # 2. ทำนายผลล่วงหน้า (Pre-calculate Predictions)
    print(f"\n2. กำลังประมวลผลรูปภาพทั้งหมด {len(img_files)} รูป (ทำครั้งเดียว)...")
    
    image_results = [] # เก็บ [(gt_has_abdn, prob_map_np), ...]
    
    for img_path in tqdm(img_files):
        # โหลดภาพและทำ Inference
        raw_img = Image.open(img_path).convert('RGB')
        timg = timg_pipe(raw_img).to(device)
        timg = (timg - mean_tensor) / std_tensor # Normalize
        
        with torch.no_grad():
            output = learn_seg.model(timg.unsqueeze(0))
            probs_tensor = F.softmax(output, dim=1)[0].cpu()
            
        # ดึงเฉพาะ Probability Map ของ Abandoned Lead (Class 3)
        abdn_probs_np = probs_tensor[ABDN_CLASS_IDX].numpy()
        
        # โหลด Ground Truth (เฉลย) จากไฟล์ Mask
        # ลองหาไฟล์ _mask.png ก่อน ถ้าไม่มีค่อยใช้ .png
        mask_path = pathlib.Path(DIR_MASKS) / f"{img_path.stem}_mask.png"
        if not mask_path.exists():
            mask_path = pathlib.Path(DIR_MASKS) / f"{img_path.stem}.png"
            
        gt_has_abdn = False
        if mask_path.exists():
            mask_arr = np.array(PILImage.create(mask_path))
            gt_has_abdn = (ABDN_CLASS_IDX in np.unique(mask_arr))
            
        image_results.append((gt_has_abdn, abdn_probs_np))

    # 3. วนลูปคำนวณ Grid Search
    print("\n3. กำลังทดสอบจับคู่ Grid Search...")
    grid_results = []

    for prob_th in GRID_PROB_THRESHOLDS:
        for pix_min in GRID_PIXEL_MINS:
            
            TP = FP = TN = FN = 0
            
            for gt_has_abdn, abdn_probs_np in image_results:
                # คำนวณหาจำนวนพิกเซลที่ผ่านเกณฑ์ probability
                pixel_count = (abdn_probs_np > prob_th).sum()
                pred_has_abdn = (pixel_count >= pix_min)
                
                # ตรวจคำตอบ
                if pred_has_abdn and gt_has_abdn:
                    TP += 1
                elif pred_has_abdn and not gt_has_abdn:
                    FP += 1
                elif not pred_has_abdn and not gt_has_abdn:
                    TN += 1
                elif not pred_has_abdn and gt_has_abdn:
                    FN += 1

            # คำนวณสถิติ
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0 # ตรวจจับสิ่งที่ผิดปกติเจอไหม
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0 # รูปปกติ ทายว่าปกติไหม
            precision   = TP / (TP + FP) if (TP + FP) > 0 else 0.0 # ทายว่ามี แล้วมีจริงไหม
            
            # F1-Score: ค่าเฉลี่ยฮาร์โมนิกระหว่าง Precision กับ Sensitivity (ตัวชี้วัดความสมดุลที่ดีที่สุด)
            f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
            
            grid_results.append({
                'Prob_Threshold': prob_th,
                'Pixel_Min': pix_min,
                'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
                'Sensitivity': round(sensitivity, 4),
                'Specificity': round(specificity, 4),
                'Precision': round(precision, 4),
                'F1_Score': round(f1_score, 4)
            })

    # 4. สรุปผล
    df_results = pd.DataFrame(grid_results)
    df_results = df_results.sort_values(by=['F1_Score', 'Sensitivity'], ascending=[False, False])
    
    print("\n" + "="*60)
    print(" 🏆 TOP 5 BEST CONFIGURATIONS (Sorted by F1-Score)")
    print("="*60)
    print(df_results.head(5).to_string(index=False))
    
    # บันทึกลง CSV
    df_results.to_csv(CSV_OUTPUT, index=False)
    print(f"\n💾 บันทึกผลลัพธ์ทั้งหมดลงใน: {CSV_OUTPUT}")

if __name__ == "__main__":
    run_grid_search()