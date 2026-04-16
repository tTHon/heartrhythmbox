# Grid Search for Threshold and Pixel Count Parameters
# last run: Sweetspot = Threshold 0.7, Pixel_min 80 (F1 0.81, Recall 1.00, pPrecision 0.69)
import torch
import pathlib
import numpy as np
import pandas as pd
from fastai.vision.all import *
from sklearn.metrics import classification_report
import os

# 1. ตั้งค่าเส้นทางและคลาส (ตรวจสอบให้ตรงกับเครื่องของคุณ)
path_img_folder = "C:/CIEDID_data/AbdnL/data"   
path_mask_folder = "C:/CIEDID_data/AbdnL/mask"  
path_weights = "C:/CIEDID_data/AbdnL/models/best_seg.pth"
class_names = ["Background", "Generator", "Lead", "Abdn_Lead"]
threshold = 0.7
pixel_min = 50 

# 2. โหลดโมเดล
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ดึงไฟล์รูปแรกมาเพื่อสร้าง dls จำลอง
first_img = get_image_files(path_img_folder)[0]
dls = SegmentationDataLoaders.from_label_func(
    pathlib.Path("."), bs=1, fnames=[first_img], 
    label_func=lambda x: x, codes=class_names, item_tfms=Resize(512)
)
learn = unet_learner(dls, resnet50, n_out=4)
learn.model.load_state_dict(torch.load(path_weights, map_location=device))
learn.model.eval()

# 3. เตรียมตัวแปรเก็บผลลัพธ์
results_case = [] # สำหรับ Yes/No
all_y_true = []   # สำหรับพิกเซล (เฉลย)
all_y_pred = []   # สำหรับพิกเซล (ทำนาย)

print("🚀 Starting Inference and calculating metrics...")

# 4. วนลูปประมวลผลทุกรูป
for img_file in get_image_files(path_img_folder):
    # จัดการชื่อไฟล์ Mask (เติม _mask)
    mask_name = f"{img_file.stem}_mask{img_file.suffix}"
    mask_path = pathlib.Path(path_mask_folder) / mask_name
    
    if not mask_path.exists():
        continue
    
    # --- ส่วนที่ 1: Ground Truth ---
    target_mask = PILMask.create(mask_path).reshape(512, 512)
    y_true = np.array(target_mask)
    gt_exists = (y_true == 3).any() # มีคลาส 3 ในรูปไหม
    
    # --- ส่วนที่ 2: Prediction ---
    with torch.no_grad():
        mask_pred, _, probs = learn.predict(img_file)
    
    final_mask = mask_pred.numpy().copy()
    # กรอง Confidence เฉพาะคลาส 3
    abdn_probs = probs[3].numpy()
    final_mask[(final_mask == 3) & (abdn_probs < threshold)] = 0
    
    # เช็คว่าทายว่ามีไหม (Yes/No)
    pred_exists = (np.sum(final_mask == 3) > pixel_min)
    
    # --- ส่วนที่ 3: เก็บข้อมูล ---
    results_case.append({'Actual': gt_exists, 'Predicted': pred_exists})
    all_y_true.extend(y_true.flatten())
    all_y_pred.extend(final_mask.flatten())

# 5. แสดงผลตารางที่ 1: Case-level (Yes/No)
print("\n" + "="*50)
print("ตารางที่ 1: ประสิทธิภาพการจำแนกรายคน (Yes/No)")
print("="*50)
df_case = pd.DataFrame(results_case)
print(classification_report(df_case['Actual'], df_case['Predicted'], 
                            target_names=['No Abdn', 'Has Abdn'], zero_division=0))

# 6. แสดงผลตารางที่ 2: Pixel-level (แยกตามคลาส)
print("\n" + "="*50)
print("ตารางที่ 2: ประสิทธิภาพระดับพิกเซลแยกตามคลาส")
print("="*50)
report_pixel = classification_report(
    all_y_true, all_y_pred, 
    target_names=class_names, 
    labels=[0, 1, 2, 3],
    output_dict=True,
    zero_division=0
)
df_pixel = pd.DataFrame(report_pixel).transpose()
final_pixel_table = df_pixel.loc[class_names, ['precision', 'recall', 'f1-score']]
final_pixel_table.columns = ['Precision', 'Recall', 'F1-score']
print(final_pixel_table)

import pandas as pd

# 1. กำหนดค่าที่ต้องการทดสอบ (ปรับเปลี่ยนได้ตามใจชอบ)
thresholds = [0.5, 0.6, 0.7, 0.8]
pixel_mins = [30, 50, 80, 100]

grid_results = []

print("🔍 Starting Grid Search for best parameters...")

# โหลดข้อมูลภาพและทำนายไว้ก่อนรอบเดียว (เพื่อความเร็ว)
all_preds = []
for img_file in get_image_files(path_img_folder):
    mask_name = f"{img_file.stem}_mask{img_file.suffix}"
    mask_path = pathlib.Path(path_mask_folder) / mask_name
    if not mask_path.exists(): continue
    
    gt_exists = (np.array(PILMask.create(mask_path)) == 3).any()
    
    # ดึงค่า probability ของคลาส 3 (Abdn Lead) ออกมา
    with torch.no_grad():
        _, _, probs = learn.predict(img_file)
    abdn_probs = probs[3].numpy()
    
    all_preds.append({'gt': gt_exists, 'probs': abdn_probs})

# 2. รัน Grid Search
for t in thresholds:
    for p in pixel_mins:
        y_true_case = []
        y_pred_case = []
        
        for item in all_preds:
            y_true_case.append(item['gt'])
            # ตัดสินใจ Yes/No ตามเกณฑ์ t และ p
            pred = (np.sum(item['probs'] >= t) > p)
            y_pred_case.append(pred)
        
        # คำนวณ Precision/Recall เฉพาะคลาส Has Abdn (pos_label=True)
        from sklearn.metrics import precision_score, recall_score, f1_score
        prec = precision_score(y_true_case, y_pred_case, zero_division=0)
        rec = recall_score(y_true_case, y_pred_case, zero_division=0)
        f1 = f1_score(y_true_case, y_pred_case, zero_division=0)
        
        grid_results.append({
            'Threshold': t,
            'Pixel_Min': p,
            'Precision': round(prec, 2),
            'Recall': round(rec, 2),
            'F1-score': round(f1, 2)
        })

# 3. แสดงผลลัพธ์
df_grid = pd.DataFrame(grid_results)
print("\n" + "="*60)
print("📊 สรุปผลการทดสอบ Parameter ต่างๆ (Class: Has Abdn)")
print("="*60)
print(df_grid.sort_values(by='F1-score', ascending=False).to_string(index=False))

# เซฟลง CSV ไว้ใส่ในรายงาน
# final_table.to_csv("model_performance_report.csv")