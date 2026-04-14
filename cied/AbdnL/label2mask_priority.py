import json
import numpy as np
import cv2
import PIL.Image
import pathlib
from pathlib import Path

# ==============================
# ✏️ CONFIG — ตั้งค่าที่นี่
# ==============================
INPUT_DIR = Path(r"C:/CIEDID_data/AbdnL/data")      # ที่เก็บไฟล์ .json
OUTPUT_DIR = Path(r"C:/CIEDID_data/AbdnL/mask")    # ที่เก็บไฟล์ mask ผลลัพธ์

# Priority: วัตถุที่อยู่ท้ายลิสต์จะทับวัตถุที่อยู่ก่อนหน้า (ต่ำ -> สูง)
# ในที่นี้: abandoned_lead จะทับ lead และ lead จะทับ generator
CLASS_PRIORITY = ["generator", "lead", "abandoned_lead"]
CLASS_MAP = {
    "generator": 1,
    "lead": 2,
    "abandoned_lead": 3
}
# ==============================

def process_single_json(json_path: Path, output_path: Path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    height = data['imageHeight']
    width = data['imageWidth']
    
    # สร้างพื้นหลังสีดำ (0)
    final_mask = np.zeros((height, width), dtype=np.uint8)
    
    # คำนวณความหนาเส้น (0.8% ของความกว้างภาพ)
    dynamic_thickness = max(1, int(width * 0.008)) 

    # วาดแยกทีละ Layer ตามลำดับ Priority เพื่อจัดการการทับซ้อน
    for label_name in CLASS_PRIORITY:
        class_id = CLASS_MAP[label_name]
        
        # สร้าง Layer แยกสำหรับ Class นี้โดยเฉพาะ
        layer_mask = np.zeros((height, width), dtype=np.uint8)
        
        # กรองเอาเฉพาะ shape ที่ตรงกับ label ปัจจุบัน
        target_shapes = [s for s in data['shapes'] if s['label'].lower().strip() == label_name]
        
        for shape in target_shapes:
            points = np.array(shape['points'], dtype=np.int32)
            
            if shape['shape_type'] in ['linestrip', 'line']:
                cv2.polylines(layer_mask, [points], False, class_id, thickness=dynamic_thickness)
            else:
                cv2.fillPoly(layer_mask, [points], class_id)
        
        # นำ Layer นี้ไปแปะทับลงใน final_mask 
        # เฉพาะบริเวณที่ layer_mask มีค่า (ไม่ใช่ 0) จะไปทับของเดิม
        final_mask = np.where(layer_mask > 0, layer_mask, final_mask)

    # บันทึกไฟล์
    PIL.Image.fromarray(final_mask).save(output_path)
    return np.unique(final_mask)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_files = sorted(INPUT_DIR.glob("*.json"))

    if not json_files:
        print(f"❌ ไม่พบไฟล์ .json ใน {INPUT_DIR}")
        return

    print(f"🚀 เริ่มการแปลงไฟล์ {len(json_files)} ไฟล์...")
    print(f"{'Filename':40s} | {'Unique IDs'}")
    print("-" * 60)

    for json_file in json_files:
        mask_name = f"{json_file.stem}_mask.png"
        save_path = OUTPUT_DIR / mask_name
        
        try:
            unique_values = process_single_json(json_file, save_path)
            print(f"{mask_name:40s} | {list(unique_values)}")
        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {e}")

    print("-" * 60)
    print(f"✅ สำเร็จ! บันทึก Mask ไปที่: {OUTPUT_DIR}")
    print(f"⚠️  อย่าลืมเช็คว่า ID 3 (abandoned_lead) ปรากฏในลิสต์ Unique IDs หรือไม่")

if __name__ == "__main__":
    main()