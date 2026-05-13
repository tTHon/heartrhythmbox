import json
import numpy as np
import cv2
import PIL.Image
from pathlib import Path

def labelme_json_to_mask(json_path, output_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    height = data['imageHeight']
    width = data['imageWidth']
    mask = np.zeros((height, width), dtype=np.uint8)

    #remove monitor
    class_map = {
        "generator": 1,
        "lead": 2,
        "abandoned_lead": 3
    }

    for shape in data['shapes']:
        label = shape['label'].lower().strip() # ลบช่องว่างเผื่อพิมพ์เกิน
        class_id = class_map.get(label, 0)
        # เพิ่มการ Print เพื่อ Debug
        if class_id == 0:
            print(f"Warning: Unknown label '{label}' in {json_path.name}")
            continue
        
        points = np.array(shape['points'], dtype=np.int32)

        if shape['shape_type'] in ['linestrip', 'line']:
            # สำคัญ: สาย Lead ต้องหนาพอให้ ResNet เห็น (แนะนำ 5-8 pixel)
            cv2.polylines(mask, [points], False, class_id, thickness=15)
        else:
            cv2.fillPoly(mask, [points], class_id)

    PIL.Image.fromarray(mask).save(output_path)

# --- ส่วนที่ต้องแก้ไขให้ตรงกับเครื่องคุณ ---
input_folder = Path('C:\\CIEDID_data\\AbdnL\\data')    # ที่เก็บไฟล์ .json
output_folder = Path('C:\\CIEDID_data\\AbdnL\\mask')    # ที่จะเก็บไฟล์ _mask.png (ให้ตรงกับสคริปต์ Finetune)
output_folder.mkdir(parents=True, exist_ok=True)

# วนลูปจัดการทุกไฟล์ JSON
for json_file in input_folder.glob('*.json'):
    # ตั้งชื่อไฟล์ output ให้ตรงกับที่สคริปต์ finetune_seg_v2.py ต้องการ
    # เช่น img1.json -> img1_mask.png
    mask_name = f"{json_file.stem}_mask.png"
    save_path = output_folder / mask_name
    
    labelme_json_to_mask(json_file, save_path)
    print(f"Converted: {json_file.name} -> {mask_name}")