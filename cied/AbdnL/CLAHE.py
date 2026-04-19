import cv2
import os
from pathlib import Path

# กำหนด Path
data_path = Path("C:/CIEDID_data/AbdnL/data/CLAHE")

# สร้างวัตถุ CLAHE (แนะนำ clipLimit=2.0 เพื่อความสมดุล)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# นามสกุลไฟล์ที่ต้องการทำ
valid_extensions = ('.jpg', '.png', '.jpeg', '.JPG', '.PNG')

print(f"Starting CLAHE processing in {data_path}...")

count = 0
for file_path in data_path.iterdir():
    if file_path.suffix in valid_extensions:
        # 1. อ่านภาพ (Grayscale)
        img = cv2.imread(str(file_path), 0)
        
        if img is not None:
            # 2. ประมวลผล CLAHE
            final_img = clahe.apply(img)
            
            # 3. เซฟทับไฟล์เดิม
            cv2.imwrite(str(file_path), final_img)
            count += 1
            if count % 10 == 0:
                print(f"Processed {count} images...")
        else:
            print(f"Could not read: {file_path}")

print(f"Finished! Processed {count} images and overwritten them.")