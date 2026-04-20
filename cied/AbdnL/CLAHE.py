import cv2
import os
from pathlib import Path

# กำหนด Path
source_path = Path("C:/CIEDID_data/images") # main image files

# copy files from main source to here to process with CLAHE
# all previous files in this folder will be deleted before copying new ones
data_path = Path("C:/CIEDID_data/AbdnL/data/CLAHE") # processed image files


#clear folder
for file in data_path.iterdir():
    if file.is_file():
        os.remove(file)
print(f"Cleared {data_path} of existing files.")

#copy files from source to data
count = 0
for file in source_path.iterdir():
    if file.is_file():
        destination = data_path / file.name
        os.replace(file, destination)
        count += 1
print(f"Copied {count} files from {source_path} to {data_path}.")

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

print(f"Finished! Processed {count} images.")