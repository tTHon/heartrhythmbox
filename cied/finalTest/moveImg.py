import os
import shutil
import pandas as pd

# 1. กำหนดพาธของไฟล์ CSV และโฟลเดอร์
csv_file = 'C:/CIEDID_data/main_workbook.csv'
source_folder = 'C:/CIEDID_data/Images_processed'       # โฟลเดอร์ต้นทางที่เก็บรูป png
destination_folder = 'C:/CIEDID_data/Images_finalTest'  # โฟลเดอร์ปลายทางที่ต้องการ copy ไปวาง

# สร้างโฟลเดอร์ปลายทางหากยังไม่มี
os.makedirs(destination_folder, exist_ok=True)

# 2. อ่านไฟล์ CSV
df = pd.read_csv(csv_file, low_memory=False)

# 3. กรองข้อมูลเฉพาะแถวที่ FinalTest == 1
filtered_df = df[df['FinalTest'] == 1]

# 4. วนลูปเพื่อคัดลอกไฟล์
copied_count = 0
not_found_count = 0

for idx, row in filtered_df.iterrows():
    # สมมุติว่าชื่อไฟล์มาจากคอลัมน์ 'ID' หรือคอลัมน์ชื่อไฟล์ของคุณ (เช่น f"{row['ID']}.png")
    # ปรับชื่อคอลัมน์หรือรูปแบบชื่อไฟล์ตามความเหมาะสม
    filename = f"{row['ID']}.png"  
    
    src_path = os.path.join(source_folder, filename)
    dst_path = os.path.join(destination_folder, filename)
    
    # ตรวจสอบว่ามีไฟล์อยู่ในโฟลเดอร์ต้นทางหรือไม่
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        copied_count += 1
    else:
        # กรณีค้นหาไฟล์ png ที่อาจต่อท้ายด้วยนามสกุลเดิมแล้ว
        alt_src_path = os.path.join(source_folder, str(row['ID']))
        if os.path.exists(alt_src_path):
            shutil.copy2(alt_src_path, dst_path)
            copied_count += 1
        else:
            not_found_count += 1

print(f"คัดลอกสำเร็จ: {copied_count} ไฟล์")
print(f"ไม่พบไฟล์: {not_found_count} ไฟล์")