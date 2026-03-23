import torch
print(f"Is CUDA available in this script?: {torch.cuda.is_available()}")
import easyocr
import cv2
import numpy as np
import os
from glob import glob

def batch_anonymize(input_folder, output_folder):
    # 1. สร้างโฟลเดอร์ Output ถ้ายังไม่มี
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 2. โหลด OCR Reader (โหลดครั้งเดียวใช้วนลูปเพื่อประหยัด Memory)
    reader = easyocr.Reader(['en'], gpu=True)  # ใช้ GPU ถ้ามี (ปรับเป็น False ถ้าไม่มี)
    
    # 3. ค้นหาไฟล์ภาพทั้งหมด (jpg, png, jpeg)
    extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in extensions:
        image_files.extend(glob(os.path.join(input_folder, ext)))

    print(f"พบไฟล์ทั้งหมด: {len(image_files)} ไฟล์")

    for img_path in image_files:
        filename = os.path.basename(img_path)
        print(f"กำลังประมวลผล: {filename}...")

        # อ่านภาพ
        img = cv2.imread(img_path)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # 4. OCR Detection
        results = reader.readtext(img_path)

        for (bbox, text, prob) in results:
            # กรองเฉพาะข้อความที่มั่นใจเกิน 40% (ปรับค่าได้)
            if prob > 0.4:
                (tl, tr, br, bl) = bbox
                top_left = (int(tl[0]), int(tl[1]))
                bottom_right = (int(br[0]), int(br[1]))
                
                # วาด Mask ทับตำแหน่งข้อความ
                cv2.rectangle(mask, top_left, bottom_right, 255, -1)

        # 5. ขยาย Mask เล็กน้อยเพื่อให้ลบขอบตัวอักษรได้เนียนขึ้น
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # 6. Inpaint (ลบและเติมพื้นหลัง)
        result = cv2.inpaint(img, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

        # 7. บันทึกไฟล์
        save_path = os.path.join(output_folder, f"anon_{filename}")
        cv2.imwrite(save_path, result)

    print("\n--- เสร็จสิ้นการทำ Batch Processing ---")

# --- วิธีใช้งาน ---
# ใส่ชื่อโฟลเดอร์ที่เก็บภาพ X-ray และโฟลเดอร์ที่จะให้บันทึกผลลัพธ์
batch_anonymize('cied/anonymizer/in', 'cied/anonymizer/out')