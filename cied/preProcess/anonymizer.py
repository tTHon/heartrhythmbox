# --- Batch Anonymizer Script ---
# All images undergo CLAHE first,
# then anonymization.
# The preprocessing ends here.
# Check the folder "2_ANON" for the final anonymized images.
import torch
print(f"Is CUDA available in this script?: {torch.cuda.is_available()}")
import easyocr
import cv2
import numpy as np
import os
from glob import glob

def batch_anonymize(input_folder, output_folder):
    # 1. check and create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 2. Initialize EasyOCR reader
    reader = easyocr.Reader(['en'], gpu=True)  # ใช้ GPU ถ้ามี (ปรับเป็น False ถ้าไม่มี)
    
    # 3. Find all image files (jpg, png, jpeg)
    extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in extensions:
        image_files.extend(glob(os.path.join(input_folder, ext)))

    print(f"Found total files: {len(image_files)} files")

    for img_path in image_files:
        filename = os.path.basename(img_path)
        print(f"Processing: {filename}...")

        # Read image
        img = cv2.imread(img_path)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # 4. OCR Detection
        results = reader.readtext(img_path)

        for (bbox, text, prob) in results:
            # Filter only texts with confidence over 40% (adjustable)
            if prob > 0.4:
                (tl, tr, br, bl) = bbox
                top_left = (int(tl[0]), int(tl[1]))
                bottom_right = (int(br[0]), int(br[1]))
                
                # Draw Mask over text positions
                cv2.rectangle(mask, top_left, bottom_right, 255, -1)

        # 5. Dilate Mask slightly to ensure all text edges are covered
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # 6. Inpaint the image using the mask
        result = cv2.inpaint(img, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

        # 7. Save file
        save_path = os.path.join(output_folder, f"{filename}")
        cv2.imwrite(save_path, result)

    print("\n--- Finished Batch Processing ---")

# --- Usage ---
# Enter the names of the folders containing the X-ray images and the output folder
batch_anonymize('c:/CIEDID_data/Preprocessing/1_CLAHE', 'c:/CIEDID_data/Preprocessing/2_ANON')