import cv2
import PIL.Image
import numpy as np
from PIL import Image

#mask = np.array(Image.open('cied\AbdnL\mask\Adbn003_mask.png'))
#print(f"ค่าพิกเซลที่พบใน Mask: {np.unique(mask)}")



# โหลดรูปต้นฉบับและ Mask
img = cv2.imread('cied/AbdnL/data/Adbn002.jpg')
mask = cv2.imread('cied/AbdnL/mask/Adbn002_mask.png', 0)

# สร้างสีสันให้ Mask (เช่น ให้ Lead เป็นสีแดง)
color_mask = np.zeros_like(img)
color_mask[mask == 1] = [0, 0, 255]   # Generator = สีแดง
color_mask[mask == 4] = [0, 255, 0]   # Abandoned Lead = สีเขียว

# เอามาซ้อนทับกัน (Alpha Blending) ให้เห็นทั้งรูปคนและเส้น Label
overlay = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)

cv2.imwrite('cied/AbdnL/check_alignment.png', overlay)
print("สร้างไฟล์ check_alignment.png เสร็จแล้ว! ลองดูว่าเส้นตรงไหม")