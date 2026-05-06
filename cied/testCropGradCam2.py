import torch
import pathlib
import numpy as np
import skimage.measure
import matplotlib.pyplot as plt
from fastai.vision.all import *
import scipy.ndimage as ndimage
import cv2

# ==========================================================
# 1. CONFIGURATION & PARAMETERS
# ==========================================================
path_weights = "C:/CIEDID_data/AbdnL/models/best_abdn.pth"
path_img = "C:/CIEDID_data/AbdnL/data/319.png" 

IMG_Size = 512       
Crop_border = 0.15    
class_names = ["background", "generator", "lead", "abandoned_lead"]

# ==========================================================
# 2. HELPER FUNCTIONS & HOOK
# ==========================================================
def fix_bbox(bbox_org, img_shape, border, minsize=200):
    minr, minc, maxr, maxc = bbox_org
    h, w = maxr - minr, maxc - minc
    dr, dc = int(h * border), int(w * border)
    minr, minc, maxr, maxc = minr - dr, minc - dc, maxr + dr, maxc + dc
    
    side = max(maxr - minr, maxc - minc, minsize)
    center_r, center_c = (minr + maxr) // 2, (minc + maxc) // 2
    minr, maxr = center_r - side // 2, center_r + side // 2
    minc, maxc = center_c - side // 2, center_c + side // 2
    H, W = img_shape[:2]
    return max(0, minr), max(0, minc), min(H, maxr), min(W, maxc)

class Hook():
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_func)
    def hook_func(self, m, i, o): self.stored = o.detach()
    def __enter__(self, *args): return self
    def __exit__(self, *args): self.hook.remove()

# ==========================================================
# 3. LOAD MODEL & PREDICT (Raw Model Access)
# ==========================================================
encoder = create_body(resnet50(pretrained=False), pretrained=False)
model = DynamicUnet(encoder, n_out=4, img_size=(IMG_Size, IMG_Size))
model.load_state_dict(torch.load(path_weights, map_location='cpu'))
model.eval()

# check code structure to find correct layer for Hook
#print("--- Searching for UnetBlocks in Model ---")
#for i, layer in enumerate(model):
    #print(f"Index {i}: {type(layer)}")
    # ถ้าเลเยอร์นั้นมีเลเยอร์ย่อยข้างใน (เช่น Decoder ส่วนใหญ่) ให้ส่องเข้าไปอีกชั้น
    #if isinstance(layer, torch.nn.Sequential):
        #for j, sub_layer in enumerate(layer):
        #    print(f"  --> Sub-Index {j}: {type(sub_layer)}")
#print("------------------------------------------")

raw_img = Image.open(path_img).convert('RGB')
# ใช้ Resize method='pad' เพื่อให้สอดคล้องกับการคำนวณ Scaling ในส่วนถัดไป
timg_pipe = Pipeline([PILImage.create, Resize(IMG_Size, method='pad', pad_mode='zeros'), ToTensor(), IntToFloatTensor()])
timg = timg_pipe(raw_img)
inputs = timg.unsqueeze(0)

# ใช้ Hook กับเลเยอร์ที่ถูกต้อง (ตามที่คุณเช็คเจอในโครงสร้างโมเดล)
target_layer = model[7] # total 13 layers, last layer of encoder (UnetBlock = 7)

with Hook(target_layer) as hook:
    output = model(inputs)
    preds = output.argmax(dim=1)[0].cpu().numpy()
    probs = F.softmax(output, dim=1)[0].detach().cpu().numpy()

# ==========================================================
# 4. POST-PROCESSING (Enhanced for Publication)
# ==========================================================
# --- Part A: Abandoned Lead (เชื่อมเส้นประ + กรอง Noise) ---
abdn_mask_raw = (preds == 3).astype(np.uint8)
kernel = np.ones((3,3), np.uint8)
abdn_dilated = cv2.dilate(abdn_mask_raw, kernel, iterations=1) # เชื่อมเส้นที่ขาด
labeled_abdn = skimage.measure.label(abdn_dilated)
props_abdn = skimage.measure.regionprops(labeled_abdn)

min_abdn_area = 100 
abdn_mask_final = np.zeros_like(abdn_mask_raw)
for p in props_abdn:
    if p.area >= min_abdn_area:
        abdn_mask_final[labeled_abdn == p.label] = 1

abdn_mask_final = cv2.erode(abdn_mask_final, kernel, iterations=1) # หดกลับให้เส้นบางสวย
abdn_pixels = np.sum(abdn_mask_final)
has_abdn = abdn_pixels > 0

# --- Part B: Generator/Leadless (Top-1 Selection) ---
gen_mask = (preds == 1).astype(np.uint8)
labeled_gen = skimage.measure.label(gen_mask)
props_gen = skimage.measure.regionprops(labeled_gen)
# เลือกก้อนที่ใหญ่ที่สุดเพียงก้อนเดียว (รองรับ Leadless)
main_obj = sorted(props_gen, key=lambda x: x.area, reverse=True)[0] if props_gen else None

# ==========================================================
# 5. SCALING & CROPPING (Precise coordinate mapping)
# ==========================================================
crop_arr = None
if main_obj and main_obj.area > 150:
    orig_w, orig_h = raw_img.size
    scale = max(orig_w, orig_h) / IMG_Size
    # คำนวณส่วนที่เป็นสีดำจากการ Pad เพื่อหักลบออกให้พิกัดตรงกับรูปจริง
    pad_y = (IMG_Size - (orig_h / scale)) / 2 if orig_h / scale < IMG_Size else 0
    pad_x = (IMG_Size - (orig_w / scale)) / 2 if orig_w / scale < IMG_Size else 0

    minr, minc, maxr, maxc = main_obj.bbox
    real_bbox = (int((minr-pad_y)*scale), int((minc-pad_x)*scale), 
                 int((maxr-pad_y)*scale), int((maxc-pad_x)*scale))
    
    raw_np = np.array(raw_img)
    f_minr, f_minc, f_maxr, f_maxc = fix_bbox(real_bbox, raw_np.shape, Crop_border)
    crop_arr = raw_np[f_minr:f_maxr, f_minc:f_maxc]

# ==========================================================
# 6. GRAD-CAM (Abandoned Lead focus)
# ==========================================================
target_class = 3
model.zero_grad()

# แก้ไขจุดนี้: ใช้ .sum() เพื่อแปลงเป็น Scalar ก่อน backward
loss = output[0, target_class].sum() 
loss.backward(retain_graph=True)
# ดึง Feature Map จาก Hook
# hook.stored มีขนาด [1, channels, H, W] -> ดึง [channels, H, W] ออกมา
target_activations = hook.stored[0]

#output[0, target_class].backward(retain_graph=True)
# หมายเหตุ: ในสคริปต์นี้ใช้ Activation จาก Hook มาแสดงผล Heatmap
cam = torch.mean(hook.stored, dim=1)[0].cpu().numpy()
cam = np.maximum(cam, 0)
# --- เพิ่มบรรทัดนี้เพื่อกลับสี ---
cam = 1.0 - cam 
# ---------------------------
cam = cv2.resize(cam, (IMG_Size, IMG_Size))
cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

# ==========================================================
# 7. VISUALIZATION
# ==========================================================
fig, ax = plt.subplots(1, 4, figsize=(24, 6))

# A. Detection Overlay
ax[0].imshow(timg.permute(1,2,0))
masked_abdn = np.ma.masked_where(abdn_mask_final == 0, abdn_mask_final)
ax[0].imshow(masked_abdn, cmap='autumn', alpha=0.7)
ax[0].set_title(f"Abandoned Lead: {'FOUND' if has_abdn else 'NOT FOUND'}")

# B. Grad-CAM (Heatmap)
ax[1].imshow(timg.permute(1,2,0))
ax[1].imshow(cam, cmap='jet', alpha=0.45)
ax[1].set_title("Grad-CAM (Attention Map)")

# C. Generator/Leadless Mask
ax[2].imshow(gen_mask, cmap='gray')
ax[2].set_title(f"Gen Mask Area: {main_obj.area if main_obj else 0}")

# D. Cropped ROI
if crop_arr is not None:
    ax[3].imshow(crop_arr)
    ax[3].set_title("Cropped Output (ROI)")
else:
    ax[3].text(0.5, 0.5, "No Generator Found", ha='center')

plt.tight_layout()
plt.show()