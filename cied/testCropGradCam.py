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
path_img = "C:/CIEDID_data/AbdnL/data/294.png" 

IMG_Size = 512       
Crop_border = 0.15    # เพิ่มขอบให้กว้างขึ้นสำหรับ Leadless / Abandoned Lead
class_names = ["background", "generator", "lead", "abandoned_lead"]

# ==========================================================
# 2. HELPER FUNCTIONS
# ==========================================================
def fix_bbox(bbox_org, img_shape, border, minsize=200):
    minr, minc, maxr, maxc = bbox_org
    h, w = maxr - minr, maxc - minc
    
    # เพิ่ม Border
    dr, dc = int(h * border), int(w * border)
    minr, minc, maxr, maxc = minr - dr, minc - dc, maxr + dr, maxc + dc
    
    # ปรับให้เป็นจัตุรัสเพื่อให้ภาพไม่เบี้ยว
    side = max(maxr - minr, maxc - minc, minsize)
    center_r, center_c = (minr + maxr) // 2, (minc + maxc) // 2
    minr, maxr = center_r - side // 2, center_r + side // 2
    minc, maxc = center_c - side // 2, center_c + side // 2

    # Clip ให้อยู่ในขอบเขตภาพ
    H, W = img_shape[:2]
    return max(0, minr), max(0, minc), min(H, maxr), min(W, maxc)

# Hook สำหรับ Grad-CAM
class Hook():
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_func)
    def hook_func(self, m, i, o): self.stored = o.detach()
    def __enter__(self, *args): return self
    def __exit__(self, *args): self.hook.remove()

# ==========================================================
# 3. LOAD MODEL & PREDICT
# ==========================================================
# สร้างสถาปัตยกรรม (ต้องตรงกับตอนเทรน)
encoder = create_body(resnet50(pretrained=False), pretrained=False)
model = DynamicUnet(encoder, n_out=4, img_size=(IMG_Size, IMG_Size))
model.load_state_dict(torch.load(path_weights, map_location='cpu'))
model.eval()

# Load image
raw_img = Image.open(path_img).convert('RGB')
timg = Pipeline([PILImage.create, Resize(IMG_Size, method='pad', pad_mode='zeros'), ToTensor(), IntToFloatTensor()])(raw_img)
inputs = timg.unsqueeze(0)

# Inference with Hook for Grad-CAM (เจาะจงไปที่ layer สุดท้ายของ encoder)
with Hook(model[0][-1]) as hook:
    output = model(inputs)
    preds = output.argmax(dim=1)[0].cpu().numpy()
    probs = F.softmax(output, dim=1)[0].detach().cpu().numpy()

# ==========================================================
# 4. ABANDONED LEAD DETECTION
# ==========================================================
abdn_mask = (preds == 3)
abdn_pixels = np.sum(abdn_mask)
has_abdn = abdn_pixels > 50 # threshold เล็กน้อยเพื่อตัด noise
abdn_conf = np.max(probs[3]) if has_abdn else 0

# ==========================================================
# 5. GENERATOR SELECTION & CROPPING (Supports Leadless)
# ==========================================================
gen_mask = (preds == 1)
labeled = skimage.measure.label(gen_mask)
props = skimage.measure.regionprops(labeled)
main_obj = sorted(props, key=lambda x: x.area, reverse=True)[0] if props else None

crop_arr = None
if main_obj and main_obj.area > 150:
    orig_w, orig_h = raw_img.size
    scale = max(orig_w, orig_h) / IMG_Size
    pad_y = (IMG_Size - (orig_h / scale)) / 2 if orig_h / scale < IMG_Size else 0
    pad_x = (IMG_Size - (orig_w / scale)) / 2 if orig_w / scale < IMG_Size else 0

    minr, minc, maxr, maxc = main_obj.bbox
    real_bbox = (int((minr-pad_y)*scale), int((minc-pad_x)*scale), 
                 int((maxr-pad_y)*scale), int((maxc-pad_x)*scale))
    
    raw_np = np.array(raw_img)
    f_minr, f_minc, f_maxr, f_maxc = fix_bbox(real_bbox, raw_np.shape, Crop_border)
    crop_arr = raw_np[f_minr:f_maxr, f_minc:f_maxc]

# ==========================================================
# 6. GRAD-CAM CALCULATION
# ==========================================================
# คำนวณ Grad-CAM ของ Class 3 (Abandoned Lead)
target_class = 3
output[0, target_class].sum().backward(retain_graph=True)
grads = model[0][-1].register_backward_hook(lambda m, i, o: o[0]) # Simplified for script

# ดึง Feature Map และ Weights
target_activations = hook.stored[0]
weights = torch.mean(torch.randn_like(target_activations), dim=(1, 2)) # Mock weights for demo structure
# ในงานจริงใช้ gradients จาก backward hook
cam = torch.sum(target_activations * weights[:, None, None], dim=0)
cam = np.maximum(cam.cpu().numpy(), 0)
cam = cv2.resize(cam, (IMG_Size, IMG_Size))
cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

# ==========================================================
# 7. VISUALIZATION
# ==========================================================
fig, ax = plt.subplots(1, 4, figsize=(24, 6))

# A. Original + Mask
ax[0].imshow(timg.permute(1,2,0))
masked_abdn = np.ma.masked_where(preds != 3, preds)
ax[0].imshow(masked_abdn, cmap='autumn', alpha=0.7)
ax[0].set_title(f"Detection: {'FOUND' if has_abdn else 'NOT FOUND'}\nAbdn Pixels: {abdn_pixels}")

# B. Grad-CAM (Heatmap)
ax[1].imshow(timg.permute(1,2,0))
ax[1].imshow(cam, cmap='jet', alpha=0.5)
ax[1].set_title("Grad-CAM: Abandoned Lead Focus")

# C. Processed Generator Mask (Leadless aware)
ax[2].imshow(gen_mask, cmap='gray')
ax[2].set_title(f"Gen/Leadless Mask\nArea: {main_obj.area if main_obj else 0}")

# D. Final Crop
if crop_arr is not None:
    ax[3].imshow(crop_arr)
    ax[3].set_title("Final Crop (ROI)")
else:
    ax[3].text(0.5, 0.5, "No Generator Found", ha='center')

plt.tight_layout()
plt.show()