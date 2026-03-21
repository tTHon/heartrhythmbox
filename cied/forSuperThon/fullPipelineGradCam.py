import sys
import pathlib
import platform
import os
import numpy as np
import torch
import skimage.measure
import matplotlib.pyplot as plt
from PIL import Image
from fastai.vision.all import *

# ==========================================================
# STEP 1: COMPATIBILITY & PATCHES
# ==========================================================
def apply_patches():
    if not hasattr(np, 'int'): np.int = int
    import fastai.callback.fp16
    if not hasattr(fastai.callback.fp16, 'AMPMode'):
        class AMPMode:
            def __init__(self, *args, **kwargs): pass 
        fastai.callback.fp16.AMPMode = AMPMode
        sys.modules['fastai.callback.fp16'].AMPMode = AMPMode
    if platform.system() == 'Windows':
        pathlib.PosixPath = pathlib.WindowsPath
    else:
        pathlib.WindowsPath = pathlib.PosixPath

# ==========================================================
# STEP 2: HELPER FUNCTIONS
# ==========================================================
def resize_img(img, small_ax=1024):
    scale_f = small_ax / min(img.size)
    return img.resize((int(np.floor(img.size[0] * scale_f)), int(np.floor(img.size[1] * scale_f))))

def fix_bbox(bbox_org, img_shape, minsize=160):
    minr, minc, maxr, maxc = bbox_org
    h, w = maxr - minr, maxc - minc
    max_side = max(h, w, minsize)
    dr_sq, dc_sq = max_side - h, max_side - w
    minr, maxr = max(0, minr - dr_sq // 2), min(img_shape[0], maxr + dr_sq // 2)
    minc, maxc = max(0, minc - dc_sq // 2), min(img_shape[1], maxc + dc_sq // 2)
    return int(minr), int(minc), int(maxr), int(maxc)

def get_vanilla_saliency(learn, img_path, noise_threshold=0.20):
    """คำนวณ Pixel-level gradients จากไฟล์ภาพโดยตรงเพื่อเลี่ยง Bug"""
    dl = learn.dls.test_dl([img_path], num_workers=0)
    img_t = first(dl)[0]
    img_t.requires_grad_()
    
    learn.model.eval()
    output = learn.model(img_t)
    max_idx = output.argmax(dim=1)
    
    learn.model.zero_grad()
    output[0, max_idx].backward()
    
    saliency, _ = torch.max(img_t.grad.data.abs(), dim=1)
    saliency = saliency[0].cpu().numpy()
    
    # Normalization & Denoising
    if saliency.max() != 0:
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    saliency[saliency < noise_threshold] = 0
    return saliency

# ==========================================================
# STEP 3: MAIN EXECUTION
# ==========================================================
if __name__ == '__main__':
    apply_patches()
    
    # Configuration
    file_seg    = 'cied/segmentation.pkl'
    file_manuf  = 'cied/classification_manuf.pkl'
    file_model  = 'cied/classification_model.pkl'
    img_path    = 'cied/Dataset/Micra full.png'
    temp_path   = 'cied/Dataset/temp_process.jpg'

    try:
        print("--- CIED Diagnostic Pipeline Started ---")
        learn_seg   = load_learner(file_seg, cpu=True)
        learn_manuf = load_learner(file_manuf, cpu=True)
        learn_model = load_learner(file_model, cpu=True)
        raw_img     = PILImage.create(img_path)

        # 1. SEGMENTATION
        print("Step 1: Locating device...")
        img_1024 = resize_img(raw_img, 1024)
        img_1024_np = np.array(img_1024)
        _, mask, _ = learn_seg.predict(img_1024_np)
        
        labeled = skimage.measure.label(np.array(mask))
        props = skimage.measure.regionprops(labeled)
        
        if len(props) == 0:
            print("❌ Device not found.")
        else:
            # 2. CROPPING
            print("Step 2: Cropping...")
            main_obj = props[np.argmax([p.area for p in props])]
            minr, minc, maxr, maxc = fix_bbox(main_obj.bbox, img_1024_np.shape)
            crop_img = PILImage.create(img_1024_np[minr:maxr, minc:maxc]).resize((256, 256))
            crop_img.save(temp_path) # บันทึกเพื่อใช้เข้า Predict Pipeline

            # 3. CLASSIFICATION
            print("Step 3: Predicting...")
            manuf_name, _, manuf_probs = learn_manuf.predict(temp_path)
            model_name, _, model_probs = learn_model.predict(temp_path)
            
            # 4. SALIENCY MAP (Heatmap)
            print("Step 4: Generating Saliency Map...")
            # ใช้ Model Group เป็นตัวทำ Heatmap เพื่อดูจุดสังเกตรุ่น
            saliency_map = get_vanilla_saliency(learn_model, temp_path)

            # 5. VISUALIZATION & SAVING
            fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
            ax.imshow(np.array(crop_img), cmap='gray')
            # Overlay ด้วย 'hot' หรือ 'magma' เพื่อให้เหมือนรูปตัวอย่าง
            overlay = ax.imshow(saliency_map, cmap='hot', alpha=0.5, interpolation='bilinear')
            
            ax.axis('off')
            plt.title(f"AI Focus: {manuf_name} - {model_name}", color='white', fontsize=15)
            
            # Save the result
            save_name = 'cied/Dataset/superimposed_saliency.png'
            fig.savefig(save_name, facecolor='black', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            
            print(f"✅ Success! Results saved to {save_name}")

    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        import traceback; traceback.print_exc()