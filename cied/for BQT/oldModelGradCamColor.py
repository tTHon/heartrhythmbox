import pathlib
import platform
import numpy as np
import torch
import skimage.measure
import matplotlib.pyplot as plt
from fastai.vision.all import *

# 1. OS Path Compatibility (Fix for Windows/Linux cross-loading)
if platform.system() == 'Windows': 
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# --- STEP 0: Helper Functions ---

def resize_img(img, small_ax=1024):
    """Resizes image keeping aspect ratio based on the smaller axis."""
    scale_f = small_ax / min(img.size)
    return img.resize((int(np.floor(img.size[0] * scale_f)), int(np.floor(img.size[1] * scale_f))))

def fix_bbox(bbox_org, img_shape, minsize=160):
    """
    Adjusts bounding box with a tight 5% margin and ensures it is square.
    """
    minr, minc, maxr, maxc = bbox_org
    
    # REDUCED MARGIN: Tight 0% fit around the hardware
    margin_percentage = 0
    dr = int((maxr - minr) * margin_percentage)
    dc = int((maxc - minc) * margin_percentage)
    
    minr, minc, maxr, maxc = minr - dr, minc - dc, maxr + dr, maxc + dc
    
    # Square the crop and check bounds
    h, w = maxr - minr, maxc - minc
    max_side = max(h, w, minsize)
    dr_sq, dc_sq = max_side - h, max_side - w
    
    minr, maxr = max(0, minr - dr_sq // 2), min(img_shape[0], maxr + dr_sq // 2)
    minc, maxc = max(0, minc - dc_sq // 2), min(img_shape[1], maxc + dc_sq // 2)
    
    return int(minr), int(minc), int(maxr), int(maxc)

def get_vanilla_saliency(learn, pil_img, noise_threshold=0.20):
    """
    Computes pixel-level gradients (Vanilla Saliency).
    Includes a noise threshold to make hotspots 'pop' against the background.
    """
    dl = learn.dls.test_dl([pil_img])
    img_t = first(dl)[0]
    img_t.requires_grad_()
    
    learn.model.eval()
    output = learn.model(img_t)
    max_idx = output.argmax(dim=1)
    
    learn.model.zero_grad()
    output[0, max_idx].backward()
    
    # Calculate absolute gradient across RGB channels
    saliency, _ = torch.max(img_t.grad.data.abs(), dim=1)
    saliency = saliency[0].cpu().numpy()
    
    # Normalization (0 to 1)
    if saliency.max() != 0:
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    
    # Thresholding: Kill the 'fog' of low-level noise
    saliency[saliency < noise_threshold] = 0
    
    return saliency

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    
    # Configuration
    file_seg   = 'cied/segmentation.pkl'
    file_manuf = 'cied/classification_manuf.pkl'
    file_model = 'cied/classification_model.pkl'
    img_path   = 'cied/Dataset/A3 full.jpg'

    try:
        print("Loading models and image...")
        learn_seg   = load_learner(file_seg)
        learn_manuf = load_learner(file_manuf)
        learn_model = load_learner(file_model)
        raw_img     = PILImage.create(img_path)
        
        # --- STEP 1: SEGMENTATION ---
        print("Step 1: Locating device...")
        img_1024 = resize_img(raw_img, 1024)
        img_1024_np = np.array(img_1024)
        
        _, mask, _ = learn_seg.predict(img_1024_np)
        mask_np = np.array(mask)
        mask_np[mask_np != 1] = 0 
        
        labeled = skimage.measure.label(mask_np)
        props = skimage.measure.regionprops(labeled)
        
        if len(props) == 0:
            print("Error: Device not found.")
        else:
            # --- STEP 2: CROPPING ---
            print("Step 2: Cropping (Tight Fit)...")
            main_obj = props[np.argmax([p.area for p in props])]
            minr, minc, maxr, maxc = fix_bbox(main_obj.bbox, img_1024_np.shape)
            
            crop_arr = img_1024_np[minr:maxr, minc:maxc]
            crop_img = PILImage.create(crop_arr).resize((256, 256))
            
            # --- STEP 3: CLASSIFICATION ---
            print("Step 3: Predicting Manufacturer and Model...")
            manuf_name, _, manuf_probs = learn_manuf.predict(crop_img)
            model_name, _, model_probs = learn_model.predict(crop_img)
            model_conf = model_probs.max() * 100
            manuf_conf = manuf_probs.max() * 100

            print("-" * 50)
            print(f"RESULT: {manuf_name}({manuf_conf:.2f}%) -> {model_name} ({model_conf:.2f}%)")
            print("-" * 50)

            # --- STEP 4: SALIENCY VISUALIZATION ---
            print("Step 4: Generating Saliency Map...")
            saliency_map = get_vanilla_saliency(learn_model, crop_img)

            # High-end scientific plot
            fig, ax = plt.subplots(1, 2, figsize=(14, 7), facecolor='black')
            
            # Left Plot
            ax[0].imshow(crop_img, cmap='gray')
            ax[0].set_title(f"Target Hardware\n({model_name})", color='white', fontsize=14)
            ax[0].axis('off')

            # Right Plot - Saliency Map
            img_plot = ax[1].imshow(saliency_map, cmap='hot', interpolation='none')
            ax[1].set_title("AI Feature Importance\n(High-Res Gradient)", color='white', fontsize=14)
            ax[1].axis('off')

            # Professional colorbar
            cbar = plt.colorbar(img_plot, ax=ax[1], fraction=0.046, pad=0.04)
            cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
            cbar.set_label('Importance Intensity', color='white', rotation=270, labelpad=15)

            plt.tight_layout()
            plt.show()
        
            # Save final assets
            fig.savefig('cied/Dataset/noMarginSaliency.png', facecolor=fig.get_facecolor())
            crop_img.save('cied/Dataset/noMarginCrop.png')
            print("Done! Files saved to cied/Dataset/")

    except Exception as e:
        print(f"Workflow failed: {e}")