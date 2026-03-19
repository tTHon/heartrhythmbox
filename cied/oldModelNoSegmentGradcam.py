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
    img_path   = 'cied/Dataset/final_tight_crop.png'

    try:
        print("Loading models and image...")
        learn_seg   = load_learner(file_seg)
        learn_manuf = load_learner(file_manuf)
        learn_model = load_learner(file_model)
        crop_img     = PILImage.create(img_path)
        
            
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
        fig.savefig('cied/Dataset/final_saliency_report.png', facecolor=fig.get_facecolor())
        #crop_img.save('cied/Dataset/final_tight_crop.png')
        print("Done! Files saved to cied/Dataset/")

    except Exception as e:
        print(f"Workflow failed: {e}")