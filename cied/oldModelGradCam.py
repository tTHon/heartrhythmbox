import pathlib
import platform
import numpy as np
import skimage.measure
import matplotlib.pyplot as plt  # Changed to avoid conflict with platform.system()
from fastai.vision.all import *
from fastai.interpret import *

# 1. OS Path Compatibility (Fix for Windows/Linux cross-loading)
sys_platform = platform.system()
if sys_platform == 'Windows': 
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# --- STEP 0: Helper Functions ---
def resize_img(img, small_ax=1024):
    scale_f = small_ax / min(img.size)
    return img.resize((int(np.floor(img.size[0] * scale_f)), int(np.floor(img.size[1] * scale_f))))

def fix_bbox(bbox_org, img_shape, minsize=160):
    minr, minc, maxr, maxc = bbox_org
    # Add 5% margin around the device
    dr, dc = int((maxr-minr)*0.05), int((maxc-minc)*0.05)
    minr, minc, maxr, maxc = minr-dr, minc-dc, maxr+dr, maxc+dc
    # Make square and stay within image bounds
    h, w = maxr-minr, maxc-minc
    max_side = max(h, w, minsize)
    dr, dc = max_side-h, max_side-w
    minr, maxr = max(0, minr - dr//2), min(img_shape[0], maxr + dr//2)
    minc, maxc = max(0, minc - dc//2), min(img_shape[1], maxc + dc//2)
    return int(minr), int(minc), int(maxr), int(maxc)

def get_gradcam_map(learn, pil_img):
    """Calculates the Heatmap for the predicted class"""
    # 1. Prepare data
    dl = learn.dls.test_dl([pil_img])
    batch = dl.one_batch()
    
    # 2. Define the Hook function (simply stores the output)
    def hooked_func(m, i, o): return o

    # 3. Target the last layer of the encoder
    target_layer = learn.model[0][-1]
    
    # 4. Use Hooks to capture Activations and Gradients
    with Hook(target_layer, hooked_func) as hook_a:        # For Activations
        with Hook(target_layer, hooked_func, is_forward=False) as hook_g: # For Gradients
            preds = learn.model(batch[0])
            max_idx = preds.argmax(dim=1)
            
            # Zero gradients and backpropagate from the winning class
            learn.model.zero_grad()
            preds[0, max_idx].backward()
            
            acts  = hook_a.stored.cpu()
            # Gradients are stored as a list/tuple by the backward hook
            grads = hook_g.stored[0].cpu()
    
    # 5. Calculate weights (Global Average Pooling of gradients)
    weights = grads.mean(dim=(2, 3), keepdim=True)
    
    # 6. Weighted sum of activations
    cam = (weights * acts).sum(dim=1)[0] # Index 0 for the first image in batch
    
    # 7. Apply ReLU and normalize for visualization
    cam = np.maximum(cam, 0)
    if cam.max() != 0:
        cam = cam / cam.max()
        
    return cam


# --- MAIN PROCESS ---
file_seg   = 'cied/segmentation.pkl'
file_manuf = 'cied/classification_manuf.pkl'
file_model = 'cied/classification_model.pkl'

try:
    print("Loading all 3 models...")
    learn_seg   = load_learner(file_seg)
    learn_manuf = load_learner(file_manuf)
    learn_model = load_learner(file_model)

    # Load Source Image
    img_path = 'cied/Dataset/A3 full.jpg' # Change this to test different images
    raw_img = PILImage.create(img_path)
    
    # --- STEP 1: SEGMENTATION ---
    print("Step 1: Segmenting device...")
    img_1024 = resize_img(raw_img, 1024)
    img_1024_np = np.array(img_1024)
    
    _, mask, _ = learn_seg.predict(img_1024_np)
    mask_np = np.array(mask)
    mask_np[mask_np != 1] = 0 # Select only device class
    
    # Find largest connected component
    labeled = skimage.measure.label(mask_np)
    props = skimage.measure.regionprops(labeled)
    
    if len(props) == 0:
        print("Error: Device not found in image.")
    else:
        # --- STEP 2: CROPPING ---
        print("Step 2: Cropping device...")
        main_obj = props[np.argmax([p.area for p in props])]
        minr, minc, maxr, maxc = fix_bbox(main_obj.bbox, img_1024_np.shape)
        
        crop_arr = img_1024_np[minr:maxr, minc:maxc]
        crop_img = PILImage.create(crop_arr).resize((256, 256))
        
        # --- STEP 3: CLASSIFICATION ---
        print("Step 3: Classifying...")
        manuf_name, _, manuf_probs = learn_manuf.predict(crop_img)
        model_name, _, model_probs = learn_model.predict(crop_img)

        print("-" * 40)
        print(f"IMAGE: {img_path}")
        print(f"MANUFACTURER  : {manuf_name} ({manuf_probs.max()*100:.2f}%)")
        print(f"MODEL GROUP   : {model_name} ({model_probs.max()*100:.2f}%)")
        print("-" * 40)

        # --- STEP 4: HEATMAP (GRAD-CAM) ---
        print("Step 4: Generating Grad-CAM...")
        cam_map = get_gradcam_map(learn_model, crop_img)

        # Visualization
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original Cropped Image
        ax[0].imshow(crop_img)
        ax[0].set_title(f"Target: {model_name}")
        ax[0].axis('off')

        # Heatmap Overlay
        ax[1].imshow(crop_img)
        ax[1].imshow(cam_map, alpha=0.5, cmap='magma', 
                   extent=(0,256,256,0), interpolation='bilinear')
        ax[1].set_title("Grad-CAM (Model Focus)")
        ax[1].axis('off')

        plt.tight_layout()
        #plt.show()
        
        # Save Outputs
        crop_img.save('cied/Dataset/debug_crop.png')
        fig.savefig('cied/Dataset/gradcam_result.png')
        print("Done! Result saved to 'cied/Dataset/gradcam_result.png'")

except FileNotFoundError:
    print("Error: Model files not found. Check your paths.")
except Exception as e:
    print(f"An error occurred: {e}")