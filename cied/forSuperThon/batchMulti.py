import sys
import pathlib
import platform
import os
import numpy as np
import skimage.measure
from PIL import Image
from fastai.vision.all import *
import time
import csv
import warnings

# ==========================================================
# STEP 1: SETUP & PATCHES
# ==========================================================
def apply_patches():
    warnings.filterwarnings("ignore")
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

def fix_bbox(bbox_org, img_shape, minsize=160):
    minr, minc, maxr, maxc = bbox_org
    dr, dc = int((maxr-minr)*0.05), int((maxc-minc)*0.05)
    minr, minc, maxr, maxc = minr-dr, minc-dc, maxr+dr, maxc+dc
    h, w = maxr-minr, maxc-minc
    max_side = max(h, w, minsize)
    dr, dc = max_side-h, max_side-w
    minr, maxr = max(0, minr - dr//2), min(img_shape[0], maxr + dr//2)
    minc, maxc = max(0, minc - dc//2), min(img_shape[1], maxc + dc//2)
    return int(minr), int(minc), int(maxr), int(maxc)

# ==========================================================
# STEP 2: BATCH ENGINE
# ==========================================================
def run_batch_ultimate(input_folder, output_folder, batch_size=16):
    file_seg    = 'cied/segmentation.pkl'
    file_manuf  = 'cied/classification_manuf.pkl'
    file_model  = 'cied/classification_model.pkl'
    csv_output  = Path(output_folder) / 'classification_results.csv'

    try:
        apply_patches()
        in_dir, out_dir = Path(input_folder), Path(output_folder)
        out_dir.mkdir(exist_ok=True, parents=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n🚀 Running on: {torch.cuda.get_device_name(0)} | Batch Size: {batch_size}")
        
        # 1. LOAD MODELS & MOVE TO GPU
        print("1. Loading Models...")
        learn_seg = load_learner(file_seg)
        learn_manuf = load_learner(file_manuf)
        learn_model = load_learner(file_model)

        # รีดพลัง RTX 5060 ด้วย Half Precision (FP16)
        learn_seg.model.to(device).half().eval()
        learn_manuf.model.to(device).half().eval()
        learn_model.model.to(device).half().eval()

        # 2. DATA PREPARATION
        files = get_image_files(in_dir)
        # สร้าง DataLoader เพื่อใช้ CPU หลายหัวช่วยเตรียมภาพ (num_workers)
        dl_seg = learn_seg.dls.test_dl(files, bs=batch_size, num_workers=4)
        
        print(f"2. Segmenting {len(files)} images in batches...")
        start_time = time.time()
        
        # 3. BATCH SEGMENTATION (รีดพลัง GPU ตรงนี้)
        with torch.inference_mode():
            all_masks, _ = learn_seg.get_preds(dl=dl_seg)
        
        # 4. CROP & PREPARE FOR CLASSIFICATION
        print("3. Cropping and Classifying...")
        crop_list = []
        valid_files = []
        
        for i, fname in enumerate(files):
            raw_img = Image.open(fname).convert('RGB')
            w, h = raw_img.size
            # Resize สำหรับ Masking logic
            scale = 1024 / min(w, h)
            img_1024 = raw_img.resize((int(w*scale), int(h*scale)))
            
            mask_np = all_masks[i].argmax(dim=0).cpu().numpy()
            mask_np[mask_np != 1] = 0
            
            labeled = skimage.measure.label(mask_np)
            props = skimage.measure.regionprops(labeled)
            
            if len(props) > 0:
                main_obj = props[np.argmax([p.area for p in props])]
                minr, minc, maxr, maxc = fix_bbox(main_obj.bbox, np.array(img_1024).shape)
                
                crop = img_1024.crop((minc, minr, maxc, maxr)).resize((256, 256))
                crop.save(out_dir / f"crop_{fname.name}")
                crop_list.append(tensor(crop).permute(2,0,1).float().div(255.0))
                valid_files.append(fname.name)

        # 5. BATCH CLASSIFICATION
        results_list = []
        if crop_list:
            # รวมภาพ Crop เป็น Batch เดียวกัน
            batch_t = torch.stack(crop_list).to(device).half()
            
            with torch.inference_mode():
                out_m = learn_manuf.model(batch_t)
                out_g = learn_model.model(batch_t)
                
                # หาค่าสูงสุด (Predictions)
                m_preds = out_m.argmax(dim=1)
                g_preds = out_g.argmax(dim=1)
                m_confs = torch.softmax(out_m.float(), dim=1).max(dim=1)[0] * 100
                g_confs = torch.softmax(out_g.float(), dim=1).max(dim=1)[0] * 100

            # บันทึกผล
            v_manuf = learn_manuf.dls.vocab
            v_model = learn_model.dls.vocab
            
            for idx, name in enumerate(valid_files):
                results_list.append([
                    name, v_manuf[m_preds[idx]], f"{m_confs[idx]:.2f}",
                    v_model[g_preds[idx]], f"{g_confs[idx]:.2f}"
                ])

        # 6. EXPORT
        with open(csv_output, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['Filename', 'Manufacturer', 'Conf_Manuf(%)', 'Model_Group', 'Conf_Model(%)'])
            writer.writerows(results_list)

        print(f"✅ All Done! Total Time: {time.time()-start_time:.2f}s")
        print(f"📊 Results saved to: {csv_output}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    run_batch_ultimate('cied/Dataset/test', 'cied/Dataset/results', batch_size=16)