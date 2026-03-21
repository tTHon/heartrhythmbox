import sys
import time
import pathlib
import platform
import torch
import numpy as np
import skimage.measure
from PIL import Image
import types

# ต้อง import fastai ก่อนเพื่อเตรียมฉีด Patch
from fastai.vision.all import *

# ==========================================================
# 1. THE ULTIMATE FIX FOR AMPMode & Windows Paths
# ==========================================================
def apply_ultimate_patches():
    # Fix 1: แก้ปัญหาคลาส AMPMode หายไปใน FastAI เวอร์ชันใหม่
    import fastai.callback.fp16
    if not hasattr(fastai.callback.fp16, 'AMPMode'):
        class AMPMode:
            def __init__(self, *args, **kwargs): pass
        # ฉีดเข้าไปในโมดูลตรงๆ
        fastai.callback.fp16.AMPMode = AMPMode
        sys.modules['fastai.callback.fp16'].AMPMode = AMPMode
    
    # Fix 2: แก้ปัญหา numpy version เก่า
    if not hasattr(np, 'int'): np.int = int
    
    # Fix 3: แก้ปัญหา Path ระหว่าง Linux/Windows
    if platform.system() == 'Windows':
        pathlib.PosixPath = pathlib.WindowsPath

# ==========================================================
# 2. HELPERS
# ==========================================================
def get_bbox_fast(mask_np, img_shape, minsize=160):
    labeled = skimage.measure.label(mask_np == 1)
    props = skimage.measure.regionprops(labeled)
    if not props: return None
    main_obj = props[np.argmax([p.area for p in props])]
    minr, minc, maxr, maxc = main_obj.bbox
    h, w = maxr - minr, maxc - minc
    max_side = max(h, w, minsize)
    dr, dc = max_side - h, max_side - w
    minr, maxr = max(0, minr - dr // 2), min(img_shape[0], maxr + dr // 2)
    minc, maxc = max(0, minc - dc // 2), min(img_shape[1], maxc + dc // 2)
    return int(minr), int(minc), int(maxr), int(maxc)

def preprocess_manual(pil_img, size, device, is_half=False):
    img = pil_img.resize((size, size))
    t = torch.from_numpy(np.array(img)).permute(2,0,1).float().div(255.0)
    # ImageNet Stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    t = (t - mean) / std
    t = t.unsqueeze(0).to(device)
    return t.half() if is_half else t

# ==========================================================
# 3. RUN ENGINE
# ==========================================================
def run_cied_superthon():
    # สำคัญ: ต้องรัน Patch ก่อนโหลด Learner
    apply_ultimate_patches()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_gpu = device.type == 'cuda'
    
    print(f"🚀 Device Ready: {device} | Using GPU: {is_gpu}")

    path_models = pathlib.Path('cied')
    path_test   = path_models / 'Dataset' / 'test'
    path_out    = path_models / 'Dataset' / 'Results'
    path_out.mkdir(parents=True, exist_ok=True)

    img_files = get_image_files(path_test)
    if not img_files: return print("❌ ไม่พบรูปภาพ")

    try:
        print(f"📥 กำลังโหลดโมเดลยักษ์ 2.2GB (ขั้นตอนนี้อาจใช้เวลา)...", end=" ", flush=True)
        t_load = time.time()
        
        # โหลด Learner (ตอนนี้ AMPMode Error ควรจะหายไปแล้ว)
        learn_seg   = load_learner(path_models/'segmentation.pkl', cpu=(not is_gpu))
        learn_manuf = load_learner(path_models/'classification_manuf.pkl', cpu=(not is_gpu))
        learn_model = load_learner(path_models/'classification_model.pkl', cpu=(not is_gpu))

        # เก็บ Vocabulary
        v_manuf = learn_manuf.dls.vocab
        v_model = learn_model.dls.vocab

        # ย้ายไป GPU และปรับเป็นโหมดประเมินผล
        models = [learn_seg.model, learn_manuf.model, learn_model.model]
        for m in models:
            m.to(device)
            if is_gpu: m.half()
            m.eval()
        
        print(f"Done! ({time.time()-t_load:.2f}s)")

        results = []
        start_all = time.time()

        for idx, img_p in enumerate(img_files, 1):
            t_start = time.time()
            print(f"[{idx}/{len(img_files)}] {img_p.name}...", end=" ", flush=True)

            raw_pil = Image.open(img_p).convert('RGB')
            w, h = raw_pil.size
            
            # --- A. Manual Seg ---
            t_seg = preprocess_manual(raw_pil, 1024, device, is_half=is_gpu)
            with torch.no_grad():
                out_seg = learn_seg.model(t_seg)
                mask_np = out_seg.argmax(dim=1)[0].cpu().numpy()

            # --- B. Crop ---
            scale = 1024 / min(w, h)
            img_1024 = raw_pil.resize((int(w*scale), int(h*scale)), resample=Image.BILINEAR)
            bbox = get_bbox_fast(mask_np, np.array(img_1024).shape)
            
            if bbox:
                minr, minc, maxr, maxc = bbox
                crop_pil = Image.fromarray(np.array(img_1024)[minr:maxr, minc:maxc]).resize((256, 256), resample=Image.LANCZOS)

                # --- C. Manual Class ---
                t_cls = preprocess_manual(crop_pil, 256, device, is_half=is_gpu)
                with torch.no_grad():
                    out_m = learn_manuf.model(t_cls)
                    out_g = learn_model.model(t_cls)
                    
                    m_idx = out_m.argmax(dim=1).item()
                    g_idx = out_g.argmax(dim=1).item()
                    
                    m_conf = torch.softmax(out_m.float(), dim=1).max().item() * 100
                    g_conf = torch.softmax(out_g.float(), dim=1).max().item() * 100

                crop_pil.save(path_out / f"crop_{img_p.name}")
                results.append((img_p.name, str(v_manuf[m_idx]), m_conf, str(v_model[g_idx]), g_conf))
                print(f"OK ({time.time()-t_start:.2f}s)")
            else:
                print("Skipped (No BBox)")

        # --- SUMMARY ---
        print("\n" + "="*75)
        print(f"{'Filename':<20} | {'Manufacturer':<15} | {'Model Group':<15}")
        print("-"*75)
        for r in results:
            print(f"{r[0]:<20} | {r[1]:<15} ({r[2]:.1f}%) | {r[3]:<15} ({r[4]:.1f}%)")
        print("="*75)
        print(f"🔥 Total: {time.time()-start_all:.2f}s")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    run_cied_superthon()