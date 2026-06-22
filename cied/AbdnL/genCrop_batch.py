import argparse
import pathlib
import json
import numpy as np
import cv2
import torch
import scipy.ndimage as ndimage
import skimage.measure
from skimage.morphology import convex_hull_image
from fastai.vision.all import *
from tqdm import tqdm

# ── PyTorch 2.6+ fix ────────────────────────────────────────────────
_orig_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault('weights_only', False)
    return _orig_load(*a, **kw)
torch.load = _patched_load

import platform
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

CLASS_NAMES = ["background", "generator", "lead", "abandoned_lead"]
GEN_CLASS   = 1
N_CLASSES   = len(CLASS_NAMES)


# ══════════════════════════════════════════════════════════════════════
# 1. Functions ต้นฉบับจากไฟล์ eval_generator.py เป๊ะ ๆ
# ══════════════════════════════════════════════════════════════════════
def bbox_from_mask(mask: np.ndarray):
    if mask.sum() == 0:
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    minr, maxr = np.where(rows)[0][[0, -1]]
    minc, maxc = np.where(cols)[0][[0, -1]]
    return int(minr), int(minc), int(maxr)+1, int(maxc)+1


def add_border(bbox, img_shape, border_frac=0.05, min_size=160):
    minr, minc, maxr, maxc = bbox
    dr, dc = int((maxr-minr)*border_frac), int((maxc-minc)*border_frac)
    minr, minc, maxr, maxc = minr-dr, minc-dc, maxr+dr, maxc+dc

    h, w = maxr-minr, maxc-minc
    max_side = max(h, w, min_size)
    dr, dc   = max_side-h, max_side-w
    minr, maxr = max(0, minr - dr//2), min(img_shape[0], maxr + dr//2)
    minc, maxc = max(0, minc - dc//2), min(img_shape[1], maxc + dc//2)
    return int(minr), int(minc), int(maxr), int(maxc)


def postprocess_generator_mask(raw_mask: np.ndarray,
                               img_size: int,
                               min_area_frac: float = 0.00227,
                               dilate_iter: int = 3,
                               erode_iter: int = 3,
                               spatial_prior: dict = None):
    mask = raw_mask.astype(np.uint8)
    if mask.ndim > 2:
        mask = mask.squeeze()

    struct = ndimage.generate_binary_structure(2, 2)

    # ── Step 1: Closing ──
    processed = ndimage.binary_dilation(mask, structure=struct, iterations=dilate_iter).astype(np.uint8)
    processed = ndimage.binary_erosion(processed, structure=struct, iterations=erode_iter).astype(np.uint8)

    # ── Step 2: Fill holes ──
    processed = ndimage.binary_fill_holes(processed).astype(np.uint8)

    # ── Step 3: Label components, apply spatial prior, keep largest ──
    labeled = skimage.measure.label(processed)
    props   = skimage.measure.regionprops(labeled)

    min_area = (img_size * img_size) * min_area_frac
    large_props = [p for p in props if p.area > min_area]

    if spatial_prior is not None:
        h, w = processed.shape
        row_lo = spatial_prior.get("row_min", 0.0) * h
        row_hi = spatial_prior.get("row_max", 1.0) * h
        col_lo = spatial_prior.get("col_min", 0.0) * w
        col_hi = spatial_prior.get("col_max", 1.0) * w

        filtered = []
        for p in large_props:
            cy, cx = p.centroid
            if row_lo <= cy <= row_hi and col_lo <= cx <= col_hi:
                filtered.append(p)
        large_props = filtered

    if len(large_props) == 0:
        return processed, None, 0

    main_obj   = large_props[np.argmax([p.area for p in large_props])]
    isolated   = (labeled == main_obj.label).astype(np.uint8)

    # ── Step 4: Convex hull ──
    final_mask = convex_hull_image(isolated).astype(np.uint8)
    bbox       = bbox_from_mask(final_mask)
    area       = int(final_mask.sum())

    return final_mask, bbox, area


def load_spatial_prior(json_path) -> dict:
    json_path = pathlib.Path(json_path)
    if not json_path.exists():
        print(f"  ⚠️ ไม่พบไฟล์ Spatial prior: {json_path}")
        return None
    with open(json_path) as f:
        prior = json.load(f)
    print(f"  📍 Spatial prior loaded from {json_path.name}")
    return {"row_min": prior["row_min"], "row_max": prior["row_max"],
            "col_min": prior["col_min"], "col_max": prior["col_max"]}


# ══════════════════════════════════════════════════════════════════════
# 2. Build DataFrame & DataLoaders
# ══════════════════════════════════════════════════════════════════════
def build_inference_dataframe(img_dir):
    rows = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in {".jpg", ".png", ".jpeg", ".bmp"}:
            continue
        rows.append({"image": str(img.resolve()), "mask": str(img.resolve()), "is_valid": True})
    df = pd.DataFrame(rows)
    print(f"\n✅ พบรูปภาพที่จะทำการประมวลผล: {len(df)} ภาพ")
    return df


def build_dls(test_df, img_size):
    def get_x(r): return r["image"]
    def get_y(r): return r["mask"]

    BS_DUMMY = 4
    dummy = pd.concat([test_df.iloc[[0]]] * BS_DUMMY, ignore_index=True)
    dummy["is_valid"] = False
    eval_df = pd.concat([dummy, test_df], ignore_index=True)

    dblock = DataBlock(
        blocks    = (ImageBlock, MaskBlock(codes=CLASS_NAMES)),
        get_x     = get_x,
        get_y     = get_y,
        splitter  = ColSplitter(col="is_valid"),
        item_tfms = Resize(img_size, method='pad', pad_mode='zeros'),
        batch_tfms = [Normalize.from_stats([0.5027, 0.5027, 0.5027], [0.2410, 0.2410, 0.2410])],
    )
    return dblock.dataloaders(eval_df, bs=BS_DUMMY, num_workers=0, pin_memory=True)


# ══════════════════════════════════════════════════════════════════════
# 3. Main Executable
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="C:/CIEDID_data/AbdnL/test_data")
    parser.add_argument("--output_dir", default="C:/CIEDID_data/AbdnL/test_data_gen_crop")
    parser.add_argument("--folds_dir", default="C:/CIEDID_data/AbdnL/models")
    parser.add_argument("--weight_filename", default="best_abdn.pth")
    parser.add_argument("--use_fold", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--border", type=float, default=0.05)
    parser.add_argument("--min_area_frac", type=float, default=0.00227)
    parser.add_argument("--use_spatial_prior", action="store_true", default=True)
    parser.add_argument("--spatial_prior_json", default="C:/CIEDID_data/AbdnL/spatial_prior.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🖥️  Device: {device}")
    print(f"📁 Output Folder for Cropped Images → {output_dir}")

    spatial_prior = None
    if args.use_spatial_prior:
        spatial_prior = load_spatial_prior(args.spatial_prior_json)

    test_df = build_inference_dataframe(input_dir)
    if len(test_df) == 0:
        print("❌ ไม่พบรูปภาพที่ระบุ")
        return
        
    dls = build_dls(test_df, args.img_size)

    learner = unet_learner(dls, resnet50, n_out=N_CLASSES).to_fp16()
    
    folds_dir = pathlib.Path(args.folds_dir)
    wp = folds_dir / f"fold_{args.use_fold}" / args.weight_filename
    if not wp.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ Weight: {wp}")
        
    print(f"📦 โหลดน้ำหนักจากโมเดล: {wp}")
    state = torch.load(wp, map_location=device)
    learner.model.load_state_dict(state)
    model = learner.model.to(device).eval()

    paths = test_df["image"].tolist()
    path_idx = 0
    success_count = 0
    
    # 📝 สร้าง List เอาไว้สำหรับเก็บรายชื่อไฟล์ที่ไม่พบ Generator
    failed_files = []

    with torch.no_grad():
        pbar = tqdm(dls.valid, total=len(dls.valid), desc="กำลังทำสกัดและเซฟรูป Crop Generator")
        for xb, yb in pbar:
            xb = xb.to(device)
            
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                probs = model(xb).softmax(dim=1)
                preds = probs.argmax(dim=1)

            batch_paths = paths[path_idx:path_idx+len(xb)]
            path_idx += len(xb)

            for i in range(len(xb)):
                if i >= len(batch_paths): break
                img_path_str = batch_paths[i]
                img_p = pathlib.Path(img_path_str)

                pred_np = preds[i].cpu().numpy()
                raw_pred_mask = (pred_np == GEN_CLASS).astype(np.uint8)

                # ทำ Post-process ตามลำดับแก้บั๊กของคุณเป๊ะ ๆ
                _, pred_bbox_raw, _ = postprocess_generator_mask(
                    raw_pred_mask, args.img_size, args.min_area_frac, spatial_prior=spatial_prior
                )

                if pred_bbox_raw is not None:
                    pred_bbox = add_border(pred_bbox_raw, raw_pred_mask.shape, args.border)
                    minr, minc, maxr, maxc = pred_bbox

                    orig_img = cv2.imread(str(img_p))
                    if orig_img is None: 
                        failed_files.append((img_p.name, "ไม่สามารถโหลดไฟล์ภาพดั้งเดิมได้"))
                        continue
                    h_orig, w_orig = orig_img.shape[:2]

                    scale = args.img_size / max(h_orig, w_orig)
                    w_new, h_new = int(w_orig * scale), int(h_orig * scale)
                    pad_x = (args.img_size - w_new) // 2
                    pad_y = (args.img_size - h_new) // 2

                    minr_orig = int(max(0, (minr - pad_y) / scale))
                    minc_orig = int(max(0, (minc - pad_x) / scale))
                    maxr_orig = int(min(h_orig, (maxr - pad_y) / scale))
                    maxc_orig = int(min(w_orig, (maxc - pad_x) / scale))

                    cropped_img = orig_img[minr_orig:maxr_orig, minc_orig:maxc_orig]

                    if cropped_img.size > 0:
                        save_path = output_dir / img_p.name
                        cv2.imwrite(str(save_path), cropped_img)
                        success_count += 1
                    else:
                        failed_files.append((img_p.name, "ขนาดการ Crop เป็น 0 พิกเซล"))
                else:
                    # 🔴 ถ้าเป็น None แปลว่าคัดกรองแล้วไม่พบเครื่อง Generator ในรูปนี้
                    failed_files.append((img_p.name, "โมเดลหาไม่เจอ หรือติดเงื่อนไข Area / Spatial Prior"))

    print(f"\n🎉 เรียบร้อย! ระบบทำการ Crop และ Save สำเร็จ: {success_count}/{len(test_df)} ภาพ")

    # ══════════════════════════════════════════════════════════════════════
    # 4. ส่วนแสดงผลรายงานไฟล์ที่ไม่พบ Generator ลำดับแบบชัดเจน
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print(f"❌ รายงานไฟล์รูปภาพที่ไม่พบ GENERATOR (รวมทั้งหมด {len(failed_files)} ภาพ)")
    print("="*70)
    if len(failed_files) == 0:
        print("  ✨ สมบูรณ์แบบ! ทุกไฟล์สามารถตรวจจับและครอบตัดวัตถุได้ทั้งหมด")
    else:
        for idx, (filename, reason) in enumerate(failed_files, 1):
            print(f"  [{idx}] 📁 {filename}  —>  ❌ สาเหตุ: {reason}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()