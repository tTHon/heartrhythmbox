import numpy as np
from pathlib import Path
from scipy import ndimage
from label2mask_priority import process_single_json, CLASS_MAP
import PIL.Image

CLASS_NAMES = {1: "generator", 2: "lead", 3: "abandoned_lead"}

# ─────────────────────────────────────────
# Helper for GIoU: Get Bounding Box from Mask
# ─────────────────────────────────────────
def get_bbox_from_mask(mask: np.ndarray):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return np.array([rmin, cmin, rmax, cmax]) # [y1, x1, y2, x2]

# ─────────────────────────────────────────
# GIoU Calculation
# ─────────────────────────────────────────
def giou_score(mask1: np.ndarray, mask2: np.ndarray, class_id: int) -> float | None:
    m1 = (mask1 == class_id)
    m2 = (mask2 == class_id)
    
    bbox1 = get_bbox_from_mask(m1)
    bbox2 = get_bbox_from_mask(m2)
    
    if bbox1 is None or bbox2 is None:
        return None

    # Intersection coordinates
    y1_inter = max(bbox1[0], bbox2[0])
    x1_inter = max(bbox1[1], bbox2[1])
    y2_inter = min(bbox1[2], bbox2[2])
    x2_inter = min(bbox1[3], bbox2[3])
    
    inter_area = max(0, y2_inter - y1_inter + 1) * max(0, x2_inter - x1_inter + 1)
    
    # Area of each box
    area1 = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    area2 = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
    union_area = area1 + area2 - inter_area
    iou = inter_area / (float(union_area) + 1e-6)
    
    # Convex Hull (Smallest enclosing box) coordinates
    y1_c = min(bbox1[0], bbox2[0])
    x1_c = min(bbox1[1], bbox2[1])
    y2_c = max(bbox1[2], bbox2[2])
    x2_c = max(bbox1[3], bbox2[3])
    
    c_area = (y2_c - y1_c + 1) * (x2_c - x1_c + 1)
    
    # GIoU Formula: IoU - (C \ (A U B)) / C
    giou = iou - (float(c_area - union_area) / (float(c_area) + 1e-6))
    return float(giou)

# ─────────────────────────────────────────
# IoU (Pixel-wise)
# ─────────────────────────────────────────
def iou_per_class(mask1: np.ndarray, mask2: np.ndarray, num_classes: int = 4) -> dict:
    results = {}
    for c in range(1, num_classes):
        m1 = (mask1 == c)
        m2 = (mask2 == c)
        if m1.sum() == 0 and m2.sum() == 0:
            results[c] = None
            continue
        inter = (m1 & m2).sum()
        union = (m1 | m2).sum()
        results[c] = float(inter) / (float(union) + 1e-6)
    return results

# ─────────────────────────────────────────
# Centroid & Centerline Distance (Existing)
# ─────────────────────────────────────────
def centroid_distance_generator(mask1: np.ndarray, mask2: np.ndarray, class_id: int = 1) -> float | None:
    m1 = (mask1 == class_id)
    m2 = (mask2 == class_id)
    if m1.sum() == 0 or m2.sum() == 0: return None
    c1 = ndimage.center_of_mass(m1)
    c2 = ndimage.center_of_mass(m2)
    return float(np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2))

def mean_centerline_distance(mask1: np.ndarray, mask2: np.ndarray, class_id: int) -> float | None:
    pts1 = np.argwhere(mask1 == class_id)
    pts2 = np.argwhere(mask2 == class_id)
    if len(pts1) == 0 or len(pts2) == 0: return None
    max_pts = 500
    if len(pts1) > max_pts: pts1 = pts1[np.random.choice(len(pts1), max_pts, replace=False)]
    if len(pts2) > max_pts: pts2 = pts2[np.random.choice(len(pts2), max_pts, replace=False)]
    d12 = np.mean(np.min(np.linalg.norm(pts1[:, None] - pts2[None, :], axis=2), axis=1))
    d21 = np.mean(np.min(np.linalg.norm(pts2[:, None] - pts1[None, :], axis=2), axis=1))
    return float((d12 + d21) / 2)

# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def compute_interannotator_iou(annotator1_dir: Path, annotator2_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(annotator1_dir.glob("*.json"))

    all_iou = {1: [], 2: [], 3: []}
    all_giou = {1: [], 2: [], 3: []}
    all_centroid, all_cl_lead, all_cl_abdn = [], [], []

    header = (f"{'File':35s} | {'Gen IoU':8s} | {'Gen GIoU':8s} | {'Lead IoU':8s} "
              f"| {'Lead CL(px)':11s}")
    print(header)
    print("-" * len(header))

    for jf in json_files:
        jf2 = annotator2_dir / jf.name
        if not jf2.exists(): continue

        tmp1, tmp2 = output_dir / f"_tmp1_{jf.stem}.png", output_dir / f"_tmp2_{jf.stem}.png"
        process_single_json(jf, tmp1)
        process_single_json(jf2, tmp2)
        mask1, mask2 = np.array(PIL.Image.open(tmp1)), np.array(PIL.Image.open(tmp2))
        tmp1.unlink(); tmp2.unlink()

        iou = iou_per_class(mask1, mask2)
        
        # Calculate GIoU for each class
        giou_vals = {c: giou_score(mask1, mask2, c) for c in [1, 2, 3]}
        
        cd = centroid_distance_generator(mask1, mask2, class_id=1)
        cl_lead = mean_centerline_distance(mask1, mask2, class_id=2)
        cl_abdn = mean_centerline_distance(mask1, mask2, class_id=3)

        for c in [1, 2, 3]:
            if iou[c] is not None: all_iou[c].append(iou[c])
            if giou_vals[c] is not None: all_giou[c].append(giou_vals[c])
        if cd is not None: all_centroid.append(cd)
        if cl_lead is not None: all_cl_lead.append(cl_lead)
        if cl_abdn is not None: all_cl_abdn.append(cl_abdn)

        def fmt(v): return f"{v:.4f}" if v is not None else "N/A"
        print(f"{jf.name:35s} | {fmt(iou[1]):8s} | {fmt(giou_vals[1]):8s} | {fmt(iou[2]):8s} | {fmt(cl_lead):11s}")

    print("-" * len(header))
    print(f"\n===== Mean Results per Class =====")
    for c, name in CLASS_NAMES.items():
        miou = np.mean(all_iou[c]) if all_iou[c] else 0
        siou = np.std(all_iou[c]) if all_iou[c] else 0
        mgiou = np.mean(all_giou[c]) if all_giou[c] else 0
        sgiou = np.std(all_giou[c]) if all_giou[c] else 0
        print(f"  {name:20s}: IoU={miou:.4f}±{siou:.4f}, GIoU={mgiou:.4f}±{sgiou:.4f} (n={len(all_iou[c])})")

    print(f"\n===== Supplementary Metrics =====")
    if all_centroid: print(f"  Generator centroid dist: {np.mean(all_centroid):.2f}±{np.std(all_centroid):.2f} px")
    if all_cl_lead: print(f"  Lead centerline dist:    {np.mean(all_cl_lead):.2f}±{np.std(all_cl_lead):.2f} px")
    if all_cl_abdn: print(f"  Abandoned centerline dist: {np.mean(all_cl_abdn):.2f}±{np.std(all_cl_abdn):.2f} px")

# ใช้งาน
compute_interannotator_iou(
    annotator1_dir = Path(r"C:/CIEDID_data/AbdnL/iou/annotator1"),
    annotator2_dir = Path(r"C:/CIEDID_data/AbdnL/iou/annotator2"),
    output_dir     = Path(r"C:/CIEDID_data/AbdnL/iou/iou_temp")
)