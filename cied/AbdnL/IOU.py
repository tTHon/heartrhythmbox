import numpy as np
from pathlib import Path
from scipy import ndimage
from label2mask_priority import process_single_json, CLASS_MAP
import PIL.Image

CLASS_NAMES = {1: "generator", 2: "lead", 3: "abandoned_lead"}

# ─────────────────────────────────────────
# IoU
# ─────────────────────────────────────────
def iou_per_class(mask1: np.ndarray, mask2: np.ndarray,
                  num_classes: int = 4) -> dict:
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
# Centroid Distance (generator)
# ─────────────────────────────────────────
def centroid_distance_generator(mask1: np.ndarray, mask2: np.ndarray,
                                class_id: int = 1) -> float | None:
    """ระยะห่างระหว่าง centroid ของ generator (px) — ยิ่งน้อยยิ่งดี"""
    m1 = (mask1 == class_id)
    m2 = (mask2 == class_id)
    if m1.sum() == 0 or m2.sum() == 0:
        return None
    c1 = ndimage.center_of_mass(m1)
    c2 = ndimage.center_of_mass(m2)
    return float(np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2))

# ─────────────────────────────────────────
# Centerline Proximity (lead / abandoned)
# ─────────────────────────────────────────
def mean_centerline_distance(mask1: np.ndarray, mask2: np.ndarray,
                              class_id: int) -> float | None:
    """
    Mean minimum distance จาก centerline ของ annotator1 → annotator2
    และ annotator2 → annotator1 (symmetric)
    ยิ่งน้อยยิ่งดี — หน่วย px
    """
    pts1 = np.argwhere(mask1 == class_id)
    pts2 = np.argwhere(mask2 == class_id)
    if len(pts1) == 0 or len(pts2) == 0:
        return None

    # subsample ถ้าเส้นยาวมาก เพื่อความเร็ว
    max_pts = 500
    if len(pts1) > max_pts:
        pts1 = pts1[np.random.choice(len(pts1), max_pts, replace=False)]
    if len(pts2) > max_pts:
        pts2 = pts2[np.random.choice(len(pts2), max_pts, replace=False)]

    # mean min distance ทั้งสองทาง
    d12 = np.mean(np.min(np.linalg.norm(pts1[:, None] - pts2[None, :], axis=2), axis=1))
    d21 = np.mean(np.min(np.linalg.norm(pts2[:, None] - pts1[None, :], axis=2), axis=1))
    return float((d12 + d21) / 2)

# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def compute_interannotator_iou(
    annotator1_dir: Path,
    annotator2_dir: Path,
    output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(annotator1_dir.glob("*.json"))

    all_iou       = {1: [], 2: [], 3: []}
    all_centroid  = []        # generator centroid distance
    all_cl_lead   = []        # lead centerline distance
    all_cl_abdn   = []        # abandoned centerline distance

    header = (f"{'File':35s} | {'Gen IoU':8s} | {'Lead IoU':8s} | {'Abdn IoU':8s} "
              f"| {'Gen Ctr(px)':11s} | {'Lead CL(px)':11s} | {'Abdn CL(px)':11s}")
    print(header)
    print("-" * len(header))

    for jf in json_files:
        jf2 = annotator2_dir / jf.name
        if not jf2.exists():
            print(f"⚠️  {jf.name} not found in annotator 2, skipping")
            continue

        tmp1 = output_dir / f"_tmp1_{jf.stem}.png"
        tmp2 = output_dir / f"_tmp2_{jf.stem}.png"

        process_single_json(jf,  tmp1)
        process_single_json(jf2, tmp2)

        mask1 = np.array(PIL.Image.open(tmp1))
        mask2 = np.array(PIL.Image.open(tmp2))

        tmp1.unlink()
        tmp2.unlink()

        # IoU
        iou = iou_per_class(mask1, mask2)

        # Centroid distance — generator
        cd = centroid_distance_generator(mask1, mask2, class_id=1)

        # Centerline distance — lead, abandoned
        cl_lead = mean_centerline_distance(mask1, mask2, class_id=2)
        cl_abdn = mean_centerline_distance(mask1, mask2, class_id=3)

        # Accumulate
        for c in [1, 2, 3]:
            if iou[c] is not None:
                all_iou[c].append(iou[c])
        if cd       is not None: all_centroid.append(cd)
        if cl_lead  is not None: all_cl_lead.append(cl_lead)
        if cl_abdn  is not None: all_cl_abdn.append(cl_abdn)

        def fmt(v): return f"{v:.4f}" if v is not None else "N/A"

        print(f"{jf.name:35s} | {fmt(iou[1]):8s} | {fmt(iou[2]):8s} | {fmt(iou[3]):8s} "
              f"| {fmt(cd):11s} | {fmt(cl_lead):11s} | {fmt(cl_abdn):11s}")

    # ── Summary ──────────────────────────────────────────
    print("-" * len(header))
    print(f"\n===== Mean IoU per Class =====")
    for c, name in CLASS_NAMES.items():
        vals = all_iou[c]
        if vals:
            print(f"  {name:20s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}  (n={len(vals)})")
        else:
            print(f"  {name:20s}: N/A")

    overall = [v for vals in all_iou.values() for v in vals]
    print(f"\n  {'Mean IoU (all)':20s}: {np.mean(overall):.4f} ± {np.std(overall):.4f}")

    print(f"\n===== Supplementary Metrics =====")
    print(f"  {'Generator centroid dist':25s}: "
          f"{np.mean(all_centroid):.2f} ± {np.std(all_centroid):.2f} px"
          if all_centroid else "  Generator centroid dist: N/A")
    print(f"  {'Lead centerline dist':25s}: "
          f"{np.mean(all_cl_lead):.2f} ± {np.std(all_cl_lead):.2f} px"
          if all_cl_lead else "  Lead centerline dist: N/A")
    print(f"  {'Abandoned centerline dist':25s}: "
          f"{np.mean(all_cl_abdn):.2f} ± {np.std(all_cl_abdn):.2f} px"
          if all_cl_abdn else "  Abandoned centerline dist: N/A")


# ใช้งาน
compute_interannotator_iou(
    annotator1_dir = Path(r"C:/CIEDID_data/AbdnL/iou/annotator1"),
    annotator2_dir = Path(r"C:/CIEDID_data/AbdnL/iou/annotator2"),
    output_dir     = Path(r"C:/CIEDID_data/AbdnL/iou/iou_temp")
)