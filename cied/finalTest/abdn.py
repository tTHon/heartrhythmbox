"""
run trained model for abnormal lead detection on held-out data.

"""

import torch
import pathlib
import random
import re
import numpy as np
import pandas as pd
from fastai.vision.all import *
from PIL import Image
from tqdm.auto import tqdm


def normalize_id(value):
    image_id = pathlib.Path(str(value).strip()).stem
    image_id = re.sub(r"_crop$", "", image_id, flags=re.IGNORECASE)
    return re.sub(r"\.0$", "", image_id)


# ==========================================================
# 0. REPRODUCIBILITY
# ==========================================================
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ==========================================================
# 1. SETTINGS (ใช้ค่าเดียวกับที่คุณตั้งไว้)
# ==========================================================
path_csv = "C:/CIEDID_data/main_workbook.csv"
path_img_folder = "C:/CIEDID_data/Images_processed"
path_weights = "C:/CIEDID_data/AbdnL/models/best/best_abdn.pth"
path_result_csv = "C:/CIEDID_data/finalTest/finalTest_abdn_results.csv"
IMG_Size = 640
ABDN_CLASS_IDX = 3

# พารามิเตอร์ที่คุณกำหนด
threshold = 0.8  # softmax probability threshold   
pixel_min = 600 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================================
# 2. LOAD MODEL & PIPELINE
# ==========================================================
labels = pd.read_csv(path_csv, usecols=["ID", "FinalTest", "isAbandon"], dtype={"ID": "string"})
labels["FinalTest"] = pd.to_numeric(labels["FinalTest"], errors="coerce")
labels["isAbandon"] = pd.to_numeric(labels["isAbandon"], errors="coerce")
test_rows = labels[labels["FinalTest"].eq(1)].dropna(subset=["ID"]).copy()
test_rows["ID"] = test_rows["ID"].map(normalize_id)
if test_rows["ID"].duplicated().any():
    duplicate_ids = test_rows.loc[test_rows["ID"].duplicated(), "ID"].tolist()
    raise ValueError(f"Duplicate FinalTest IDs in {path_csv}: {duplicate_ids[:20]}")
if test_rows["isAbandon"].isna().any() or not test_rows["isAbandon"].isin([0, 1]).all():
    raise ValueError(f"isAbandon must contain only 0 or 1 for FinalTest rows in {path_csv}")
test_ids = set(test_rows["ID"])
truth_by_id = test_rows.set_index("ID")["isAbandon"].astype(bool).to_dict()
if not test_ids:
    raise ValueError(f"No IDs with FinalTest == 1 were found in {path_csv}")

all_img_files = get_image_files(path_img_folder)
img_files = [path for path in all_img_files if normalize_id(path.stem) in test_ids]
found_ids = {normalize_id(path.stem) for path in img_files}
missing_ids = sorted(test_ids - found_ids)
if missing_ids:
    print(f"WARNING: no image found for {len(missing_ids)} FinalTest IDs: {missing_ids[:20]}")
if not img_files:
    raise FileNotFoundError(f"No FinalTest images found in {path_img_folder}")

print(f"Selected {len(img_files)} images for {len(test_ids)} IDs with FinalTest == 1.")

# โหลดโมเดล (โครงสร้างเดียวกับตอนเทรน)
dls = SegmentationDataLoaders.from_label_func(
    pathlib.Path("."), bs=1, fnames=img_files[:1], 
    label_func=lambda x: x, codes=["B","G","L","A"], 
    item_tfms=Resize(IMG_Size, method='pad')
)
learn = unet_learner(dls, resnet50, n_out=4).to_fp16()
learn.model.load_state_dict(torch.load(path_weights, map_location=device))
learn.model.to(device).eval()

timg_pipe = Pipeline([PILImage.create, Resize(IMG_Size, method='pad', pad_mode='zeros'), ToTensor(), IntToFloatTensor()])
mean = torch.tensor([0.5150052309036255]*3, device=device).view(3, 1, 1)
std = torch.tensor([0.23788487911224365]*3, device=device).view(3, 1, 1)

# ==========================================================
# 3. INFERENCE & REPORTING
# ==========================================================
img_files_sorted = sorted(img_files)
n_imgs = len(img_files_sorted)

print(f"🚀 กำลังประมวลผลรูปภาพทั้งหมด {n_imgs} รูป... กรุณารอสักครู่")

results = []
for i, img_path in enumerate(tqdm(img_files_sorted, desc="Processing FinalTest images", unit="image")):
    # --- Inference ---
    timg = timg_pipe(img_path).to(device)
    timg_norm = (timg - mean) / std
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            probs = learn.model(timg_norm.unsqueeze(0)).softmax(dim=1)[0].cpu()
        abdn_prob_map = probs[ABDN_CLASS_IDX].numpy()
    
    # --- Detection Decision ---
    pixel_count = int((abdn_prob_map > threshold).sum())
    is_detected = pixel_count > pixel_min
    image_id = normalize_id(img_path.stem)
    is_abandon = truth_by_id[image_id]
    correct = is_detected == is_abandon
    results.append({
        "ID": image_id,
        "image": img_path.name,
        "pixel_count": pixel_count,
        "is_detected": is_detected,
        "isAbandon": is_abandon,
        "correct": correct,
    })
    tqdm.write(
        f"ID={image_id} | pixel_count={pixel_count} | "
        f"is_detected={int(is_detected)} | "
        f"isAbandon={int(is_abandon)} | {'MATCH' if correct else 'MISMATCH'}"
    )

results_df = pd.DataFrame(results)
pathlib.Path(path_result_csv).parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(path_result_csv, index=False)
tp = int((results_df["is_detected"] & results_df["isAbandon"]).sum())
tn = int((~results_df["is_detected"] & ~results_df["isAbandon"]).sum())
fp = int((results_df["is_detected"] & ~results_df["isAbandon"]).sum())
fn = int((~results_df["is_detected"] & results_df["isAbandon"]).sum())
accuracy = (tp + tn) / len(results_df)
print(f"\nComparison for {len(results_df)} images (TP={tp}, TN={tn}, FP={fp}, FN={fn})")
print(f"Accuracy: {accuracy:.1%}")
print(f"Per-image results saved to: {path_result_csv}")
