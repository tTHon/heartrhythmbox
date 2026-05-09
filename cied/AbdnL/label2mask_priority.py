# label2mask_priority.py
# this script converts label .json files to mask images with a specific priority for overlapping classes.
# use this file !!!!

import json
from turtle import width
import numpy as np
import cv2
import PIL.Image
import pathlib
from pathlib import Path

# ==============================
# ✏️ CONFIG — all paths
# ==============================
INPUT_DIR = Path(r"C:/CIEDID_data/AbdnL/data")      # .json
OUTPUT_DIR = Path(r"C:/CIEDID_data/AbdnL/mask")    # mask folder

# Priority: low to high
# here: abandoned_lead over lead and lead will override generator
CLASS_PRIORITY = ["generator", "lead", "abandoned_lead"]
CLASS_MAP = {
    "generator": 1,
    "lead": 2,
    "abandoned_lead": 3
}
# ==============================

Train_Size = 512  # ขนาดที่ใช้ train model
def process_single_json(json_path: Path, output_path: Path):
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    height = data['imageHeight']
    width = data['imageWidth']
    
    final_mask = np.zeros((height, width), dtype=np.uint8)
    
    # line thickness -- adjustable
    # ใน label2mask_priority.py
    # ปรับตามขนาดภาพ (6-12 px หรือประมาณ 0.8% ของความกว้าง)
    dynamic_thickness = max(6, min(12, int(width * 0.008))) 
    print(f"Processing {json_path.name} | Image size: {width}x{height} | Line thickness: {dynamic_thickness}px")

    for label_name in CLASS_PRIORITY:
        class_id = CLASS_MAP[label_name]
        
        layer_mask = np.zeros((height, width), dtype=np.uint8)
        
        target_shapes = [s for s in data['shapes'] if s['label'].lower().strip() == label_name]
        
        for shape in target_shapes:
            points = np.array(shape['points'], dtype=np.int32)
            
            if shape['shape_type'] in ['linestrip', 'line']:
                cv2.polylines(layer_mask, [points], False, class_id, thickness=dynamic_thickness)
            else:
                cv2.fillPoly(layer_mask, [points], class_id)
        

        final_mask = np.where(layer_mask > 0, layer_mask, final_mask)

    # savefile
    PIL.Image.fromarray(final_mask).save(output_path)
    return np.unique(final_mask)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_files = sorted(INPUT_DIR.glob("*.json"))

    if not json_files:
        print(f"❌ No .json files found in {INPUT_DIR}")
        return

    print(f"🚀 Starting conversion of {len(json_files)} files...")
    print(f"{'Filename':40s} | {'Unique IDs'}")
    print("-" * 60)

    files_with_class = {
        1: 0,
        2: 0,
        3: 0,
    }

    for json_file in json_files:
        mask_name = f"{json_file.stem}_mask.png"
        save_path = OUTPUT_DIR / mask_name
        
        try:
            unique_values = process_single_json(json_file, save_path)
            print(f"{mask_name:40s} | {list(unique_values)}")

            for class_id in files_with_class:
                if class_id in unique_values:
                    files_with_class[class_id] += 1
        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {e}")

    print("-" * 60)
    print(f"✅ Success! Saved masks to: {OUTPUT_DIR}")
    print("Summary — files containing each class:")
    print(f"  1 = generator         : {files_with_class[1]}")
    print(f"  2 = lead              : {files_with_class[2]}")
    print(f"  3 = abandoned_lead    : {files_with_class[3]}")
    

if __name__ == "__main__":
    main()

# test line thickness
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from label2mask_priority import process_single_json

# ==============================
# CONFIG
# ==============================
JSON_PATH = Path(r"C:/CIEDID_data/AbdnL/data/n_x14.json")  # เปลี่ยนชื่อไฟล์
# ==============================

# Build mask โดยไม่ต้อง save
import json, cv2, PIL.Image

def build_mask_preview(json_path: Path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    height = data['imageHeight']
    width  = data['imageWidth']
    final_mask = np.zeros((height, width), dtype=np.uint8)

    dynamic_thickness = max(6, min(12, int(width * 0.008)))

    print(f"Image size : {width} x {height}")
    print(f"Thickness  : {dynamic_thickness} px")

    from label2mask_priority import CLASS_PRIORITY, CLASS_MAP
    for label_name in CLASS_PRIORITY:
        class_id = CLASS_MAP[label_name]
        layer_mask = np.zeros((height, width), dtype=np.uint8)
        target_shapes = [s for s in data['shapes'] if s['label'].lower().strip() == label_name]
        for shape in target_shapes:
            points = np.array(shape['points'], dtype=np.int32)
            if shape['shape_type'] in ['linestrip', 'line']:
                cv2.polylines(layer_mask, [points], False, class_id, thickness=dynamic_thickness)
            else:
                cv2.fillPoly(layer_mask, [points], class_id)
        final_mask = np.where(layer_mask > 0, layer_mask, final_mask)

    return final_mask

# Plot
mask = build_mask_preview(JSON_PATH)

COLOR_MAP = np.array([
    [0,   0,   0],    # 0 = background → black
    [0,   255, 0],    # 1 = generator  → green
    [255, 165, 0],    # 2 = lead       → orange
    [0,   200, 255],  # 3 = abandoned  → cyan
], dtype=np.uint8)

rgb = COLOR_MAP[mask]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: mask only
axes[0].imshow(rgb)
axes[0].set_title(f"Mask  (thickness={max(1, int(mask.shape[1]*0.008))}px)")
axes[0].axis('off')

# Right: zoom บริเวณ lead เพื่อดู thickness
# crop แถบขวาของ image ที่มักมี generator + lead
h, w = mask.shape
crop = rgb[0:h, w//2:]
axes[1].imshow(crop)
axes[1].set_title("Zoom right half — ดู lead thickness")
axes[1].axis('off')

# Legend
patches = [
    mpatches.Patch(color='green',  label='1 = generator'),
    mpatches.Patch(color='orange', label='2 = lead'),
    mpatches.Patch(color='cyan',   label='3 = abandoned_lead'),
]
fig.legend(handles=patches, loc='lower center', ncol=3)
plt.tight_layout()
plt.show()

# Summary
print(f"\nUnique classes in mask: {np.unique(mask).tolist()}")
print(f"Generator pixels : {(mask==1).sum()}")
print(f"Lead pixels      : {(mask==2).sum()}")
print(f"Abandoned pixels : {(mask==3).sum()}")
"""