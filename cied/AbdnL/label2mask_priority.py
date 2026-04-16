# label2mask_priority.py
# this script converts label .json files to mask images with a specific priority for overlapping classes.
# use this file !!!!

import json
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

def process_single_json(json_path: Path, output_path: Path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    height = data['imageHeight']
    width = data['imageWidth']
    
    final_mask = np.zeros((height, width), dtype=np.uint8)
    
    # line thickness -- adjustable
    dynamic_thickness = max(1, int(width * 0.008)) 

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

    for json_file in json_files:
        mask_name = f"{json_file.stem}_mask.png"
        save_path = OUTPUT_DIR / mask_name
        
        try:
            unique_values = process_single_json(json_file, save_path)
            print(f"{mask_name:40s} | {list(unique_values)}")
        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {e}")

    print("-" * 60)
    print(f"✅ Success! Saved masks to: {OUTPUT_DIR}")
    print(f"⚠️  Don't forget to check if ID 3 (abandoned_lead) appears in the Unique IDs list")

if __name__ == "__main__":
    main()