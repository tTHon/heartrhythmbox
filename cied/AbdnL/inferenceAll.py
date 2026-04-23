# inferenceAll.py
# This script loads a our best pretrained pth 
# 2 parameters can be adjusted: 
    # threshold for confidence and 
    # pixel_min for minimum pixels 
    # optimal parameters can be found using gridsearch.py
# Results are shown in two tables:
# 1. Case-level (Yes/No) - whether the model predicts the presence of Abdn_Lead in each image
# 2. Pixel-level (separated by class) - precision, recall, and F1-score for each class (Background, Generator, Lead, Abdn_Lead)
# 

import torch
import pathlib
import numpy as np
import pandas as pd
from fastai.vision.all import *
from sklearn.metrics import classification_report
import os

# 1. path and parameter settings
path_img_folder = "C:/CIEDID_data/AbdnL/data"   
path_mask_folder = "C:/CIEDID_data/AbdnL/mask"  
path_weights = "C:/CIEDID_data/AbdnL/models/best_seg.pth"
class_names = ["Background", "Generator", "Lead", "Abdn_Lead"]
# Adjust these parameters based on your gridSearch.py results
threshold = 0.5
pixel_min = 10 

# 2. model loading
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# get a sample image to create dataloader (we only need the structure, not the actual data for training)
first_img = get_image_files(path_img_folder)[0]
dls = SegmentationDataLoaders.from_label_func(
    pathlib.Path("."), bs=1, fnames=[first_img], 
    label_func=lambda x: x, codes=class_names, item_tfms=Resize(512, method='pad', pad_mode='zeros')
)
learn = unet_learner(dls, resnet50, n_out=4)
learn.model.load_state_dict(torch.load(path_weights, map_location=device))
learn.model.eval()

# 3. variables to store results
results_case = [] # Yes/No
all_y_true = []   # pixel-level ground truth (flattened)
all_y_pred = []   # pixel-level predictions (flattened)

print("🚀 Starting Inference and calculating metrics...")

# 4. Loop through each image and its corresponding mask
for img_file in get_image_files(path_img_folder):
    # add _mask to the image filename to get the corresponding mask filename
    mask_name = f"{img_file.stem}_mask{img_file.suffix}"
    mask_path = pathlib.Path(path_mask_folder) / mask_name
    
    if not mask_path.exists():
        continue
    
    # --- # 1: Ground Truth ---
    target_mask = PILMask.create(mask_path).reshape(512, 512)
    y_true = np.array(target_mask)
    gt_exists = (y_true == 3).any() # มีคลาส 3 ในรูปไหม
    
    # --- # 2: Prediction ---
    with torch.no_grad():
        mask_pred, _, probs = learn.predict(img_file)
    
    final_mask = mask_pred.numpy().copy()
    # adjust prediction based on confidence threshold for Abdn_Lead (class 3)
    abdn_probs = probs[3].numpy()
    final_mask[(final_mask == 3) & (abdn_probs < threshold)] = 0
    
    # --- # 3: Check if predicted (Yes/No)
    pred_exists = (np.sum(final_mask == 3) > pixel_min)
    
    # --- # 3: Store results ---
    results_case.append({'Actual': gt_exists, 'Predicted': pred_exists})
    all_y_true.extend(y_true.flatten())
    all_y_pred.extend(final_mask.flatten())

# 5. Table 1: Case-level (Yes/No)
print("\n" + "="*50)
print("Table 1: Case-level Performance (Abdn_Lead Presence: Yes/No)")
print("="*50)
df_case = pd.DataFrame(results_case)
print(classification_report(df_case['Actual'], df_case['Predicted'], 
                            target_names=['No Abdn', 'Has Abdn'], zero_division=0))

# 6. Table 2: Pixel-level (Separate by Class)
print("\n" + "="*50)
print("Table 2: Pixel-level Performance (Separate by Class)")
print("="*50)
report_pixel = classification_report(
    all_y_true, all_y_pred, 
    target_names=class_names, 
    labels=[0, 1, 2, 3],
    output_dict=True,
    zero_division=0
)
df_pixel = pd.DataFrame(report_pixel).transpose()
final_pixel_table = df_pixel.loc[class_names, ['precision', 'recall', 'f1-score']]
final_pixel_table.columns = ['Precision', 'Recall', 'F1-score']
print(final_pixel_table)

# Save to CSV for reporting
# final_table.to_csv("model_performance_report.csv")