import cv2
import os
import shutil
from pathlib import Path

# This script applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to all images in a specified folder.
# specify the source folder containing the original images and the destination folder where the processed images will be saved.
source_path = Path("c:/CIEDID_data/Preprocessing/in") # main image files

# copy files from main source to here to process with CLAHE
data_path = Path("c:/CIEDID_data/Preprocessing/1_CLAHE") # processed image files


#clear folder
for file in data_path.iterdir():
    if file.is_file():
        os.remove(file)
print(f"Cleared {data_path} of existing files.")

#copy files from source to data
count = 0
for file in source_path.iterdir():
    if file.is_file():
        destination = data_path / file.name
        shutil.copy2(file, destination)
        count += 1
print(f"Copied {count} files from {source_path} to {data_path}.")

# create a CLAHE object with specified parameters
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# define valid image extensions to process
valid_extensions = ('.jpg', '.png', '.jpeg', '.JPG', '.PNG')

print(f"Starting CLAHE processing in {data_path}...")

count = 0
for file_path in data_path.iterdir():
    if file_path.suffix in valid_extensions:
        # 1. Read image (Grayscale)
        img = cv2.imread(str(file_path), 0)
        
        if img is not None:
            # 2. Apply CLAHE
            final_img = clahe.apply(img)
            
            # 3. Save the processed image
            cv2.imwrite(str(file_path), final_img)
            count += 1
            if count % 10 == 0:
                print(f"Processed {count} images...")
        else:
            print(f"Could not read: {file_path}")

print(f"Finished! Processed {count} images.")