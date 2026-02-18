import os
from PIL import Image

def crop_left_half(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(valid_extensions):
            file_path = os.path.join(input_folder, filename)
            
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    
                    # --- ADJUST THIS VALUE ---
                    # Instead of width // 2, set the exact pixel width you want to keep.
                    # Example: if your image is 3840px wide, and you want to keep 
                    # the first 1900px, set right_boundary = 1900
                    right_boundary = 6085
                    
                    # Define the crop box: (left, upper, right, lower)
                    crop_box = (0, 0, right_boundary, height)
                    
                    cropped_img = img.crop(crop_box)
                    
                    output_path = os.path.join(output_folder, f"cropped_{filename}")
                    cropped_img.save(output_path)
                    print(f"Processed: {filename} -> {output_path}")
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # Use raw string (r'...') to avoid Windows path issues
    input_directory = 'playground\slideMod\source\input'
    output_directory = 'playground\slideMod\source\output'
    
    crop_left_half(input_directory, output_directory)