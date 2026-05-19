import os
from PIL import Image

def resize_pngs_in_folder(folder_path):
    # Ensure the folder path exists
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    # Create an output folder so we don't overwrite your original images
    output_folder = os.path.join(folder_path, "resized_50")
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the directory
    for filename in os.listdir(folder_path):
        # Target only PNG files (case-insensitive)
        if filename.lower().endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            
            try:
                with Image.open(img_path) as img:
                    # Calculate new dimensions (50% of original)
                    new_width = int(img.width * 0.25)
                    new_height = int(img.height * 0.25)
                    
                    # Resize the image using high-quality resampling
                    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # Save the scaled image to the output folder
                    save_path = os.path.join(output_folder, filename)
                    resized_img.save(save_path)
                    
                    print(f"Resized: {filename} -> {new_width}x{new_height}")
            
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    print(f"\nDone! All resized images are saved in: {output_folder}")

# --- How to Run It ---
# Replace 'your_folder_path_here' with the actual path to your PNG folder
# Example: r"C:\Users\Name\Pictures" or "./my_images"
target_folder = r"C:/Users/Sirin/Desktop/img" 
resize_pngs_in_folder(target_folder)