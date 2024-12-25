import os
import pydicom
import imageio
import numpy as np

input_folder = 'playground/dcm2webM/videos/dcm'
output_folder = 'playground/dcm2webM/videos/output'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def resize_to_macro_block_size(image, macro_block_size=16):
    if image.ndim == 2:
        height, width = image.shape
    else:
        height, width = image.shape[:2]
    new_height = (height + macro_block_size - 1) // macro_block_size * macro_block_size
    new_width = (width + macro_block_size - 1) // macro_block_size * macro_block_size
    if new_height != height or new_width != width:
        resized_image = np.zeros((new_height, new_width) + image.shape[2:], dtype=image.dtype)
        resized_image[:height, :width] = image
        return resized_image
    return image

def convert_dcm_to_webm(input_file, output_file):
    # Load the DICOM file
    dicom = pydicom.dcmread(input_file)
    # Extract the pixel data
    pixel_data = dicom.pixel_array
    # Ensure pixel data is a sequence of ndimages
    if pixel_data.ndim == 2:
        pixel_data = [pixel_data]
    # Resize images if necessary
    pixel_data = [resize_to_macro_block_size(frame) for frame in pixel_data]
    # Save the pixel data to WebM using imageio
    imageio.mimwrite(output_file, pixel_data, format='webm', codec='vp9')

for filename in os.listdir(input_folder):
    if filename.endswith('.dcm'):
        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + '.webm')
        convert_dcm_to_webm(input_file, output_file)
