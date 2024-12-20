import os
import pydicom
import cv2
import numpy as np

def convert_dcm_to_mp4(input_folder, output_folder, frame_rate=30):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    for filename in os.listdir(input_folder):
        if filename.endswith(".dcm"):
            dcm_path = os.path.join(input_folder, filename)
            try:
                ds = pydicom.dcmread(dcm_path)
                pixel_array = ds.pixel_array

                # Ensure pixel_array is 3D (frames, height, width)
                if pixel_array.ndim == 2:
                    pixel_array = np.expand_dims(pixel_array, axis=0)

                # Ensure pixel_array has 3 channels
                if pixel_array.shape[-1] != 3:
                    pixel_array = np.repeat(pixel_array[..., np.newaxis], 3, axis=-1)

                output_path = os.path.join(output_folder, filename.replace(".dcm", ".mp4"))
                height, width = pixel_array.shape[1:3]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height), True)

                for frame in pixel_array:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_bgr)

                video_writer.release()
                if os.path.exists(output_path):
                    print(f"Successfully converted {dcm_path} to {output_path}")
                else:
                    print(f"Failed to create video file at: {output_path}")
            except Exception as e:
                print(f"Error processing {dcm_path}: {e}")

input_folder = 'model_data/mClips/clips'
output_folder = 'model_data/mClips/mp4'
convert_dcm_to_mp4(input_folder, output_folder)