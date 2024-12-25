import os
import subprocess
import imageio_ffmpeg as ffmpeg

input_folder = 'playground/VDOFormatConversion/videos/input'
output_folder = 'playground/VDOFormatConversion/videos/output'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

ffmpeg_path = ffmpeg.get_ffmpeg_exe()

for filename in os.listdir(input_folder):
    if filename.endswith('.avi'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.webm')
        
        print(f'Converting {input_path} to {output_path}')
        
        ffmpeg_cmd = [
            ffmpeg_path,
            '-i', input_path,
            '-c:v', 'libvpx',
            '-b:v', '1M',
            '-c:a', 'libvorbis',
            output_path
        ]
        
        try:
            subprocess.run(ffmpeg_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f'Error occurred while converting {input_path}: {e}')
        
print('Conversion complete.')