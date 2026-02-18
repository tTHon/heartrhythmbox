import numpy as np
from moviepy.video.io.VideoFileClip import VideoClip

# --- Configuration ---
WIDTH, HEIGHT = 1920, 1080 
FPS = 30
DURATION = 15 # Slower duration for a premium feel

# --- Premium Palette (Dark Mode) ---
# Color 1: 
color1 = np.array([0, 20, 20]) 
# Color 2:   
color2 = np.array([80, 80, 80]) 

def make_frame(t):
    # Create a gradient that moves slowly
    
    # Create a base frame
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    # Create diagonal gradient coordinates
    x = np.linspace(0, 1, WIDTH)
    y = np.linspace(0, 1, HEIGHT)
    xx, yy = np.meshgrid(x, y)
    
    # Slow movement formula based on time 't'
    shift = np.sin(t * (2 * np.pi / DURATION) / 2) * 0.5 + 0.5
    
    # Create diagonal gradient intensity map
    intensity = (xx + yy + shift) / 2
    intensity = np.clip(intensity, 0, 1)
    
    # Interpolate colors based on intensity map
    for i in range(3):
        frame[:, :, i] = (color1[i] * (1 - intensity) + color2[i] * intensity)
    
    return frame

# Create the video clip
clip = VideoClip(make_frame, duration=DURATION)
clip.fps = FPS

# --- Render and Save ---
output_filename = "playground/slideMod/animatedGd.mp4"
# Using a high bitrate for smoother gradients
clip.write_videofile(output_filename, codec='libx264', fps=FPS, bitrate="5000k")
print(f"Premium video saved as {output_filename}")