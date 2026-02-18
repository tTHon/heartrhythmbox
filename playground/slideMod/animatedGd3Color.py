import numpy as np
from moviepy.video.io.VideoFileClip import VideoClip
from scipy.interpolate import interp1d

# --- Configuration ---
WIDTH, HEIGHT = 1920, 1080 
FPS = 30
DURATION = 25 # Very slow for maximum premium feel

# --- Premium Palette (3 Colors) ---
# 1: Deep Navy, 2: Deep Purple, 3: Dark Charcoal
colors = np.array([
    [10, 20, 40],   # Navy
    [40, 10, 50],   # Purple
    [20, 20, 20]    # Charcoal
])

# Function to interpolate between colors
color_interp = interp1d(np.linspace(0, 1, len(colors)), colors, axis=0)

def make_frame(t):
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    # Create coordinate grid
    x = np.linspace(0, 1, WIDTH)
    y = np.linspace(0, 1, HEIGHT)
    xx, yy = np.meshgrid(x, y)
    
    # Slow diagonal movement formula
    shift = np.sin(t * (2 * np.pi / DURATION) / 2) * 0.5 + 0.5
    
    # Intensity map for diagonal flow
    intensity = (xx + yy + shift) / 2
    intensity = np.clip(intensity, 0, 1)
    
    # Map intensity to interpolated colors
    for i in range(3):
        # Color interpolation logic
        color_val = color_interp(intensity)
        frame[:, :, i] = color_val[:, :, i]
    
    return frame

# Create the video clip
clip = VideoClip(make_frame, duration=DURATION)
clip.fps = FPS

# --- Render and Save ---
output_filename = "playground/slideMod/animatedGd3Color.mp4"
clip.write_videofile(output_filename, codec='libx264', fps=FPS, bitrate="6000k")
print(f"Premium 3-color video saved as {output_filename}")