import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, hex2color
from matplotlib.patches import Polygon

# 1. Generate Data
x = np.linspace(-4, 4, 1000)
mu = 0
sigma = 1
y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

# 2. Setup Plot
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(20, 10))

# 3. Define Colors
neon_hex = '#0FF0FC' # Bright Cyan
neon_rgb = hex2color(neon_hex)

# 4. Create the Main "Glow" Effect on the Line
# A white core gives it a "hot" look
ax.plot(x, y, color='white', linewidth=2.5, alpha=0.9, zorder=10)
# The main neon line
ax.plot(x, y, color=neon_hex, linewidth=3, alpha=1.0, zorder=9)

# The outer glow layers with decaying opacity
for n in range(1, 15):
    linewidth = 3 + (n * 2.5)
    alpha = 0.2 * (0.8 ** n)
    ax.plot(x, y, color=neon_hex, linewidth=linewidth, alpha=alpha, zorder=9-n)

# 5. Create the Smooth Gradient Fill
# Custom colormap: Transparent at bottom -> Vibrant neon at top
colors = [ (neon_rgb[0], neon_rgb[1], neon_rgb[2], 0.0),  # Bottom
           (neon_rgb[0], neon_rgb[1], neon_rgb[2], 0.8) ] # Top
cm = LinearSegmentedColormap.from_list('neon_gradient', colors, N=100)

# Create a gradient image and stretch it over the plot area
gradient = np.linspace(0, 1, 256).reshape(-1, 1)
img = ax.imshow(gradient, extent=[x.min(), x.max(), 0, y.max()], aspect='auto', cmap=cm, origin='lower', zorder=1)

# Create a polygon mask from the curve data
verts = [(x[0], 0)] + list(zip(x, y)) + [(x[-1], 0)]
poly = Polygon(verts, facecolor='none')
ax.add_patch(poly) # Must add the patch to the axes first

# Clip the gradient image to the shape of the polygon
img.set_clip_path(poly)

# 6. FIX: Adjust Limits to prevent cutoff
# We set the top limit to 115% of the max height to leave room for the glow
ax.set_ylim(-0.01, y.max() * 1.15)
ax.set_xlim(x.min(), x.max())

# 6. Remove Axis and Grid
ax.axis('off')

# 7. Styling
#plt.title('Smooth Neon Normal Distribution', color=neon_hex, fontsize=18, weight='bold')

# --- SAVE COMMAND ---
filename = "playground/bellCurve/smoothNeon.svg"
print(f"Saving {filename} at 300 DPI...")
plt.savefig(filename, dpi=300, bbox_inches='tight')
print("Done.")
plt.show()