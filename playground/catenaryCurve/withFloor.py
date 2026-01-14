import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- 1. Geometry Generation ---
def gateway_arch_curve(x):
    A, B, C = 693.8597, 68.7672, 0.0100333
    return A - B * np.cosh(C * x)

def get_derivative(x):
    B, C = 68.7672, 0.0100333
    return -B * C * np.sinh(C * x)

def create_triangle_profile(size):
    h = size * (np.sqrt(3)/2)
    return np.array([
        [0, h * (2/3)], [-size/2, -h * (1/3)], [size/2, -h * (1/3)]
    ])

def generate_data():
    half_span = 299.2239
    num_steps = 150
    base_size = 54.0
    top_size = 17.0
    
    x_vals = np.linspace(-half_span, half_span, num_steps)
    layers = []

    for x in x_vals:
        y = gateway_arch_curve(x)
        progress = 1 - (abs(x) / half_span)
        current_size = base_size - (base_size - top_size) * progress
        
        triangle = create_triangle_profile(current_size)
        theta = np.arctan(get_derivative(x))
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        layer_verts = []
        for point in triangle:
            u, v = point[0], point[1] 
            wx = x - v * sin_t
            wy = u
            wz = y + v * cos_t
            layer_verts.append([wx, wy, wz])
        layers.append(layer_verts)
    
    return np.array(layers)

# --- 2. Blueprint Helper Functions ---
def add_blueprint_floor(ax, size=400, step=50, color='#00ffff'):
    """Draws a technical grid and border on the Z=0 plane"""
    
    # 1. Draw Grid Lines (X and Y)
    # We use a lower alpha for the grid so it doesn't distract from the arch
    grid_alpha = 0.2
    
    for i in range(-size, size + 1, step):
        # X-lines (constant x, varying y)
        ax.plot([i, i], [-size, size], [0, 0], color=color, alpha=grid_alpha, linewidth=0.5)
        # Y-lines (constant y, varying x)
        ax.plot([-size, size], [i, i], [0, 0], color=color, alpha=grid_alpha, linewidth=0.5)

    # 2. Draw Heavy Border (Frame)
    #ax.plot([-size, size, size, -size, -size], 
            #[-size, -size, size, size, -size], 
            #[0, 0, 0, 0, 0], 
            #color='white', alpha=0.8, linewidth=2.0)

    # 3. Add Corner Markers (Technical Look)
    #marker_len = size * 0.1
    #for corner_x in [-size, size]:
        #for corner_y in [-size, size]:
            # Vertical tick at corner
            #ax.plot([corner_x, corner_x], [corner_y, corner_y], [0, 20], color=color, linewidth=1)

# --- 3. Visualization & Saving ---
def view_colored_lines(layers):
    # CHANGED: Deep Navy Blue for Blueprint feel (instead of pure Black)
   #bg_color = '#001525' 
    bg_color = 'none'
    
    fig = plt.figure(figsize=(16, 16)) # Slightly larger for the frame
    fig.patch.set_facecolor(bg_color)

    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(bg_color) 
    ax.grid(False)
    ax.axis('off')

    # --- ADD THE FLOOR ---
    add_blueprint_floor(ax, size=400, step=50, color='#00ffff')

    # --- GRADIENT SETTINGS ---
    color_base = np.array([1.0, 1.0, 1.0]) # White
    color_top  = np.array([0.0, 1.0, 1.0]) # Neon Cyan

    faces = []
    edge_colors = []

    # Calculate Z-limits for normalization
    all_z = layers[:, :, 2]
    min_z, max_z = np.min(all_z), np.max(all_z)

    for i in range(len(layers) - 1):
        l1 = layers[i]
        l2 = layers[i+1]
        
        segment_z = (np.mean(l1[:, 2]) + np.mean(l2[:, 2])) / 2
        t = (segment_z - min_z) / (max_z - min_z)
        t = np.clip(t, 0, 1)
        
        segment_color = color_base * (1 - t) + color_top * t
        
        for k in range(3):
            k_next = (k + 1) % 3
            face = [l1[k], l2[k], l2[k_next], l1[k_next]]
            faces.append(face)
            edge_colors.append(segment_color)

    # Apply the Gradient Colors
    poly3d = Poly3DCollection(
        faces, 
        alpha=0.07, 
        edgecolors=edge_colors, 
        linewidths=0.9            
    )
    ax.add_collection3d(poly3d)

    # Add Technical Text Labels (Floating in 3D space)
    #ax.text(0, -420, 0, "ELEVATION: 0.00", color='white', fontsize=8, ha='center')
    #ax.text(0, 0, 720, "MAX HEIGHT: 630'", color='cyan', fontsize=8, ha='center')

    ax.set_xlim(-450, 450)
    ax.set_ylim(-450, 450)
    ax.set_zlim(0, 750)
    ax.set_box_aspect((1, 1, 0.8)) # Flattened z slightly for perspective
    ax.view_init(elev=10, azim=-60) # Classic Isometric Drafting Angle
    
    plt.tight_layout()
    
    # --- SAVE COMMAND ---
    # Using 'weighted_arch_blueprint.svg' to reflect the new style
    filename = "playground/catenaryCurve/weighted_arch_blueprint.png"
    print(f"Saving {filename}...")
    plt.savefig(filename, format='png', bbox_inches='tight', facecolor=bg_color, dpi=300)
    print("Done.")
    
    plt.show()

if __name__ == "__main__":
    data = generate_data()
    view_colored_lines(data)