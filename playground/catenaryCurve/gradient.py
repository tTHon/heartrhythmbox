import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- 1. Geometry Generation ---
def gateway_arch_curve(x):
    # Constants for the weighted catenary (St. Louis Arch)
    A, B, C = 693.8597, 68.7672, 0.0100333
    return A - B * np.cosh(C * x)

def get_derivative(x):
    # Derivative for rotation calculation
    B, C = 68.7672, 0.0100333
    return -B * C * np.sinh(C * x)

def create_triangle_profile(size):
    # Equilateral triangle
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

# --- 2. Visualization & Saving ---
def view_colored_lines(layers):
    bg_color = '#000000' # Black Background

    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_facecolor(bg_color)

    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(bg_color) 
    ax.grid(False)
    ax.axis('off')

    # --- GRADIENT SETTINGS ---
    # Define RGB Colors (0-1 range)
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
        
        # Calculate height (progress) of this segment
        segment_z = (np.mean(l1[:, 2]) + np.mean(l2[:, 2])) / 2
        t = (segment_z - min_z) / (max_z - min_z)
        t = np.clip(t, 0, 1)
        
        # Interpolate Color
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

    ax.set_xlim(-350, 350)
    ax.set_ylim(-350, 350)
    ax.set_zlim(0, 700)
    ax.set_box_aspect((1, 1, 1))
    #ax.view_init(elev=20, azim=-45)
    ax.view_init(elev=15, azim=70)
    
    plt.tight_layout()
    
    # --- SAVE COMMAND ---
    filename = "playground/catenaryCurve/weighted_arch_gradient.svg"
    print(f"Saving {filename} at 300 DPI...")
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor=bg_color)
    print("Done.")
    
    plt.show()

if __name__ == "__main__":
    data = generate_data()
    view_colored_lines(data)