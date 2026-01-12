import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ==========================================
# BLUEPRINT STYLE SETTINGS
# ==========================================
BG_COLOR = '#001a33'    # Deep Blueprint Blue
LINE_COLOR = '#00ffff'  # Glowing Cyan
LINE_WIDTH = 1.0        
# ==========================================

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
    num_steps = 60      
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

def view_blueprint_wiremesh(layers):
    fig = plt.figure(figsize=(12, 12))
    fig.patch.set_facecolor(BG_COLOR)

    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(BG_COLOR) 
    ax.grid(False)
    ax.axis('off')

    faces = []
    for i in range(len(layers) - 1):
        l1 = layers[i]
        l2 = layers[i+1]
        for k in range(3):
            k_next = (k + 1) % 3
            face = [l1[k], l2[k], l2[k_next], l1[k_next]]
            faces.append(face)

    # --- THE FIX IS HERE ---
    # We removed 'alpha=1.0' argument to avoid overriding our custom colors.
    # We changed facecolors to (0,0,0,0) which is RGBA for "Transparent Black".
    poly3d = Poly3DCollection(
        faces, 
        facecolors=(0, 0, 0, 0),  # <--- Explicit transparent tuple
        edgecolor=LINE_COLOR,   
        linewidths=LINE_WIDTH
    )
    ax.add_collection3d(poly3d)

    # Center line
    ax.plot([-300, 300], [0, 0], [0, 0], color=LINE_COLOR, linestyle=':', linewidth=0.5, alpha=0.5)

    limit = 350
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(0, 700)
    
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=20, azim=-50)
    
    plt.tight_layout()
    
    filename = "playground/catenaryCurve/poster.png"
    print(f"Saving {filename}...")
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor=BG_COLOR)
    plt.show()

if __name__ == "__main__":
    data = generate_data()
    view_blueprint_wiremesh(data)