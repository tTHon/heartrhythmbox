import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ==========================================
# [SETTINGS] CHANGE COLOR HERE
# ==========================================
ARCH_LINE_COLOR = '#39ff14'  # Neon Green
# Try these others:
# '#ffffff' = White
# '#ff00ff' = Magenta / Pink
# '#00ffff' = Cyan
# '#ffcc00' = Gold
# ==========================================

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
    num_steps = 200 
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

# --- 2. Visualization ---
def view_colored_lines(layers):
    bg_color = '#001a33' # Dark Navy Background

    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_facecolor(bg_color)

    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(bg_color) 
    ax.grid(False)
    ax.axis('off')

    print(f"Plotting with Line Color: {ARCH_LINE_COLOR}")
    
    faces = []
    for i in range(len(layers) - 1):
        l1 = layers[i]
        l2 = layers[i+1]
        for k in range(3):
            k_next = (k + 1) % 3
            face = [l1[k], l2[k], l2[k_next], l1[k_next]]
            faces.append(face)

    # Apply the Color
    poly3d = Poly3DCollection(
        faces, 
        alpha=0.05,               # Very transparent fill (ghost)
        edgecolor=ARCH_LINE_COLOR, # <--- THIS SETS THE LINE COLOR
        linewidths=0.8            
    )
    ax.add_collection3d(poly3d)

    # Hardcoded limits
    ax.set_xlim(-350, 350)
    ax.set_ylim(-350, 350)
    ax.set_zlim(0, 700)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=20, azim=-45)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data = generate_data()
    view_colored_lines(data)