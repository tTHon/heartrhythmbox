import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- 1. Define the Math ---
def gateway_arch_curve(x):
    # Constants for the St. Louis Arch weighted catenary
    A, B, C = 693.8597, 68.7672, 0.0100333
    return A - B * np.cosh(C * x)

def get_derivative(x):
    B, C = 68.7672, 0.0100333
    return -B * C * np.sinh(C * x)

def create_triangle_profile(size):
    h = size * (np.sqrt(3)/2)
    # Equilateral triangle points centered relative to centroid
    return np.array([
        [0, h * (2/3)],        # Top
        [-size/2, -h * (1/3)], # Bottom Left
        [size/2, -h * (1/3)]   # Bottom Right
    ])

# --- 2. Generate the Geometry ---
def generate_data():
    half_span = 299.2239
    num_steps = 60 # Lower count for faster plotting
    
    # Taper settings (Feet)
    base_size = 54.0
    top_size = 17.0
    
    x_vals = np.linspace(-half_span, half_span, num_steps)
    layers = []

    for x in x_vals:
        y = gateway_arch_curve(x)
        
        # Calculate Taper
        progress = 1 - (abs(x) / half_span)
        current_size = base_size - (base_size - top_size) * progress
        
        # Get triangle and Rotate
        triangle = create_triangle_profile(current_size)
        theta = np.arctan(get_derivative(x))
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        # Transform to 3D World Coordinates
        # X = Span, Z = Height, Y = Thickness
        layer_verts = []
        for point in triangle:
            u, v = point[0], point[1] 
            
            wx = x - v * sin_t
            wy = u
            wz = y + v * cos_t
            layer_verts.append([wx, wy, wz])
            
        layers.append(layer_verts)
    
    return np.array(layers)

# --- 3. Visualization Function ---
def view_mesh_matplotlib(layers):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    print("Generating 3D Plot...")
    
    faces = []
    for i in range(len(layers) - 1):
        l1 = layers[i]
        l2 = layers[i+1]
        for k in range(3):
            k_next = (k + 1) % 3
            face = [l1[k], l2[k], l2[k_next], l1[k_next]]
            faces.append(face)

    # --- ADJUST WIREFRAME SETTINGS HERE ---
    # linewidths: 0.1 is very thin, 2.0 is thick.
    # alpha: 0.0 is invisible, 1.0 is solid. 0.2 is ghost-like.
    poly3d = Poly3DCollection(
        faces, 
        alpha=0.8,          # <--- Make it transparent (0.3) to see inside
        edgecolor='black', 
        linewidths=1.5      # <--- Thicker lines (was 0.1)
    )
    
    poly3d.set_facecolor('#0b5394') 
    ax.add_collection3d(poly3d)

    # Calculate limits (Standard scaling code)
    all_points = layers.reshape(-1, 3)
    max_range = np.array([all_points[:,0].max()-all_points[:,0].min(), 
                          all_points[:,1].max()-all_points[:,1].min(), 
                          all_points[:,2].max()-all_points[:,2].min()]).max() / 2.0

    mid_x = (all_points[:,0].max()+all_points[:,0].min()) * 0.5
    mid_y = (all_points[:,1].max()+all_points[:,1].min()) * 0.5
    mid_z = (all_points[:,2].max()+all_points[:,2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_axis_off() 
    ax.view_init(elev=20, azim=-45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data = generate_data()
    view_mesh_matplotlib(data)