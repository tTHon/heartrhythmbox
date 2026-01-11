import numpy as np

def gateway_arch_curve(x):
    """
    Returns the y (height) coordinate for the weighted catenary.
    """
    A = 693.8597
    B = 68.7672
    C = 0.0100333
    return A - B * np.cosh(C * x)

def get_derivative(x):
    """
    Returns dy/dx to help calculate orientation/rotation of the cross-section.
    Derivative of A - B*cosh(C*x) is -B*C*sinh(C*x).
    """
    B = 68.7672
    C = 0.0100333
    return -B * C * np.sinh(C * x)

def create_triangle_profile(size):
    """
    Creates a simple equilateral triangle centered at (0,0).
    Returns 3 points (x, y).
    """
    h = size * (np.sqrt(3)/2)
    # Point 1: Top
    p1 = [0, h * (2/3)]
    # Point 2: Bottom Left
    p2 = [-size/2, -h * (1/3)]
    # Point 3: Bottom Right
    p3 = [size/2, -h * (1/3)]
    return np.array([p1, p2, p3])

def generate_arch_mesh(filename="playground/catenaryCurve/weighted_arch.stl"):
    # --- Settings ---
    half_span = 299.2239
    num_steps = 100
    
    # Taper settings (Feet) - Gateway arch is ~54ft at base, ~17ft at top
    base_size = 54.0
    top_size = 17.0
    
    # Generate points along the span
    x_vals = np.linspace(-half_span, half_span, num_steps)
    
    # Storage for all vertices in 3D space
    # Shape: (num_steps, 3_vertices_per_layer, 3_coords)
    layers = []

    for i, x in enumerate(x_vals):
        y = gateway_arch_curve(x)
        
        # 1. Calculate Taper Size based on height ratio
        # (Simple linear interpolation based on current x position relative to span)
        progress = 1 - (abs(x) / half_span) # 0 at base, 1 at center
        current_size = base_size - (base_size - top_size) * progress
        
        # 2. Get base triangle
        triangle = create_triangle_profile(current_size)
        
        # 3. Calculate Rotation
        # The tangent angle of the curve
        dy_dx = get_derivative(x)
        theta = np.arctan(dy_dx) # Angle in radians
        
        # Rotation matrix to align triangle perpendicular to the curve
        # We rotate around the Z-axis (which is actually out-of-page in 2D, 
        # but here we treat our 2D curve as lying in X-Z plane, Y is depth).
        # Let's align:
        # World X = Arch Span
        # World Z = Arch Height
        # World Y = Arch Thickness
        
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # Transform triangle points (which are in local Y-Z plane) to World Space
        # Local triangle u (width), v (height)
        layer_verts = []
        for point in triangle:
            u, v = point[0], point[1] # u is width (World Y), v is normal height
            
            # Rotate v to align with curve normal
            # The normal vector is perpendicular to tangent.
            # Effective coordinates:
            # wx = x + v * -sin(theta)
            # wy = u
            # wz = y + v * cos(theta)
            
            wx = x - v * sin_t
            wy = u
            wz = y + v * cos_t
            
            layer_verts.append([wx, wy, wz])
            
        layers.append(layer_verts)
    
    layers = np.array(layers)

    # --- Triangulation (Stitching layers) ---
    faces = []
    
    for i in range(len(layers) - 1):
        # Current layer vertices (A, B, C) and Next layer (A', B', C')
        current_L = layers[i]
        next_L = layers[i+1]
        
        for k in range(3):
            # Indices for triangle corners (0, 1, 2)
            k_next = (k + 1) % 3
            
            # Form two triangles (quad) between the two layers
            v1 = current_L[k]
            v2 = next_L[k]
            v3 = current_L[k_next]
            
            v4 = next_L[k]
            v5 = next_L[k_next]
            v6 = current_L[k_next]
            
            faces.append([v1, v2, v3]) # Triangle 1
            faces.append([v4, v5, v6]) # Triangle 2

    # --- Cap the ends (Base of legs) ---
    # Start cap (Layer 0) - Reverse order for normal consistency
    faces.append([layers[0][0], layers[0][2], layers[0][1]])
    # End cap (Last Layer)
    faces.append([layers[-1][0], layers[-1][1], layers[-1][2]])

    # --- Write to ASCII STL ---
    print(f"Writing {len(faces)} faces to {filename}...")
    with open(filename, 'w') as f:
        f.write("solid weighted_arch\n")
        for face in faces:
            # Calculate normal (simple cross product)
            v0 = np.array(face[0])
            v1 = np.array(face[1])
            v2 = np.array(face[2])
            normal = np.cross(v1 - v0, v2 - v0)
            norm_len = np.linalg.norm(normal)
            if norm_len > 0: normal /= norm_len
            
            f.write(f"facet normal {normal[0]:.4f} {normal[1]:.4f} {normal[2]:.4f}\n")
            f.write("outer loop\n")
            for vertex in face:
                f.write(f"vertex {vertex[0]:.4f} {vertex[1]:.4f} {vertex[2]:.4f}\n")
            f.write("endloop\n")
            f.write("endfacet\n")
        f.write("endsolid weighted_arch\n")
        
    print(f"Done! Saved as {filename}")

if __name__ == "__main__":
    generate_arch_mesh()