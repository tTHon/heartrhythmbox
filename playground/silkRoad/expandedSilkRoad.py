import matplotlib.pyplot as plt
import geopandas as gpd
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION ---
NUM_POINTS = 8000     
CONNECTIVITY = 4      
SCATTER_WIDTH = 1.8   
CURVE_RESOLUTION = 300 # How many points make up a curved segment (higher = smoother)

# Cyberpunk Neon Palette
BG_COLOR = '#050505'       
LAND_COLOR = '#1A1A1A'     
LAND_BORDER = '#333333'    
WEB_COLOR = '#2a0040'      
LAND_ROUTE_GLOW = '#FF0055' 
SEA_ROUTE_GLOW = '#00CCFF'  
NODE_COLOR = '#FFFFFF'     
TEXT_NEON = '#E0FFFF'      

# --- 2. HELPER FUNCTION FOR CURVES ---
def get_curved_path(coords, resolution=CURVE_RESOLUTION):
    """
    Takes a list of (lon, lat) tuples and returns smooth curved arrays of x and y coordinates
    using cubic spline interpolation.
    """
    if len(coords) < 2:
        return [], []
    
    # Separate into x and y lists
    x = [p[0] for p in coords]
    y = [p[1] for p in coords]
    
    # If only 2 or 3 points, a high-degree spline gets weird. Use simpler interpolation.
    if len(coords) <= 3:
        # Linear interpolation for short segments to ensure stability
        t = np.linspace(0, 1, resolution)
        lerp_x = np.interp(t, np.linspace(0, 1, len(x)), x)
        lerp_y = np.interp(t, np.linspace(0, 1, len(y)), y)
        return lerp_x, lerp_y

    # Cubic Spline Interpolation for longer routes
    try:
        # k=3 for cubic spline, s=0 forces it to pass exactly through control points
        tck, u = splprep([x, y], k=3, s=0) 
        u_new = np.linspace(u.min(), u.max(), resolution)
        x_smooth, y_smooth = splev(u_new, tck)
        return x_smooth, y_smooth
    except Exception as e:
        print(f"Spline error on route ending {coords[-1]}: {e}. Falling back to linear.")
        # Fallback if spline fails (e.g., duplicate points)
        t = np.linspace(0, 1, resolution)
        lerp_x = np.interp(t, np.linspace(0, 1, len(x)), x)
        lerp_y = np.interp(t, np.linspace(0, 1, len(y)), y)
        return lerp_x, lerp_y


# --- 3. DEFINE ROUTES (Rationalized Junctions) ---
# Key Junctions defined once to ensure exact connectivity
DUNHUANG = (94.7, 40.1)
KASHGAR = (75.9, 39.5)
SAMARKAND = (66.9, 39.6)
MERV = (62.2, 37.6)
BAGHDAD = (44.4, 33.3)
ANTIOCH_AREA = (36.1, 36.2)

# --- A. CENTRAL SILK ROAD ---
route_corridor = [(108.9, 34.3), (103.8, 36.0), (98.2, 39.7), DUNHUANG] 
route_tarim_north = [DUNHUANG, (93.5, 42.8), (89.2, 42.9), (83.0, 41.7), KASHGAR] 
route_tarim_south = [DUNHUANG, (85.5, 38.1), (82.5, 37.5), (79.9, 37.1), (77.0, 37.5), KASHGAR] 
route_central = [KASHGAR, (70.0, 40.0), SAMARKAND, (63.5, 39.0), MERV] 
route_persia = [MERV, (58.5, 37.9), (54.4, 36.4), (51.4, 35.7), (48.5, 34.8), BAGHDAD] 
route_levant = [BAGHDAD, (40.0, 35.0), ANTIOCH_AREA, (35.5, 33.5), (34.0, 31.0)] 

# --- B. NORTHERN STEPPE ---
route_steppe = [
    (116.4, 39.9), (106.9, 47.9), (88.0, 48.0), (76.9, 43.2), 
    (60.0, 46.0), (48.0, 46.5), (39.0, 47.0), (30.5, 50.4), (26.1, 44.4)
]
route_steppe_connector = [(76.9, 43.2), SAMARKAND]

# --- C. ARABIAN LOOP ---
route_arabia = [
    BAGHDAD, (47.8, 30.5), (51.0, 26.0), (56.0, 21.0), (52.0, 15.0), 
    (44.0, 13.0), (41.0, 18.0), (39.0, 21.5), (35.0, 28.0), (35.5, 31.0)
]

# --- D. SOUTHERN CORRIDOR ---
route_chengdu_singapore = [
    (104.1, 30.7), (102.8, 24.9), (100.5, 19.9), (100.5, 13.7), 
    (99.3, 9.1), (101.7, 3.1), (103.8, 1.35)
]

# --- E. INDIAN WEB ---
route_india_north = [SAMARKAND, (69.2, 34.5), (73.0, 33.7), (77.2, 28.6), (82.0, 26.0), (88.0, 22.5)]
route_india_south = [(77.2, 28.6), (75.0, 22.0), (77.0, 15.0), (79.0, 11.0)]

# --- F. EXTENSIONS ---
route_korea = [(116.4, 39.9), (122.0, 40.5), (125.7, 39.0), (127.0, 37.5)]
route_europe_main = [ANTIOCH_AREA, (28.9, 41.0), (20.0, 42.0), (12.5, 41.9)]
route_europe_north = [(28.9, 41.0), (19.0, 47.0), (13.0, 48.0), (2.3, 48.8)]

# --- G. MARITIME ---
route_maritime = [
    (118.0, 24.5), (113.2, 23.1), (109.0, 13.0), (104.0, 1.3), (95.0, 5.5), (80.0, 6.0),
    (75.0, 10.0), (65.0, 15.0), (50.0, 13.0), (43.0, 12.5), (35.0, 25.0), (32.5, 29.9)
]

land_routes_raw = [
    route_corridor, route_tarim_north, route_tarim_south, route_central, 
    route_persia, route_levant, route_steppe, route_steppe_connector,
    route_arabia, route_chengdu_singapore, route_india_north, route_india_south,
    route_korea, route_europe_main, route_europe_north
]

# --- 4. PROCESS ROUTES (CURVING & WEB GENERATION) ---
print("Smoothing routes and generating web...")
all_x, all_y = [], []
smoothed_land_routes = []

# Process Land Routes into smooth curves
for route in land_routes_raw:
    # 1. Calculate smooth path
    smooth_x, smooth_y = get_curved_path(route)
    smoothed_land_routes.append((smooth_x, smooth_y))
    
    # 2. Use smooth path to seed background noise
    noise_x = np.random.normal(0, SCATTER_WIDTH, size=len(smooth_x))
    noise_y = np.random.normal(0, SCATTER_WIDTH * 0.7, size=len(smooth_y))
    all_x.extend(smooth_x + noise_x)
    all_y.extend(smooth_y + noise_y)

# Also add maritime points to web generation for consistency
m_x, m_y = get_curved_path(route_maritime)
noise_mx = np.random.normal(0, SCATTER_WIDTH, size=len(m_x))
noise_my = np.random.normal(0, SCATTER_WIDTH * 0.7, size=len(m_y))
all_x.extend(m_x + noise_mx)
all_y.extend(m_y + noise_my)


# Build Background Graph
points = np.column_stack((all_x, all_y))
tree = cKDTree(points)
dist_matrix, neighbors = tree.query(points, k=CONNECTIVITY + 1)

G = nx.Graph()
for i in range(len(points)):
    for j in range(1, CONNECTIVITY + 1):
        target = neighbors[i][j]
        G.add_edge(i, target, weight=dist_matrix[i][j])

# --- 5. PLOTTING SETUP ---
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(26, 14))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)
ax.set_xlim(0, 135)
ax.set_ylim(-5, 60)
ax.set_aspect('equal')
ax.axis('off')

# --- 6. LOAD MAP ---
print("Loading background map...")
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()
shapefile_name = "ne_110m_admin_0_countries.shp"
shp_path = os.path.join(script_dir, shapefile_name)
try:
    if os.path.exists(shp_path):
        world = gpd.read_file(shp_path)
        world.plot(ax=ax, color=LAND_COLOR, edgecolor=LAND_BORDER, linewidth=0.5, zorder=0)
    else:
        url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
        world = gpd.read_file(url)
        world.plot(ax=ax, color=LAND_COLOR, edgecolor=LAND_BORDER, linewidth=0.5, zorder=0)
except Exception as e:
    print(f"Map Error: {e}")

# --- 7. RENDER THE WEB ---
print("Rendering background...")
lines_x, lines_y = [], []
for u, v in G.edges():
    lines_x.extend([points[u, 0], points[v, 0], None])
    lines_y.extend([points[u, 1], points[v, 1], None])
ax.plot(lines_x, lines_y, color=WEB_COLOR, alpha=0.15, linewidth=0.6, zorder=1)

# --- 8. RENDER SMOOTHED ROUTES ---
print("Rendering curved routes...")

# Maritime (Blue) - Now curved
mx, my = get_curved_path(route_maritime)
ax.plot(mx, my, color=SEA_ROUTE_GLOW, alpha=0.1, linewidth=15, solid_capstyle='round', zorder=1) 
ax.plot(mx, my, color=SEA_ROUTE_GLOW, alpha=0.4, linewidth=4, solid_capstyle='round', zorder=2)  
ax.plot(mx, my, color='white', alpha=0.6, linewidth=1.5, linestyle=':', zorder=3)                

# Land (Red/Pink) - Plotting the pre-calculated smooth arrays
for sx, sy in smoothed_land_routes:
    # Glow Effects
    ax.plot(sx, sy, color=LAND_ROUTE_GLOW, alpha=0.1, linewidth=10, solid_capstyle='round', zorder=3)
    ax.plot(sx, sy, color=LAND_ROUTE_GLOW, alpha=0.5, linewidth=3, solid_capstyle='round', zorder=4)
    ax.plot(sx, sy, color='#FFCCDD', alpha=0.8, linewidth=1.0, zorder=5)

# --- 9. LABELS & NODES ---
# Collect all distinct nodes from raw data for plotting cities
unique_nodes = set()
for route in land_routes_raw + [route_maritime]:
    for node in route:
        unique_nodes.add(node)

# Plot all nodes as small glowing dots
for node in unique_nodes:
    ax.scatter(node[0], node[1], facecolor=NODE_COLOR, edgecolors=LAND_ROUTE_GLOW, linewidth=1, s=20, zorder=6)
    ax.scatter(node[0], node[1], color=LAND_ROUTE_GLOW, s=60, alpha=0.2, zorder=5)

#labels = {
    #"Xi'an": (108.9, 34.3), "Beijing": (116.4, 39.9), "Dunhuang": DUNHUANG,
    #"Kashgar": KASHGAR, "Samarkand": SAMARKAND, "Baghdad": BAGHDAD,
    #"Mecca": (39.8, 21.4), "Rome": (12.5, 41.9), "Singapore": (103.8, 1.35)
#}

#for name, coords in labels.items():
    #y_off = -2.5
    #if name in ["Singapore", "Rome", "Beijing"]: y_off = 2.5
    #ax.text(coords[0], coords[1]+y_off, name, color=TEXT_NEON, fontsize=10, fontweight='bold', ha='center', zorder=8)
    #ax.scatter([coords[0]], [coords[1]], color='white', edgecolor=LAND_ROUTE_GLOW, linewidth=2, s=80, zorder=9)

#plt.title("THE SILK ROAD: CURVED & CONNECTED NETWORK", color='white', fontsize=22, pad=20, fontname='Arial', fontweight='bold')
plt.tight_layout()
#output_file = "silk_road_curved_connected.png"
#plt.savefig(output_file, dpi=300, facecolor=BG_COLOR)
#print(f"Map successfully saved to: {output_file}")
plt.show()