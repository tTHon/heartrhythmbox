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

# --- 1. CONFIGURATION (HIGH VISIBILITY MODE) ---
NUM_POINTS = 25000    # Optimal density for structure without becoming a blob
CONNECTIVITY = 5      # 5 connections per node creates a "mesh" look
SCATTER_WIDTH = 3.5   # Increased: spreads the web further out from main roads
CURVE_RESOLUTION = 300 

# Cyberpunk Neon Palette
BG_COLOR = '#020202'       # Pure Black
LAND_COLOR = '#111111'     # Very Dark Grey
LAND_BORDER = '#333333'    

# VISIBILITY UPDATES: Brighter colors, higher contrast
WEB_COLOR = '#7B00FF'      # Electric Violet (Much brighter than before)
LAND_ROUTE_GLOW = '#FF0055' # Neon Pink
SEA_ROUTE_GLOW = '#00F2FF'  # Neon Cyan
NODE_COLOR = '#FFFFFF'     
TEXT_NEON = '#FFFFFF'      

# --- 2. HELPER FUNCTION FOR CURVES ---
def get_curved_path(coords, resolution=CURVE_RESOLUTION):
    if len(coords) < 2: return [], []
    x = [p[0] for p in coords]
    y = [p[1] for p in coords]
    
    # Linear fallback for short segments
    if len(coords) <= 3:
        t = np.linspace(0, 1, resolution)
        lerp_x = np.interp(t, np.linspace(0, 1, len(x)), x)
        lerp_y = np.interp(t, np.linspace(0, 1, len(y)), y)
        return lerp_x, lerp_y

    # Cubic Spline
    try:
        tck, u = splprep([x, y], k=3, s=0) 
        u_new = np.linspace(u.min(), u.max(), resolution)
        x_smooth, y_smooth = splev(u_new, tck)
        return x_smooth, y_smooth
    except:
        # Fallback
        t = np.linspace(0, 1, resolution)
        lerp_x = np.interp(t, np.linspace(0, 1, len(x)), x)
        lerp_y = np.interp(t, np.linspace(0, 1, len(y)), y)
        return lerp_x, lerp_y

# --- 3. DEFINE ROUTES ---
DUNHUANG = (94.7, 40.1)
KASHGAR = (75.9, 39.5)
SAMARKAND = (66.9, 39.6)
MERV = (62.2, 37.6)
BAGHDAD = (44.4, 33.3)
ANTIOCH_AREA = (36.1, 36.2)

# Land
route_corridor = [(108.9, 34.3), (103.8, 36.0), (98.2, 39.7), DUNHUANG] 
route_tarim_north = [DUNHUANG, (93.5, 42.8), (89.2, 42.9), (83.0, 41.7), KASHGAR] 
route_tarim_south = [DUNHUANG, (85.5, 38.1), (82.5, 37.5), (79.9, 37.1), (77.0, 37.5), KASHGAR] 
route_central = [KASHGAR, (70.0, 40.0), SAMARKAND, (63.5, 39.0), MERV] 
route_persia = [MERV, (58.5, 37.9), (54.4, 36.4), (51.4, 35.7), (48.5, 34.8), BAGHDAD] 
route_levant = [BAGHDAD, (40.0, 35.0), ANTIOCH_AREA, (35.5, 33.5), (34.0, 31.0)] 
route_steppe = [(116.4, 39.9), (106.9, 47.9), (88.0, 48.0), (76.9, 43.2), (60.0, 46.0), (48.0, 46.5), (39.0, 47.0), (30.5, 50.4), (26.1, 44.4)]
route_steppe_connector = [(76.9, 43.2), SAMARKAND]
route_arabia = [BAGHDAD, (47.8, 30.5), (51.0, 26.0), (56.0, 21.0), (52.0, 15.0), (44.0, 13.0), (41.0, 18.0), (39.0, 21.5), (35.0, 28.0), (35.5, 31.0)]
route_chengdu_singapore = [(104.1, 30.7), (102.8, 24.9), (100.5, 19.9), (100.5, 13.7), (99.3, 9.1), (101.7, 3.1), (103.8, 1.35)]
route_india_north = [SAMARKAND, (69.2, 34.5), (73.0, 33.7), (77.2, 28.6), (82.0, 26.0), (88.0, 22.5)]
route_india_south = [(77.2, 28.6), (75.0, 22.0), (77.0, 15.0), (79.0, 11.0)]
route_korea = [(116.4, 39.9), (122.0, 40.5), (125.7, 39.0), (127.0, 37.5)]
route_europe_main = [ANTIOCH_AREA, (28.9, 41.0), (20.0, 42.0), (12.5, 41.9)]
route_europe_north = [(28.9, 41.0), (19.0, 47.0), (13.0, 48.0), (2.3, 48.8)]

# Maritime
route_maritime = [(118.0, 24.5), (113.2, 23.1), (109.0, 13.0), (104.0, 1.3), (95.0, 5.5), (80.0, 6.0), (75.0, 10.0), (65.0, 15.0), (50.0, 13.0), (43.0, 12.5), (35.0, 25.0), (32.5, 29.9)]

land_routes_raw = [route_corridor, route_tarim_north, route_tarim_south, route_central, route_persia, route_levant, route_steppe, route_steppe_connector, route_arabia, route_chengdu_singapore, route_india_north, route_india_south, route_korea, route_europe_main, route_europe_north]

# --- 4. GENERATE POINTS ---
print("Generating cloud points...")
all_x, all_y = [], []
smoothed_land_routes = []

for route in land_routes_raw:
    smooth_x, smooth_y = get_curved_path(route)
    smoothed_land_routes.append((smooth_x, smooth_y))
    
    # Generate vascular noise points AROUND the route
    # We use 4 layers of noise with increasing spread to create a gradient "fuzz"
    for width_multiplier in [0.5, 1.5, 3.0]:
        noise_x = np.random.normal(0, SCATTER_WIDTH * width_multiplier, size=len(smooth_x))
        noise_y = np.random.normal(0, SCATTER_WIDTH * 0.7 * width_multiplier, size=len(smooth_y))
        all_x.extend(smooth_x + noise_x)
        all_y.extend(smooth_y + noise_y)

# Add Maritime to web too
mx, my = get_curved_path(route_maritime)
for width_multiplier in [1.0, 2.5]:
    noise_mx = np.random.normal(0, SCATTER_WIDTH * width_multiplier, size=len(mx))
    noise_my = np.random.normal(0, SCATTER_WIDTH * 0.7 * width_multiplier, size=len(my))
    all_x.extend(mx + noise_mx)
    all_y.extend(my + noise_my)

points = np.column_stack((all_x, all_y))

# Downsample if too massive
if len(points) > NUM_POINTS:
    idx = np.random.choice(len(points), NUM_POINTS, replace=False)
    points = points[idx]

print(f"Building vascular connections for {len(points)} nodes...")
tree = cKDTree(points)
# k=CONNECTIVITY means each dot connects to 5 neighbors
dist_matrix, neighbors = tree.query(points, k=CONNECTIVITY+1)

G = nx.Graph()
for i in range(len(points)):
    for j in range(1, CONNECTIVITY+1):
        target = neighbors[i][j]
        G.add_edge(i, target, weight=dist_matrix[i][j])

# --- 5. PLOTTING ---
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(26, 14))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)
ax.set_xlim(0, 135)
ax.set_ylim(-5, 60)
ax.set_aspect('equal')
ax.axis('off')

# Load Map
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except:
    script_dir = os.getcwd()
shp_path = os.path.join(script_dir, "ne_110m_admin_0_countries.shp")

if os.path.exists(shp_path):
    world = gpd.read_file(shp_path)
else:
    world = gpd.read_file("https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip")

world.plot(ax=ax, color=LAND_COLOR, edgecolor=LAND_BORDER, linewidth=0.5, zorder=0)

# --- 6. RENDER LAYERS ---

# A. THE VASCULAR WEB
print("Rendering vascular web...")
# We extract edge coordinates in bulk for faster plotting
lines_x_fine, lines_y_fine = [], []

for u, v in G.edges():
    lines_x_fine.extend([points[u, 0], points[v, 0], None])
    lines_y_fine.extend([points[u, 1], points[v, 1], None])

# Draw the web with HIGH VISIBILITY
# Alpha=0.25 makes it clearly visible. linewidth=0.6 makes lines distinct.
ax.plot(lines_x_fine, lines_y_fine, color=WEB_COLOR, alpha=0.25, linewidth=0.6, zorder=1)

# B. MAIN ARTERIES (GLOW)
print("Rendering main arteries...")
# Maritime
ax.plot(mx, my, color=SEA_ROUTE_GLOW, alpha=0.15, linewidth=16, solid_capstyle='round', zorder=2)
ax.plot(mx, my, color=SEA_ROUTE_GLOW, alpha=0.8, linewidth=3, solid_capstyle='round', zorder=3)
ax.plot(mx, my, color='white', alpha=0.6, linewidth=1, linestyle=':', zorder=4)

# Land
for sx, sy in smoothed_land_routes:
    # Outer Glow (Wide)
    ax.plot(sx, sy, color=LAND_ROUTE_GLOW, alpha=0.15, linewidth=14, solid_capstyle='round', zorder=2)
    # Inner Glow (Medium)
    ax.plot(sx, sy, color=LAND_ROUTE_GLOW, alpha=0.6, linewidth=4, solid_capstyle='round', zorder=3)
    # Core (Thin/White)
    ax.plot(sx, sy, color='#FFDDDD', alpha=0.9, linewidth=1, zorder=4)

# C. CITIES
labels = {
    "Xi'an": (108.9, 34.3), "Dunhuang": DUNHUANG, "Samarkand": SAMARKAND, 
    "Baghdad": BAGHDAD, "Rome": (12.5, 41.9), "Singapore": (103.8, 1.35)
}

for name, coords in labels.items():
    y_off = -3 if name != "Singapore" else 3
    ax.text(coords[0], coords[1]+y_off, name, color=TEXT_NEON, fontsize=12, fontweight='bold', ha='center', zorder=8)
    ax.scatter([coords[0]], [coords[1]], color='white', edgecolor=LAND_ROUTE_GLOW, linewidth=2, s=100, zorder=9)

plt.title("SILK ROAD: VASCULAR WEB", color='white', fontsize=24, pad=20, fontname='Arial', fontweight='bold')
plt.tight_layout()

output_file = "silk_road_visible_web.png"
#plt.savefig(output_file, dpi=300, facecolor=BG_COLOR)
print(f"✅ Map saved to: {output_file}")
plt.show()