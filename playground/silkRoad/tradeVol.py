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
# Total points for the background web (kept distinct from route dots)
NUM_WEB_POINTS = 15000    
CONNECTIVITY = 5      
SCATTER_WIDTH = 1.8   
CURVE_RESOLUTION = 300 # smoother curves for better dot placement


BG_COLOR = "#002233"        # Very Dark Blue
LAND_COLOR = '#050a14'     
LAND_BORDER = '#232323'    

# --- COLOR PALETTE (Neon Yellow Theme) ---
#BG_COLOR = '#050a14'       
#LAND_COLOR = '#252525'     
#LAND_BORDER = '#404040'



# --- OPTION 1: HIGH CONTRAST (Electric Blue & Yellow) ---
DOT_COLOR = '#00FFFF'      # Neon Yellow
WEB_COLOR = '#002233'      # Deep Midnight Blue (Darker than before to let yellow pop)

NODE_COLOR = '#FFFFFF'     # Pure White (Keep this for city markers)
TEXT_NEON = '#FFFACD'      # Lemon Chiffon (Subtle yellow tint for text)


# --- 2. HELPER FUNCTIONS ---
def get_curved_path(coords, resolution=CURVE_RESOLUTION):
    if len(coords) < 2: return [], []
    x = [p[0] for p in coords]
    y = [p[1] for p in coords]
    
    if len(coords) <= 3:
        t = np.linspace(0, 1, resolution)
        lerp_x = np.interp(t, np.linspace(0, 1, len(x)), x)
        lerp_y = np.interp(t, np.linspace(0, 1, len(y)), y)
        return lerp_x, lerp_y

    try:
        # s=0.1 allows slight smoothing so dots flow better
        tck, u = splprep([x, y], k=3, s=0.1) 
        u_new = np.linspace(u.min(), u.max(), resolution)
        x_smooth, y_smooth = splev(u_new, tck)
        return x_smooth, y_smooth
    except:
        t = np.linspace(0, 1, resolution)
        lerp_x = np.interp(t, np.linspace(0, 1, len(x)), x)
        lerp_y = np.interp(t, np.linspace(0, 1, len(y)), y)
        return lerp_x, lerp_y

# --- 3. DEFINE ROUTES & TRAFFIC INTENSITY ---
DUNHUANG = (94.7, 40.1)
KASHGAR = (75.9, 39.5)
SAMARKAND = (66.9, 39.6)
MERV = (62.2, 37.6)
BAGHDAD = (44.4, 33.3)
ANTIOCH_AREA = (36.1, 36.2)
MAKKAH = (39.8, 21.4)
ALEXANDRIA = (29.9, 31.2)
SHANGHAI = (121.5, 31.2)
KYOTO = (135.7, 35.0)
DELHI = (77.1, 28.7)
TAXILA = (72.8, 33.7)
SEOUL = (127.0, 37.5)

# Score determines dot density multiplier
routes_data = [
    # MAIN TRUNK
    {"path": [(108.9, 34.3), (103.8, 36.0), (98.2, 39.7), DUNHUANG], "score": 1.0},
    {"path": [DUNHUANG, (93.5, 42.8), (89.2, 42.9), (83.0, 41.7), KASHGAR], "score": 0.8},
    {"path": [DUNHUANG, (85.5, 38.1), (82.5, 37.5), (79.9, 37.1), (77.0, 37.5), KASHGAR], "score": 0.8},
    {"path": [KASHGAR, (70.0, 40.0), SAMARKAND, (63.5, 39.0), MERV], "score": 0.9},
    {"path": [MERV, (58.5, 37.9), (54.4, 36.4), (51.4, 35.7), (48.5, 34.8), BAGHDAD], "score": 0.9},
    {"path": [BAGHDAD, (40.0, 35.0), ANTIOCH_AREA, (35.5, 33.5), (34.0, 31.0)], "score": 0.9},
    
    # BRANCHES
    {"path": [ANTIOCH_AREA, ALEXANDRIA], "score": 0.8},
    {"path": [ALEXANDRIA, (20.0, 35.0), (12.5, 41.9)], "score": 0.7}, 
    {"path": [ANTIOCH_AREA, (28.9, 41.0), (20.0, 42.0), (12.5, 41.9)], "score": 0.7},
    
    # India Route (Updated with Delhi)
    {"path": [SAMARKAND, (69.2, 34.5), TAXILA, DELHI, (82.0, 26.0), (88.0, 22.5)], "score": 0.7},
    {"path": [DELHI, (75.0, 22.0), (77.0, 15.0), (79.0, 11.0)], "score": 0.5},
    
    {"path": [(104.1, 30.7), (102.8, 24.9), (100.5, 19.9), (100.5, 13.7), (99.3, 9.1), (101.7, 3.1), (103.8, 1.35)], "score": 0.6},
    
    # China/Korea/Japan Extensions (Updated with Seoul)
    {"path": [(108.9, 34.3), (115.0, 32.0), SHANGHAI], "score": 0.7}, 
    {"path": [BAGHDAD, (44.0, 28.0), MAKKAH, (39.0, 15.0)], "score": 0.6}, 
    # Route through Seoul to Kyoto
    {"path": [(116.4, 39.9), (123.0, 40.0), SEOUL, (129.5, 35.5), KYOTO], "score": 0.6}, 

    # STEPPE & LOW DENSITY
    {"path": [(116.4, 39.9), (106.9, 47.9), (88.0, 48.0), (76.9, 43.2), (60.0, 46.0), (48.0, 46.5), (39.0, 47.0), (30.5, 50.4), (26.1, 44.4)], "score": 0.3},
    {"path": [(76.9, 43.2), SAMARKAND], "score": 0.3},
    {"path": [BAGHDAD, (47.8, 30.5), (51.0, 26.0), (56.0, 21.0), (52.0, 15.0), (44.0, 13.0), (41.0, 18.0)], "score": 0.3},
]
route_maritime = {"path": [(118.0, 24.5), SHANGHAI, (125.0, 28.0), (113.2, 23.1), (109.0, 13.0), (104.0, 1.3), (95.0, 5.5), (80.0, 6.0), (75.0, 10.0), (65.0, 15.0), (50.0, 13.0), (43.0, 12.5), MAKKAH, (35.0, 25.0), ALEXANDRIA], "score": 0.8}
all_routes = routes_data + [route_maritime]
# --- 4. GENERATE DATA DOTS & BACKGROUND WEB ---
print("Generating density point clouds...")

# Lists for the main data visualization
route_dots_x = []
route_dots_y = []

# Lists for the background network web
web_x = []
web_y = []

all_routes = routes_data + [route_maritime]

BASE_DOTS_PER_ROUTE = 3000 # Baseline number of dots before density scaling

for r_data in all_routes:
    path = r_data['path']
    score = r_data['score']
    smooth_x, smooth_y = get_curved_path(path)

    # --- A. Generate Background Web Points ---
    # Add points to the background web generator (uniform density for the web)
    web_noise_x = np.random.normal(0, SCATTER_WIDTH * 2, size=len(smooth_x))
    web_noise_y = np.random.normal(0, SCATTER_WIDTH * 1.5, size=len(smooth_y))
    web_x.extend(smooth_x + web_noise_x)
    web_y.extend(smooth_y + web_noise_y)

    # --- B. Generate Trade Density Dots ---
    # Calculate number of dots based on score.
    # Using score^1.5 emphasizes the difference between high and low density.
    num_dots = int(BASE_DOTS_PER_ROUTE * (score ** 1.5))
    
    # Pick random centers along the smoothed path
    idx = np.random.randint(0, len(smooth_x), num_dots)
    center_x = smooth_x[idx]
    center_y = smooth_y[idx]
    
    # Scatter width tightens slightly for higher density routes for a crisper look
    spread = SCATTER_WIDTH * (1.1 - (score * 0.3))
    
    dot_noise_x = np.random.normal(0, spread, size=num_dots)
    dot_noise_y = np.random.normal(0, spread * 0.7, size=num_dots)
    
    route_dots_x.extend(center_x + dot_noise_x)
    route_dots_y.extend(center_y + dot_noise_y)


# Build Background Web Graph
web_points = np.column_stack((web_x, web_y))
if len(web_points) > NUM_WEB_POINTS:
    idx = np.random.choice(len(web_points), NUM_WEB_POINTS, replace=False)
    web_points = web_points[idx]

print(f"Building background web connecting {len(web_points)} nodes...")
tree = cKDTree(web_points)
dist_matrix, neighbors = tree.query(web_points, k=CONNECTIVITY+1)

G = nx.Graph()
for i in range(len(web_points)):
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

# Map
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

# --- 6. RENDER BACKGROUND WEB (Very Faint) ---
#print("Rendering background web...")
#lines_x, lines_y = [], []
#for u, v in G.edges():
    #lines_x.extend([web_points[u, 0], web_points[v, 0], None])
    #lines_y.extend([web_points[u, 1], web_points[v, 1], None])

# Uniform faint color for background structure
#ax.plot(lines_x, lines_y, color=WEB_COLOR, alpha=0.1, linewidth=0.5, zorder=1)

# --- 7. RENDER TRADE DENSITY DOTS ---
print(f"Rendering {len(route_dots_x)} data points...")

# The core visualization: A single scatter plot of all generated dots.
# s=2: Very small dots.
# alpha=0.2: High transparency. Overlapping dots create brighter areas.
ax.scatter(route_dots_x, route_dots_y,
           c=DOT_COLOR,
           s=3, 
           alpha=0.3, 
           linewidth=0, # No borders on dots for smoother cloud look
           zorder=3)

# Add a very faint glow blob layer for atmosphere
ax.scatter(route_dots_x[::10], route_dots_y[::10], # Plot fewer for glow
           c=DOT_COLOR, s=50, alpha=0.03, linewidth=0, zorder=2)


# --- 8. LABELS & LEGEND ---
labels = {
    "Xi'an": (108.9, 34.3), "Dunhuang": DUNHUANG, "Samarkand": SAMARKAND, 
    "Baghdad": BAGHDAD, "Rome": (12.5, 41.9), "Singapore": (103.8, 1.35),
    "Constantinople": (28.9, 41.0),
    "Merv": MERV,
    "Rayy": (51.4, 35.7),
    "Taxila": (72.8, 33.7),
    "Khotan": (79.9, 37.1),"Shanghai": SHANGHAI,
    "Makkah": MAKKAH,
    "Alexandria": ALEXANDRIA,
    "Kyoto": KYOTO,
    "Seoul": SEOUL,
    "Delhi": DELHI
}

for name, coords in labels.items():
    y_off = -3 if name != "Singapore" else 3
    #ax.text(coords[0], coords[1]+y_off, name, color=TEXT_NEON, fontsize=12, fontweight='bold', ha='center', zorder=8)
    # White city markers to pop against the cyan dots
    ax.scatter([coords[0]], [coords[1]], color='#FFFACD', edgecolor=DOT_COLOR, linewidth=1, s=15, zorder=9)

# Legend explaining the visual style
#plt.text(5, 5, "TRADE VOLUME DENSITY", color=DOT_COLOR, fontsize=14, fontweight='bold', alpha=0.9)
#plt.text(5, 2, "Brighter Areas = Higher Dot Density = Higher Trade Volume", color=TEXT_NEON, fontsize=11, alpha=0.7)

#plt.title("SILK ROAD: TRADE DENSITY POINT CLOUD", color=TEXT_NEON, fontsize=24, pad=20, fontname='Arial', fontweight='bold')
plt.tight_layout()

output_file = "playground/silkRoad/silk_road_dot_density.png"
plt.savefig(output_file, dpi=300, facecolor=BG_COLOR)
#print(f"✅ Map saved to: {output_file}")
plt.show()