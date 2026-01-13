import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from scipy.interpolate import splprep, splev
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION ---
SCATTER_WIDTH = 1.8   
CURVE_RESOLUTION = 400 

# --- COLOR PALETTE ---
BG_COLOR = '#002233'       
LAND_COLOR = '#050a14'     
LAND_BORDER = '#222222'    

# SPLIT COLORS
LAND_ROUTE_COLOR = '#FFE900'  # Neon Yellow for Silk Road (Hot)
SEA_ROUTE_COLOR = '#00FFFF'   # Cyan for Maritime Route (Cool)
CITY_COLOR = '#FFD700'      
TEXT_COLOR = '#FFFACD'      

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
        tck, u = splprep([x, y], k=3, s=0.1) 
        u_new = np.linspace(u.min(), u.max(), resolution)
        x_smooth, y_smooth = splev(u_new, tck)
        return x_smooth, y_smooth
    except:
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
MAKKAH = (39.8, 21.4)
ALEXANDRIA = (29.9, 31.2)
SHANGHAI = (121.5, 31.2)
KYOTO = (135.7, 35.0)
DELHI = (77.1, 28.7)
TAXILA = (72.8, 33.7)
SEOUL = (127.0, 37.5)

# Land Routes
routes_land = [
    {"path": [(108.9, 34.3), (103.8, 36.0), (98.2, 39.7), DUNHUANG], "score": 1.0},
    {"path": [DUNHUANG, (93.5, 42.8), (89.2, 42.9), (83.0, 41.7), KASHGAR], "score": 0.8},
    {"path": [DUNHUANG, (85.5, 38.1), (82.5, 37.5), (79.9, 37.1), (77.0, 37.5), KASHGAR], "score": 0.8},
    {"path": [KASHGAR, (70.0, 40.0), SAMARKAND, (63.5, 39.0), MERV], "score": 0.9},
    {"path": [MERV, (58.5, 37.9), (54.4, 36.4), (51.4, 35.7), (48.5, 34.8), BAGHDAD], "score": 0.9},
    {"path": [BAGHDAD, (40.0, 35.0), ANTIOCH_AREA, (35.5, 33.5), (34.0, 31.0)], "score": 0.9},
    {"path": [ANTIOCH_AREA, ALEXANDRIA], "score": 0.8},
    {"path": [ALEXANDRIA, (20.0, 35.0), (12.5, 41.9)], "score": 0.7}, 
    {"path": [ANTIOCH_AREA, (28.9, 41.0), (20.0, 42.0), (12.5, 41.9)], "score": 0.7},
    {"path": [SAMARKAND, (69.2, 34.5), TAXILA, DELHI, (82.0, 26.0), (88.0, 22.5)], "score": 0.7},
    {"path": [DELHI, (75.0, 22.0), (77.0, 15.0), (79.0, 11.0)], "score": 0.5},
    {"path": [(104.1, 30.7), (102.8, 24.9), (100.5, 19.9), (100.5, 13.7), (99.3, 9.1), (101.7, 3.1), (103.8, 1.35)], "score": 0.6},
    {"path": [(108.9, 34.3), (115.0, 32.0), SHANGHAI], "score": 0.7}, 
    {"path": [BAGHDAD, (44.0, 28.0), MAKKAH, (39.0, 15.0)], "score": 0.6}, 
    {"path": [(116.4, 39.9), (123.0, 40.0), SEOUL, (129.5, 35.5), KYOTO], "score": 0.6}, 
    {"path": [(116.4, 39.9), (106.9, 47.9), (88.0, 48.0), (76.9, 43.2), (60.0, 46.0), (48.0, 46.5), (39.0, 47.0), (30.5, 50.4), (26.1, 44.4)], "score": 0.3},
    {"path": [(76.9, 43.2), SAMARKAND], "score": 0.3},
    {"path": [BAGHDAD, (47.8, 30.5), (51.0, 26.0), (56.0, 21.0), (52.0, 15.0), (44.0, 13.0), (41.0, 18.0)], "score": 0.3},
]

# Maritime Route (List of 1 dictionary to make processing uniform)
routes_sea = [{"path": [(118.0, 24.5), SHANGHAI, (125.0, 28.0), (113.2, 23.1), (109.0, 13.0), (104.0, 1.3), (95.0, 5.5), (80.0, 6.0), (75.0, 10.0), (65.0, 15.0), (50.0, 13.0), (43.0, 12.5), MAKKAH, (35.0, 25.0), ALEXANDRIA], "score": 0.8}]

# --- 4. GENERATE DOTS FUNCTION ---
def generate_dots(route_list, base_dots=3000):
    dx, dy = [], []
    for r_data in route_list:
        path = r_data['path']
        score = r_data['score']
        smooth_x, smooth_y = get_curved_path(path)
        if len(smooth_x) == 0: continue
        
        num_dots = int(base_dots * (score ** 1.5))
        idx = np.random.randint(0, len(smooth_x), num_dots)
        center_x, center_y = smooth_x[idx], smooth_y[idx]
        
        spread = SCATTER_WIDTH * (1.1 - (score * 0.3))
        dot_noise_x = np.random.normal(0, spread, size=num_dots)
        dot_noise_y = np.random.normal(0, spread * 0.7, size=num_dots)
        dx.extend(center_x + dot_noise_x)
        dy.extend(center_y + dot_noise_y)
    return dx, dy

# Generate separate datasets
print("Generating Land Route Dots...")
land_x, land_y = generate_dots(routes_land, base_dots=3000)

print("Generating Sea Route Dots...")
sea_x, sea_y = generate_dots(routes_sea, base_dots=3500) # Higher count for sea to make it distinct

# --- 5. PLOTTING ---
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(26, 14))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)
ax.set_xlim(0, 140)
ax.set_ylim(-5, 60)
ax.set_aspect('equal')
ax.axis('off')

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except:
    script_dir = os.getcwd()
shp_path = os.path.join(script_dir, "ne_110m_admin_0_countries.shp")
if os.path.exists(shp_path):
    world = gpd.read_file(shp_path)
else:
    world = gpd.read_file("https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip")

world.plot(ax=ax, color=LAND_COLOR, edgecolor=LAND_BORDER, linewidth=0.7, zorder=0)

# --- 6. RENDER DATA (TWO LAYERS) ---
print("Rendering Land Routes (Yellow)...")
ax.scatter(land_x, land_y, 
           c=LAND_ROUTE_COLOR, 
           s=3, 
           alpha=0.3, 
           linewidth=0, 
           zorder=3)

print("Rendering Maritime Routes (Cyan)...")
ax.scatter(sea_x, sea_y, 
           c=SEA_ROUTE_COLOR, 
           s=3, 
           alpha=0.4, # Slightly higher alpha for cyan to pop against dark blue
           linewidth=0, 
           zorder=4)

# --- 7. LABELS & CITIES ---
labels = {
    "Xi'an": (108.9, 34.3), "Dunhuang": DUNHUANG, "Samarkand": SAMARKAND, 
    "Baghdad": BAGHDAD, "Rome": (12.5, 41.9), "Singapore": (103.8, 1.35),
    "Constantinople": (28.9, 41.0), "Merv": MERV, "Rayy": (51.4, 35.7),
    "Taxila": (72.8, 33.7), "Khotan": (79.9, 37.1), "Shanghai": SHANGHAI,
    "Makkah": MAKKAH, "Alexandria": ALEXANDRIA, "Kyoto": KYOTO,
    "Seoul": SEOUL, "Delhi": DELHI
}

for name, coords in labels.items():
    #ax.text(coords[0], coords[1]+y_off, name, color=TEXT_COLOR, fontsize=13, fontweight='bold', ha='center', zorder=10)
    ax.scatter([coords[0]], [coords[1]], color=CITY_COLOR, edgecolor='white', linewidth=1, s=90, zorder=9)

plt.tight_layout()
output_file = "playground/silkRoad/silk_road_dual_color.png"
plt.savefig(output_file, dpi=300, facecolor=BG_COLOR)
#print(f"✅ Map saved to: {output_file}")
plt.show()