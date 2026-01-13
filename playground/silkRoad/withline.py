import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from scipy.interpolate import splprep, splev
import os
import warnings
import urllib.request
import zipfile

# Suppress warnings
warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION ---
CURVE_RESOLUTION = 400  # High resolution for smooth thick lines

# --- MULTILINE & THICKNESS CONFIGURATION ---
MAX_STRANDS = 10        # Max parallel lines (Main Arteries)
MIN_STRANDS = 3         # Min parallel lines (Minor Paths)
RIBBON_WIDTH_BASE = 0.7 # Spread width

# NEW: Line Thickness Settings
MIN_LW = 0.5            # Thinnest line (for low traffic)
MAX_LW = 1.5            # Thickest line (for high traffic)

LINE_NOISE_SCALE = 0.08 # Wiggle amount

# --- COLOR PALETTE (Dual Tone) ---
BG_COLOR = '#002233'       
LAND_COLOR = '#1a1a1a'     
LAND_BORDER = '#232323'    

ROUTE_COLOR = '#00FFFF'     # Cyan
CITY_COLOR = 'cyan'      # Gold
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
        return np.array(lerp_x), np.array(lerp_y)
    try:
        tck, u = splprep([x, y], k=3, s=0.05) 
        u_new = np.linspace(u.min(), u.max(), resolution)
        x_smooth, y_smooth = splev(u_new, tck)
        return np.array(x_smooth), np.array(y_smooth)
    except:
        t = np.linspace(0, 1, resolution)
        lerp_x = np.interp(t, np.linspace(0, 1, len(x)), x)
        lerp_y = np.interp(t, np.linspace(0, 1, len(y)), y)
        return np.array(lerp_x), np.array(lerp_y)

# --- 3. DATA & ROUTES ---
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

routes_data = [
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
route_maritime = {"path": [(118.0, 24.5), SHANGHAI, (125.0, 28.0), (113.2, 23.1), (109.0, 13.0), (104.0, 1.3), (95.0, 5.5), (80.0, 6.0), (75.0, 10.0), (65.0, 15.0), (50.0, 13.0), (43.0, 12.5), MAKKAH, (35.0, 25.0), ALEXANDRIA], "score": 0.8}
all_routes = routes_data + [route_maritime]

# --- 4. PLOTTING SETUP ---
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(26, 14))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)
ax.set_xlim(0, 140)
ax.set_ylim(-5, 60)
ax.set_aspect('equal')
ax.axis('off')

# Map Loading
script_dir = os.getcwd()
shp_file = "ne_110m_admin_0_countries.shp"
shp_path = os.path.join(script_dir, shp_file)

if not os.path.exists(shp_path):
    try:
        print("Downloading map data...")
        url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
        zip_path = os.path.join(script_dir, "world_map.zip")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(script_dir)
    except Exception as e:
        print(f"Error: {e}")

try:
    world = gpd.read_file(shp_path)
    world.plot(ax=ax, color=LAND_COLOR, edgecolor=LAND_BORDER, linewidth=0.7, zorder=0)
except:
    pass

# --- 5. RENDER VARIABLE THICKNESS RIBBONS ---
print("Rendering trade flows with variable thickness...")

for r_data in all_routes:
    path = r_data['path']
    score = r_data['score']
    
    # 1. Get path
    sx, sy = get_curved_path(path)
    if len(sx) == 0: continue

    # 2. Calculate normals for ribbon width
    dx = np.gradient(sx)
    dy = np.gradient(sy)
    norms = np.hypot(dx, dy)
    norms[norms == 0] = 1 
    nx = -dy / norms
    ny = dx / norms

    # 3. Calculate DYNAMIC PROPERTIES based on score
    # A. Number of Strands (Density)
    num_strands = int(MIN_STRANDS + (MAX_STRANDS - MIN_STRANDS) * (score**1.2))
    
    # B. Line Thickness (Weight)
    # Using score to interpolate between MIN_LW and MAX_LW
    current_linewidth = MIN_LW + (MAX_LW - MIN_LW) * score
    
    # 4. Generate Strands
    offsets = np.linspace(-RIBBON_WIDTH_BASE, RIBBON_WIDTH_BASE, num_strands)
    
    for base_offset in offsets:
        noise = np.random.normal(0, LINE_NOISE_SCALE, size=len(sx))
        total_offset = base_offset + noise
        
        lx = sx + nx * total_offset
        ly = sy + ny * total_offset
        
        # High score = slightly higher opacity to make the thick lines shine
        line_alpha = 0.2 + (0.3 * score * np.random.rand())
        
        ax.plot(lx, ly, 
                color=ROUTE_COLOR, 
                linewidth=current_linewidth, # <--- DYNAMIC THICKNESS APPLIED HERE
                alpha=line_alpha, 
                zorder=3)


# --- 6. LABELS & CITIES ---
labels = {
    "Xi'an": (108.9, 34.3), "Dunhuang": DUNHUANG, "Samarkand": SAMARKAND, 
    "Baghdad": BAGHDAD, "Rome": (12.5, 41.9), "Singapore": (103.8, 1.35),
    "Constantinople": (28.9, 41.0), "Merv": MERV, "Rayy": (51.4, 35.7),
    "Taxila": (72.8, 33.7), "Khotan": (79.9, 37.1), "Shanghai": SHANGHAI,
    "Makkah": MAKKAH, "Alexandria": ALEXANDRIA, "Kyoto": KYOTO,
    "Seoul": SEOUL, "Delhi": DELHI
}

for name, coords in labels.items():
    y_off = -3.5
    if name in ["Singapore", "Constantinople", "Samarkand", "Rayy", "Kyoto", "Shanghai", "Seoul", "Taxila"]: y_off = 3.5
    if name == "Makkah": y_off = -4.5
    
    #ax.text(coords[0], coords[1]+y_off, name, color=TEXT_COLOR, fontsize=13, fontweight='bold', ha='center', zorder=10)
    ax.scatter([coords[0]], [coords[1]], color=CITY_COLOR, edgecolor='white', linewidth=1, s=90, zorder=9)

#plt.title("SILK ROAD: VOLUME-SCALED FLOWS", color=ROUTE_COLOR, fontsize=24, pad=20, fontname='Arial', fontweight='bold')
plt.tight_layout()

output_file = "silk_road_thickness.png"
#plt.savefig(output_file, dpi=300, facecolor=BG_COLOR)
print(f"✅ Map saved to: {output_file}")
plt.show()