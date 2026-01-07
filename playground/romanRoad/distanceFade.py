import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from shapely.geometry import Point, MultiLineString
import numpy as np
import networkx as nx

# --- 1. CONFIGURATION ---
FADE_STEEPNESS = 2.5   # Higher = Fades faster (Darker edges). Try 2.0 to 5.0.
MIN_ALPHA = 0.3       # Minimum visibility for farthest roads (0.0 = invisible)
NEON_YELLOW = '#FFE900'
SEA_COLOR = '#3e404d'
LAND_COLOR = "#1C2D3B"

# --- 2. LOAD DATA ---
gdf = gpd.read_file('playground/romanRoad/dataverse')
gdf = gdf.to_crs(epsg=3857) 

world = gpd.read_file("playground/romanRoad/dataverse/ne_110m_admin_0_countries.zip")
world = world.to_crs(epsg=3857)

rome_geo = gpd.GeoSeries([Point(12.4964, 41.9028)], crs=4326)
rome_projected = rome_geo.to_crs(epsg=3857).iloc[0]

# --- 3. PREPARE GRAPH ---
print("Building Graph...")
G = nx.Graph()
for idx, row in gdf.iterrows():
    geom = row.geometry
    parts = geom.geoms if isinstance(geom, MultiLineString) else [geom]
    for part in parts:
        start = (round(part.coords[0][0], -2), round(part.coords[0][1], -2))
        end   = (round(part.coords[-1][0], -2), round(part.coords[-1][1], -2))
        G.add_edge(start, end, weight=part.length, geometry=part)

def get_node(point, graph):
    p = (point.x, point.y)
    return min(graph.nodes, key=lambda n: (n[0]-p[0])**2 + (n[1]-p[1])**2)

# --- 4. CITIES & PATHS ---
rome_node = get_node(rome_projected, G)
cities = {
    "LUGDUNUM": Point(4.8357, 45.7640), "BURDIGALA": Point(-0.5792, 44.8378),
    "TREVERORUM": Point(6.6394, 49.7557), "COLONIA": Point(6.9603, 50.9375),
    "OLISIPO": Point(-9.1393, 38.7223), "TARRACO": Point(1.2445, 41.1189),
    "CAESARAUGUSTA": Point(-0.8877, 41.6488), "MEDIOLANUM": Point(9.1900, 45.4642),
    "GENUA": Point(8.9463, 44.4056), "AQUILEIA": Point(13.3710, 45.7700),
    "RHEGIUM": Point(15.65, 38.11), "BRUNDISIUM": Point(17.9417, 40.6327),
    "SALONA": Point(16.4402, 43.5081), "THESSALONICA": Point(22.9444, 40.6401),
    "ATHENAE": Point(23.7275, 37.9838), "BYZANTIUM": Point(28.9784, 41.0082),
    "NICOMEDIA": Point(29.9167, 40.7667), "ANCYRA": Point(32.8597, 39.9334),
    "ANTIOCHIA": Point(36.1604, 36.2021)
}

paths_data = []
for name, pt in cities.items():
    try:
        target_proj = gpd.GeoSeries([pt], crs=4326).to_crs(epsg=3857).iloc[0]
        target_node = get_node(target_proj, G)
        route_nodes = nx.shortest_path(G, rome_node, target_node, weight='weight')
        lines = []
        for i in range(len(route_nodes)-1):
            u, v = route_nodes[i], route_nodes[i+1]
            data = G.get_edge_data(u, v)
            geom = data['geometry'] if 'geometry' in data else data[0]['geometry']
            lines.append(geom)
        paths_data.append({'name': name, 'geometry': MultiLineString(lines)})
    except nx.NetworkXNoPath: pass
path_gdf = gpd.GeoDataFrame(paths_data, crs=gdf.crs)

# --- 5. CREATE SHARP FADE GRADIENT ---
# 1. Calculate distances
gdf['dist_from_rome'] = gdf.distance(rome_projected)
nodes = gdf.geometry.boundary.explode(index_parts=True)
nodes_gdf = gpd.GeoDataFrame(geometry=nodes, crs=gdf.crs).drop_duplicates()
nodes_gdf['dist_from_rome'] = nodes_gdf.distance(rome_projected)

# 2. Build Custom 'Sharp Decay' Colormap
N = 256
vals = np.ones((N, 4))
# Color: Yellow
vals[:, 0] = 1.0; vals[:, 1] = 0.91; vals[:, 2] = 0.0

# Alpha Curve: Exponential Decay
# We map 0..1 (Distance) to Alpha
t = np.linspace(0, 1, N)
# Formula: (1 - t)^Power. Power > 1 makes it drop faster.
alpha_curve = np.power((1.0 - t), FADE_STEEPNESS)
# Clip to minimum so outer roads don't vanish completely
vals[:, 3] = np.clip(alpha_curve, MIN_ALPHA, 1.0)

sharp_fade_cmap = mcolors.ListedColormap(vals)

# --- 6. PLOT ---
f, ax = plt.subplots(1, 1, figsize=(30, 30))
f.patch.set_facecolor(SEA_COLOR)
ax.set_facecolor(SEA_COLOR)

# Land
world.plot(ax=ax, color=LAND_COLOR, edgecolor='none', zorder=0)

# A. BASE ROADS (Uses the Sharp Fade)
gdf.plot(column='dist_from_rome', ax=ax, cmap=sharp_fade_cmap, linewidth=0.6, zorder=2)
nodes_gdf.plot(column='dist_from_rome', ax=ax, cmap=sharp_fade_cmap, markersize=1.2, alpha=0.6, zorder=3)

# B. HIGHLIGHTS (Stay Bright)
if not path_gdf.empty:
    path_gdf.plot(ax=ax, color=NEON_YELLOW, linewidth=4, alpha=0.15, zorder=2)
    path_gdf.plot(ax=ax, color=NEON_YELLOW, linewidth=2.5, alpha=0.6, zorder=3)
    path_gdf.plot(ax=ax, color='#FFFFE0', linewidth=0.8, alpha=0.9, zorder=4)

    # Rome
    ax.plot(rome_projected.x, rome_projected.y, marker='o', markersize=10, 
            color=SEA_COLOR, markeredgecolor=NEON_YELLOW, markeredgewidth=1.5, zorder=6)

    # Destinations
    for idx, row in path_gdf.iterrows():
        last_geom = row.geometry.geoms[-1] if isinstance(row.geometry, MultiLineString) else row.geometry
        end_pt = last_geom.coords[-1]
        ax.plot(end_pt[0], end_pt[1], marker='o', markersize=8, 
                color=NEON_YELLOW, markeredgecolor='white', markeredgewidth=1, alpha=0.88, zorder=5)

# Crop
minx, miny, maxx, maxy = gdf.total_bounds
padding = 100000 
ax.set_xlim(minx - padding, maxx + padding)
ax.set_ylim(miny - padding, maxy + padding)
ax.set_axis_off()

plt.savefig("playground/romanRoad/Roman_Network_gradient.png", facecolor=SEA_COLOR, dpi=300, bbox_inches='tight')
print("Saved sharp fade map.")
plt.show()