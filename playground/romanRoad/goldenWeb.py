import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from shapely.geometry import Point, MultiLineString, LineString
import numpy as np
import networkx as nx

# --- 1. LOAD DATA ---
gdf = gpd.read_file('playground/romanRoad/dataverse')
gdf = gdf.to_crs(epsg=3857) 

world = gpd.read_file("playground/romanRoad/dataverse/ne_110m_admin_0_countries.zip")
world = world.to_crs(epsg=3857)

# Rome (Center)
rome_geo = gpd.GeoSeries([Point(12.4964, 41.9028)], crs=4326)
rome_projected = rome_geo.to_crs(epsg=3857).iloc[0]

# --- 2. PREPARE GRAPH ---
print("Building Road Network Graph...")
G = nx.Graph()

for idx, row in gdf.iterrows():
    geom = row.geometry
    parts = geom.geoms if isinstance(geom, MultiLineString) else [geom]
    for part in parts:
        start_node = (round(part.coords[0][0], -2), round(part.coords[0][1], -2))
        end_node   = (round(part.coords[-1][0], -2), round(part.coords[-1][1], -2))
        G.add_edge(start_node, end_node, weight=part.length, geometry=part)

def get_node(point, graph):
    p = (point.x, point.y)
    return min(graph.nodes, key=lambda n: (n[0]-p[0])**2 + (n[1]-p[1])**2)

# --- 3. DEFINE CITIES ---
rome_node = get_node(rome_projected, G)
cities = {
    "LUGDUNUM":     Point(4.8357, 45.7640),
    "BURDIGALA":    Point(-0.5792, 44.8378),
    "TREVERORUM":   Point(6.6394, 49.7557),
    "COLONIA":      Point(6.9603, 50.9375),
    "OLISIPO":      Point(-9.1393, 38.7223),
    "TARRACO":      Point(1.2445, 41.1189),
    "CAESARAUGUSTA": Point(-0.8877, 41.6488),
    "MEDIOLANUM":   Point(9.1900, 45.4642),
    "GENUA":        Point(8.9463, 44.4056),
    "AQUILEIA":     Point(13.3710, 45.7700),
    "RHEGIUM":      Point(15.65, 38.11),
    "BRUNDISIUM":   Point(17.9417, 40.6327),
    "SALONA":       Point(16.4402, 43.5081),
    "THESSALONICA": Point(22.9444, 40.6401),
    "ATHENAE":      Point(23.7275, 37.9838),
    "BYZANTIUM":    Point(28.9784, 41.0082),
    "NICOMEDIA":    Point(29.9167, 40.7667),
    "ANCYRA":       Point(32.8597, 39.9334),
    "ANTIOCHIA":    Point(36.1604, 36.2021)
}

# --- 4. CALCULATE PATHS ---
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
            edge_geom = data['geometry'] if 'geometry' in data else data[0]['geometry']
            lines.append(edge_geom)
        full_line = MultiLineString(lines)
        paths_data.append({'name': name, 'geometry': full_line})
    except nx.NetworkXNoPath:
        pass
path_gdf = gpd.GeoDataFrame(paths_data, crs=gdf.crs)

# --- 5. VISUALIZATION SETUP ---
# Create nodes for visualization
nodes = gdf.geometry.boundary.explode(index_parts=True)
nodes_gdf = gpd.GeoDataFrame(geometry=nodes, crs=gdf.crs).drop_duplicates()

# --- COLOR DEFINITIONS ---
neon_yellow = '#FFE900'  # The primary electric yellow
white_hot   = '#FFFFE0'  # Almost white for the "hot" centers
#sea_color   = '#050a14'  # Very dark blue/black
sea_color   = '#3e404d'  # Deep navy blue
land_color  = "#1C2D3B"

# --- 6. PLOT ---
f, ax = plt.subplots(1, 1, figsize=(30, 30))
f.patch.set_facecolor(sea_color)
ax.set_facecolor(sea_color)

# Land Mask
world.plot(ax=ax, color=land_color, edgecolor='none', zorder=0)

# A. BASE ROADS (The faint background network)
# We use the single neon yellow but with low opacity (alpha)
gdf.plot(
    ax=ax, 
    color=neon_yellow, 
    linewidth=0.6, 
    alpha=0.5, # Very faint to let the paths pop
    zorder=2
)

# B. NODES (The intersection dots)
nodes_gdf.plot(
    ax=ax, 
    color=neon_yellow, 
    markersize=1.2, 
    alpha=0.6, 
    zorder=3
)

# C. HIGHLIGHTED PATHS (The active routes)
if not path_gdf.empty:
    # 1. Broad Glow (Yellow)
    path_gdf.plot(ax=ax, color=neon_yellow, linewidth=4, alpha=0.15, zorder=2)
    # 2. Medium Beam (Yellow)
    path_gdf.plot(ax=ax, color=neon_yellow, linewidth=2.5, alpha=0.6, zorder=3)
    # 3. White-Hot Core (To show intensity)
    path_gdf.plot(ax=ax, color=white_hot, linewidth=0.8, alpha=0.9, zorder=4)

    # --- START CAP (ROME) ---
    ax.plot(rome_projected.x, rome_projected.y, 
            marker='o', markersize=10, 
            color= sea_color,          
            markeredgecolor=neon_yellow, 
            markeredgewidth=1,
            alpha=1.0, zorder=6)

    # --- END CAPS (DESTINATIONS) ---
    for idx, row in path_gdf.iterrows():
        last_geom = row.geometry.geoms[-1] if isinstance(row.geometry, MultiLineString) else row.geometry
        end_pt = last_geom.coords[-1]
        
        # Plot glowing hub at destination
        ax.plot(end_pt[0], end_pt[1], 
                marker='o', markersize=8, 
                color=neon_yellow, 
                markeredgecolor=white_hot, 
                markeredgewidth=1,
                alpha=0.88, zorder=5)

# Focus
minx, miny, maxx, maxy = gdf.total_bounds
padding = 100000 
ax.set_xlim(minx - padding, maxx + padding)
ax.set_ylim(miny - padding, maxy + padding)
ax.set_axis_off()

# Save
plt.savefig("playground/romanRoad/Roman_Network_Yellow.svg", facecolor=sea_color, dpi=300, bbox_inches='tight')
#print("Saved 'Roman_Network_Yellow.png'")

plt.show()