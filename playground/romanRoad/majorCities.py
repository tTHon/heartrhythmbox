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

# --- 2. PREPARE GRAPH (WITH SNAPPING) ---
print("Building Road Network Graph...")
G = nx.Graph()

for idx, row in gdf.iterrows():
    geom = row.geometry
    parts = geom.geoms if isinstance(geom, MultiLineString) else [geom]
    for part in parts:
        # Snap to nearest 100m to fix small road gaps
        start_node = (round(part.coords[0][0], -2), round(part.coords[0][1], -2))
        end_node   = (round(part.coords[-1][0], -2), round(part.coords[-1][1], -2))
        G.add_edge(start_node, end_node, weight=part.length, geometry=part)

# Helper: Find nearest graph node
def get_node(point, graph):
    p = (point.x, point.y)
    return min(graph.nodes, key=lambda n: (n[0]-p[0])**2 + (n[1]-p[1])**2)

# --- 3. DEFINE CITIES (Land Routes Only) ---
rome_node = get_node(rome_projected, G)

cities = {
    "LUGDUNUM (Gaul)": Point(4.8357, 45.7640),   # Lyon - Capital of Gaul
    "BYZANTIUM":       Point(28.9784, 41.0082),  # East
    "OLISIPO":         Point(-9.1393, 38.7223),  # West
    "RHEGIUM":         Point(15.65, 38.11)       # South (Tip of Italy)
}

# --- 4. CALCULATE PATHS ---
paths_data = []
for name, pt in cities.items():
    try:
        target_proj = gpd.GeoSeries([pt], crs=4326).to_crs(epsg=3857).iloc[0]
        target_node = get_node(target_proj, G)
        
        # Calculate Shortest Path
        route_nodes = nx.shortest_path(G, rome_node, target_node, weight='weight')
        
        # Reconstruct Geometry
        lines = []
        for i in range(len(route_nodes)-1):
            u, v = route_nodes[i], route_nodes[i+1]
            data = G.get_edge_data(u, v)
            # Handle multiple edges
            edge_geom = data['geometry'] if 'geometry' in data else data[0]['geometry']
            lines.append(edge_geom)
            
        full_line = MultiLineString(lines)
        paths_data.append({'name': name, 'geometry': full_line})
        print(f"Route found: Rome -> {name}")
        
    except nx.NetworkXNoPath:
        print(f"No path to {name}")

path_gdf = gpd.GeoDataFrame(paths_data, crs=gdf.crs)

# --- 5. VISUALIZATION SETUP ---
# Calculate distances for styling
gdf['dist_sqrt'] = np.sqrt(gdf.distance(rome_projected))

# Nodes (Intersections)
nodes = gdf.geometry.boundary.explode(index_parts=True)
nodes_gdf = gpd.GeoDataFrame(geometry=nodes, crs=gdf.crs).drop_duplicates()
nodes_gdf['dist_sqrt'] = np.sqrt(nodes_gdf.distance(rome_projected))

# Colors
colors = [(1, 1, 1), (1, 1, 0), (0, 1, 1), (0, 0, 1), (1, 0.5, 0), (1, 0, 0), (0.5, 0, 0.5)]
neon_cmap = mcolors.LinearSegmentedColormap.from_list("neon_smooth", colors)

sea_color = '#060d24'
land_color = 'black'

# --- 6. PLOT ---
f, ax = plt.subplots(1, 1, figsize=(20, 12))
f.patch.set_facecolor(sea_color)
ax.set_facecolor(sea_color)

# Land Mask
world.plot(ax=ax, color=land_color, edgecolor='none', zorder=0)

# Roads (Dimmed Background)
gdf.plot(column='dist_sqrt', ax=ax, cmap=neon_cmap, linewidth=0.7, alpha=0.5, zorder=2)

# Nodes (Dots)
nodes_gdf.plot(column='dist_sqrt', ax=ax, cmap=neon_cmap, markersize=1.5, alpha=0.9, zorder=4)

# HIGHLIGHTED PATHS
if not path_gdf.empty:
    # Outer Glow
    path_gdf.plot(ax=ax, color='white', linewidth=3, alpha=0.2, zorder=2)
    # Inner Glow
    path_gdf.plot(ax=ax, color='cyan', linewidth=3, alpha=0.5, zorder=2)
    # Core
    path_gdf.plot(ax=ax, color='white', linewidth=1, alpha=1.0, zorder=3)

    # Labels
    #for idx, row in path_gdf.iterrows():
        #last_geom = row.geometry.geoms[-1] if isinstance(row.geometry, MultiLineString) else row.geometry
        #end_pt = last_geom.coords[-1]
        #ax.plot(end_pt[0], end_pt[1], marker='*', color='white', markersize=15, zorder=6)
        #plt.text(end_pt[0], end_pt[1] + 60000, row['name'], color='white', ha='center', fontsize=12, fontweight='bold', zorder=7)

# Focus
minx, miny, maxx, maxy = gdf.total_bounds
padding = 100000 
ax.set_xlim(minx - padding, maxx + padding)
ax.set_ylim(miny - padding, maxy + padding)
ax.set_axis_off()

plt.tight_layout()
plt.show()