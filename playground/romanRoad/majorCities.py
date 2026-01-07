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
    # GAUL & GERMANIA (North-West)
    "LUGDUNUM (Lyon)":        Point(4.8357, 45.7640),
    "BURDIGALA (Bordeaux)":   Point(-0.5792, 44.8378),
    "TREVERORUM (Trier)":     Point(6.6394, 49.7557),
    "COLONIA (Cologne)":      Point(6.9603, 50.9375),

    # HISPANIA (West)
    "OLISIPO (Lisbon)":       Point(-9.1393, 38.7223),
    "TARRACO (Tarragona)":    Point(1.2445, 41.1189),
    "CAESARAUGUSTA":          Point(-0.8877, 41.6488), # Zaragoza

    # ITALY & ALPINE (Center)
    "MEDIOLANUM (Milan)":     Point(9.1900, 45.4642),
    "GENUA (Genoa)":          Point(8.9463, 44.4056),
    "AQUILEIA":               Point(13.3710, 45.7700), # Gateway to Balkans
    "RHEGIUM":                Point(15.65, 38.11),     # Toe of Italy
    "BRUNDISIUM":             Point(17.9417, 40.6327), # Heel of Italy

    # BALKANS & GREECE (East-Central)
    "SALONA (Split)":         Point(16.4402, 43.5081),
    "THESSALONICA":           Point(22.9444, 40.6401),
    "ATHENAE (Athens)":       Point(23.7275, 37.9838),

    # ASIA MINOR & LEVANT (Far East)
    "BYZANTIUM":              Point(28.9784, 41.0082),
    "NICOMEDIA":              Point(29.9167, 40.7667), # Izmit
    "ANCYRA (Ankara)":        Point(32.8597, 39.9334),
    "ANTIOCHIA":              Point(36.1604, 36.2021)  # Antioch
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
f, ax = plt.subplots(1, 1, figsize=(30, 30))
f.patch.set_facecolor(sea_color)
ax.set_facecolor(sea_color)

# Land Mask
world.plot(ax=ax, color=land_color, edgecolor='none', zorder=0)

# Roads (Dimmed Background)
gdf.plot(column='dist_sqrt', ax=ax, cmap=neon_cmap, linewidth=0.9, alpha=0.5, zorder=2)

# Nodes (Dots)
nodes_gdf.plot(column='dist_sqrt', ax=ax, cmap=neon_cmap, markersize=2, alpha=0.9, zorder=4)

# HIGHLIGHTED PATHS
if not path_gdf.empty:
    # Outer Glow
    path_gdf.plot(ax=ax, color='cyan', linewidth=4, alpha=0.15, zorder=2)
    # Inner Glow
    path_gdf.plot(ax=ax, color='cyan', linewidth=1.5, alpha=0.6, zorder=3)
    # Core
    path_gdf.plot(ax=ax, color='cyan', linewidth=0.5, alpha=1.0, zorder=4)

    # --- START CAP (ROME) ---
    # Plot a large, glowing hub at the origin
    ax.plot(rome_projected.x, rome_projected.y, 
            marker='o',           # Circle for the hub center
            markersize=6,        # Large size
            color='yellow',        # White center
            markeredgecolor='cyan', # Cyan neon rim
            markeredgewidth=2,    # Thick rim
            alpha=0.9,
            zorder=5)
    # Add a prominent label for Rome
    #plt.text(rome_projected.x, rome_projected.y - 100000, "ROMA (Origin)", 
             #color='white', ha='center', fontsize=14, fontweight='bold', zorder=7)


    # --- END CAPS (DESTINATIONS) ---
    for idx, row in path_gdf.iterrows():
        # Find the very last point of the path
        last_geom = row.geometry.geoms[-1] if isinstance(row.geometry, MultiLineString) else row.geometry
        end_pt = last_geom.coords[-1]
        
        # Plot a glowing star at the destination
        ax.plot(end_pt[0], end_pt[1], 
                marker='o',           # Star for destination
                markersize=6, 
                color='white', 
                markeredgecolor='cyan', # Cyan neon rim
                markeredgewidth=2,
                alpha=0.9,
                zorder=4)

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

# Save the high-res file
plt.savefig("Roman_Network_4K.png", facecolor=sea_color, dpi=300, bbox_inches='tight')
print("High-resolution map saved as 'Roman_Network_4K.png'")

#plt.show()

plt.tight_layout()
plt.show()