import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx
from shapely.geometry import Point

# 1. Load Data
gdf_edges = gpd.read_file('playground/romanRoad/dataverse')

# 2. Project to Web Mercator (Required for contextily background)
gdf_edges = gdf_edges.to_crs(epsg=4326) 

# 3. Extract Nodes (The Start/End points of lines)
# 'boundary' gets the endpoints, 'explode' separates them into individual rows
nodes = gdf_edges.geometry.boundary.explode(index_parts=True)

# Optional: Remove duplicate points (intersections share coordinates)
# This makes the rendering faster and the alpha blending more accurate
nodes_unique = nodes.drop_duplicates()

# Rome coordinates: Lon 12.4964, Lat 41.9028
rome_geo = gpd.GeoSeries([Point(12.4964, 41.9028)], crs=4326)
rome_projected = rome_geo.to_crs(epsg=3857).iloc[0]

# 4. Setup the Plot
f, ax = plt.subplots(1, 1, figsize=(15, 10))

# --- LAYER 1: The Roads (Edges) ---
gdf_edges.plot(
    ax=ax,
    linewidth=0.5,
    color='#4a4a4a', # Dark grey for roads (subtle)
    alpha=0.9,        # Slight transparency
    zorder=1         # Draw this first (bottom layer)
)

# --- LAYER 2: The Cities (Nodes) ---
# We use the extracted 'nodes' series here
nodes_unique.plot(
    ax=ax,
    color='#ff00ff',  # Cyan/Neon Blue or Magenta
    markersize=1,     # Small dots look more like a "data cloud"
    alpha=0.8,        # Slight transparency
    zorder=2          # Draw on top of lines
)

# --- HIGHLIGHT: Roads leading to Rome ---
# project edges to WebMercator for metric distances, find nearby edges, then plot them
gdf_edges_3857 = gdf_edges.to_crs(epsg=3857)
distances = gdf_edges_3857.distance(rome_projected)
threshold_m = 1000000  # 1000 km  
near_edges_idx = distances[distances <= threshold_m].index
if len(near_edges_idx) == 0:
    # fallback: pick the single nearest edge
    near_edges_idx = [distances.idxmin()]
gdf_to_rome = gdf_edges.loc[near_edges_idx]
gdf_to_rome.plot(
    ax=ax,
    color='cyan',
    linewidth=0.8,
    alpha=0.6,
    zorder=3
)

# 5. Styling
ax.set_axis_off()
f.patch.set_facecolor('black')
ax.set_facecolor('black')



plt.tight_layout()
plt.show()