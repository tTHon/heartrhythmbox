import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from shapely.geometry import Point
import numpy as np

# --- 1. SETUP & DATA LOADING ---
gdf = gpd.read_file('playground/romanRoad/dataverse')
gdf = gdf.to_crs(epsg=3857) # Project to Web Mercator (Meters)

# Define Rome (Center)
rome_geo = gpd.GeoSeries([Point(12.4964, 41.9028)], crs=4326)
rome_projected = rome_geo.to_crs(epsg=3857).iloc[0]

# --- 2. PREPARE ROADS (EDGES) ---
# Calculate distance for the lines
gdf['dist'] = gdf.distance(rome_projected)

# --- OPTION B: Logarithmic Scale (Recommended for Maps) ---
# This spreads the colors out better. We add +1 to avoid log(0) errors.
#gdf['dist_log'] = np.log1p(gdf['dist_to_rome'])

# Calculate raw distance in meters
gdf['dist_raw'] = gdf.distance(rome_projected)

# --- THE FIX: SQUARE ROOT NORMALIZATION ---
# Using Square Root spreads the "center" colors out further than Linear or Log.
# It makes the "glow" larger and the fade smoother.
gdf['dist_sqrt'] = np.sqrt(gdf['dist_raw'])

# 3. CALCULATE LINE WIDTH (INVERSE DISTANCE)
# Step A: Normalize distance between 0.0 and 1.0
max_dist = gdf['dist'].max()
gdf['dist_norm'] = gdf['dist'] / max_dist

# Step B: Invert it (Close = 1.0, Far = 0.0)
gdf['inverse_dist'] = 1 - gdf['dist_norm']

# Step C: Scale to actual pixel sizes
# Max thickness = 3.0 (Rome), Min thickness = 0.1 (Scotland/Syria)
min_width = 0.1
max_width_boost = 2.5
linewidths = min_width + (gdf['inverse_dist']**2 * max_width_boost)


# --- 3. PREPARE NODES (INTERSECTIONS) ---
# Extract start/end points from lines
nodes = gdf.geometry.boundary.explode(index_parts=True)

# Convert to a GeoDataFrame so we can calculate distances
nodes_gdf = gpd.GeoDataFrame(geometry=nodes, crs=gdf.crs)

# Important: Remove duplicates (shared intersections) to prevent "blobbing"
nodes_gdf = nodes_gdf.drop_duplicates()

# Calculate distance for the nodes (so they match the road color at that location)
nodes_gdf['dist_raw'] = nodes_gdf.distance(rome_projected)
nodes_gdf['dist_sqrt'] = np.sqrt(nodes_gdf['dist_raw'])

# --- 4. DEFINE NEON PALETTE ---
colors = [
    (1, 1, 1),    # White (Center)
    (1, 1, 0),    # Yellow
    (0, 1, 1),     # Cyan 
    (0, 0, 1),    # Blue
    (1, 0.5, 0),  # Orange
    (1, 0, 0),    # Red
    (0.5, 0, 0.5) # Purple
]
neon_cmap = mcolors.LinearSegmentedColormap.from_list("neon_rainbow", colors)

# --- 5. PLOTTING ---
f, ax = plt.subplots(1, 1, figsize=(15, 12))
f.patch.set_facecolor('black')
ax.set_facecolor('black')

# Layer 1: The Roads
gdf.plot(
    column='dist_sqrt',  # Use square root distance for better color spread
    ax=ax,
    cmap=neon_cmap,
    linewidth=0.5,
    alpha=0.5,    # Slightly transparent so nodes pop more
    zorder=1
)

# Layer 2: The Nodes
nodes_gdf.plot(
    column='dist_sqrt', # Color by same distance metric
    ax=ax,
    cmap=neon_cmap,        # Use same palette
    markersize=1,        # Use the same linewidths for nodes
    alpha=0.9,             # Solid opacity for nodes
    zorder=2
)

# Optional: Add a subtle glow layer underneath (Blur effect simulation)
# We plot the same lines again, thicker and transparent, to look like "light"
#gdf.plot(
    #ax=ax,
    #color='red',           # A generic glow color
    #linewidth=linewidths*3,# Much thicker
    #alpha=0.1,             # Very transparent
    #zorder=1
#)


# Focus View (Zoom in to see the node detail)
#zoom = 800000 
#ax.set_xlim(rome_projected.x - zoom, rome_projected.x + zoom)
#ax.set_ylim(rome_projected.y - zoom, rome_projected.y + zoom)
ax.set_axis_off()

plt.tight_layout()
plt.show()