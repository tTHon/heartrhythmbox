import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from shapely.geometry import Point
import numpy as np

# 1. LOAD DATA (ROADS)
gdf = gpd.read_file('playground/romanRoad/dataverse')
gdf = gdf.to_crs(epsg=3857) 

# 2. LOAD DATA (LAND / WORLD MAP)
# Geopandas includes a low-res world map we can use as a "mask"
# NEW (Works everywhere)
world = gpd.read_file("playground/romanRoad/dataverse/ne_110m_admin_0_countries.zip")
world = world.to_crs(epsg=3857)

# 3. CALCULATE DISTANCE (For styling)
rome_geo = gpd.GeoSeries([Point(12.4964, 41.9028)], crs=4326)
rome_projected = rome_geo.to_crs(epsg=3857).iloc[0]
gdf['dist'] = gdf.distance(rome_projected)
gdf['dist_sqrt'] = np.sqrt(gdf['dist'])

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


# 4. DEFINE COLORS
# Neon Road Palette
colors = [
    (1, 1, 1),    # White (Center)
    (1, 1, 0),    # Yellow
    (0, 1, 1),     # Cyan 
    (0, 0, 1),    # Blue
    (1, 0.5, 0),  # Orange
    (1, 0, 0),    # Red
    (0.5, 0, 0.5) # Purple
]
neon_cmap = mcolors.LinearSegmentedColormap.from_list("neon_smooth", colors)

# Sea and Land Colors
sea_color = '#060d24'  # A deep, midnight blue (Darker looks more premium)
land_color = 'none'   # Solid black to make the roads pop

# 5. PLOTTING
f, ax = plt.subplots(1, 1, figsize=(20, 12))

# --- STEP A: THE SEA (Background) ---
# We color the entire square canvas blue. 
f.patch.set_facecolor(sea_color)
ax.set_facecolor(sea_color)

# --- STEP B: THE LAND (Mask) ---
# We plot the land on top. Any "empty" space (sea) will show the blue background.
# zorder=0 ensures it sits at the very bottom.
world.plot(
    ax=ax, 
    color=land_color, 
    edgecolor='none', # No outlines for land, just a silhouette
    zorder=0
)

# --- STEP C: THE ROADS ---
# We plot the roads on top of the land.
# zorder=1 ensures it sits on top of the land.
gdf.plot(
    column='dist_sqrt',
    ax=ax,
    cmap=neon_cmap,
    linewidth=0.7,
    alpha=0.5,
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

# 6. FOCUS VIEW
# Since we loaded the "Whole World" for the land mask, we MUST zoom in 
# to the Roman Empire, otherwise the map will show Antarctica too.
minx, miny, maxx, maxy = gdf.total_bounds
padding = 100000 # Add a little padding (200km) around the edges
ax.set_xlim(minx - padding, maxx + padding)
ax.set_ylim(miny - padding, maxy + padding)

# Label
#plt.text(0.5, 0.05, "IMPERIUM ROMANUM", transform=ax.transAxes, 
         #ha='center', color='white', fontsize=24, fontname='Century Gothic', alpha=0.8)

ax.set_axis_off()
plt.tight_layout()
plt.show()