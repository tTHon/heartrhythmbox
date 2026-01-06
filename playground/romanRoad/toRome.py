import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import contextily as cx

# 1. Load Data
gdf = gpd.read_file('playground/romanRoad/dataverse')

# 2. Project to Web Mercator (Meters)
# We need meters (EPSG:3857) to calculate accurate distances, not degrees.
gdf = gdf.to_crs(epsg=3857)

# 3. Define Rome's Location
# Rome coordinates: Lon 12.4964, Lat 41.9028
rome_geo = gpd.GeoSeries([Point(12.4964, 41.9028)], crs=4326)
rome_projected = rome_geo.to_crs(epsg=3857).iloc[0]

# 4. Calculate Distance to Rome for every road
# Returns distance in meters
gdf['dist_to_rome'] = gdf.distance(rome_projected)

# 5. Setup the Plot
f, ax = plt.subplots(1, 1, figsize=(15, 12))

# 6. Plot the Roads
# cmap='magma_r': 'magma' is purple-to-yellow. '_r' reverses it so Yellow (Hot) is 0 distance.
gdf.plot(
    column='dist_to_rome',
    ax=ax,
    cmap='magma_r',      # Bright center, dark edges
    linewidth=0.6,       # Keep lines thin for crisp detail
    alpha=0.8
)

# 7. Highlight Rome (The Node)
#ax.plot(
    #rome_projected.x, rome_projected.y, 
    #marker='*', color='white', markersize=15, zorder=5, label='Roma'
#)

# 5. Styling
ax.set_axis_off()
f.patch.set_facecolor('black')
ax.set_facecolor('black')

# 9. Styling
ax.set_axis_off()
plt.title("All Roads Lead to Rome (Proximity Heatmap)", fontsize=20, color='white')
plt.legend(loc='upper left')

# Optional: Zoom in slightly to focus on the Italian peninsula + surroundings
# (You can remove these lines to see the whole empire)
ax.set_xlim(rome_projected.x - 1000000, rome_projected.x + 1000000) 
ax.set_ylim(rome_projected.y - 1000000, rome_projected.y + 1000000)

plt.show()