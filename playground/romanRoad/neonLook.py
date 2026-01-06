import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx  # This library adds the background map

# 1. Load Data (Using your fixed path)
gdf = gpd.read_file('playground/romanRoad/dataverse')

# 2. PROJECTION IS KEY: Convert to Web Mercator (EPSG:3857)
# Most background maps (Google Maps, OpenStreetMap) use this projection.
gdf_web = gdf.to_crs(epsg=3857)

# 3. Setup the plot
f, ax = plt.subplots(1, 1, figsize=(15, 10))

# 4. Plot the data
gdf_web.plot(
    column='CERTAINTY', 
    ax=ax,
    cmap='plasma',      # 'plasma', 'magma', or 'inferno' look great on dark
    linewidth=0.8,      # Thinner lines look more precise
    alpha=0.8,          # Slight transparency handles overlapping roads better
    legend=False,
    #legend_kwds={'shrink': 0.5} # Make legend smaller
)

# 5. Add the "Dark Look" basemap

# set both figure and axes to black background so tiles and plots sit on dark canvas
f.patch.set_facecolor('black')
ax.set_facecolor('black')

# 6. Clean up: Remove the box and numbers
ax.set_axis_off() 
plt.title("Network of Roman Roads", fontsize=20, color='white')

# Adjust layout to remove whitespace
plt.tight_layout()
plt.show()