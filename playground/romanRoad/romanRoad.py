import geopandas as gpd # version: 0.9.0
import matplotlib.pyplot as plt # version: 3.7.1

gdf = gpd.read_file('playground/romanRoad/dataverse')
gdf = gdf.to_crs(4326)
print(len(gdf))
gdf.head(3)

f, ax = plt.subplots(1,1,figsize=(15,10))
ax.set_facecolor('black')
gdf.plot(column = 'CERTAINTY', ax=ax)
plt.show()