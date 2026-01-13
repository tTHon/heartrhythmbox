import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, LineString
import os

# --- 1. SETUP COMPLEX DATA ---
cities_data = {
    "Chang'an (Xi'an)": [108.9398, 34.3416],
    "Luoyang": [112.4536, 34.6197],
    "Lanzhou": [103.8343, 36.0611],
    "Dunhuang": [94.6620, 40.1421],
    "Turpan": [89.1722, 42.9513],
    "Kucha": [82.9622, 41.7173],
    "Kashgar": [75.9938, 39.4677],
    "Niya": [82.7466, 37.0666],
    "Khotan": [79.9222, 37.1142],
    "Almaty": [76.9122, 43.2220],
    "Samarkand": [66.9750, 39.6270],
    "Bukhara": [64.4556, 39.7686],
    "Merv": [62.1670, 37.6611],
    "Bactra (Balkh)": [66.8972, 36.7563],
    "Taxila": [72.8021, 33.7458],
    "Barbarikon (Port)": [67.6652, 24.5928],
    "Mashhad (Tus)": [59.6057, 36.2605],
    "Tehran (Rayy)": [51.3890, 35.6892],
    "Ecbatana (Hamadan)": [48.5134, 34.7987],
    "Persepolis": [52.8923, 29.9361],
    "Baghdad": [44.3661, 33.3152],
    "Palmyra": [38.2672, 34.5601],
    "Damascus": [36.2765, 33.5138],
    "Antioch": [36.1606, 36.2021],
    "Tyre": [35.1962, 33.2705],
    "Constantinople": [28.9784, 41.0082],
    "Ephesus": [27.3431, 37.9388],
    "Rome": [12.4964, 41.9028],
    "Alexandria": [29.9187, 31.2001],
    "Berenike": [35.5333, 23.9333],
    "Muscat": [58.4059, 23.5859],
}

route_pairs = [
    ("Luoyang", "Chang'an (Xi'an)"), ("Chang'an (Xi'an)", "Lanzhou"), ("Lanzhou", "Dunhuang"),
    ("Dunhuang", "Turpan"), ("Turpan", "Kucha"), ("Kucha", "Kashgar"),
    ("Dunhuang", "Niya"), ("Niya", "Khotan"), ("Khotan", "Kashgar"),
    ("Kashgar", "Samarkand"), ("Kashgar", "Bactra (Balkh)"), ("Samarkand", "Bukhara"),
    ("Bukhara", "Merv"), ("Bactra (Balkh)", "Merv"), ("Turpan", "Almaty"), ("Almaty", "Samarkand"),
    ("Bactra (Balkh)", "Taxila"), ("Taxila", "Barbarikon (Port)"),
    ("Merv", "Mashhad (Tus)"), ("Mashhad (Tus)", "Tehran (Rayy)"), ("Tehran (Rayy)", "Ecbatana (Hamadan)"),
    ("Ecbatana (Hamadan)", "Baghdad"), ("Tehran (Rayy)", "Persepolis"),
    ("Baghdad", "Palmyra"), ("Palmyra", "Damascus"), ("Palmyra", "Antioch"),
    ("Damascus", "Tyre"), ("Antioch", "Constantinople"), ("Antioch", "Ephesus"),
    ("Ephesus", "Rome"), ("Tyre", "Alexandria"), ("Alexandria", "Berenike"),
    ("Barbarikon (Port)", "Muscat"), ("Muscat", "Persepolis"), ("Muscat", "Berenike"),
]

# --- 2. PREPARE GEODATAFRAMES ---
city_points = [Point(coords[0], coords[1]) for coords in cities_data.values()]
gdf_cities = gpd.GeoDataFrame({'city': list(cities_data.keys())}, geometry=city_points, crs="EPSG:4326")

lines = []
for start, end in route_pairs:
    if start in cities_data and end in cities_data:
        p1, p2 = Point(cities_data[start]), Point(cities_data[end])
        lines.append(LineString([p1, p2]))
gdf_routes = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")

# --- 3. PLOTTING SETUP ---
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(20, 12))
fig.patch.set_facecolor('#050505')
ax.set_facecolor('#0a0a0a')

ax.set_xlim(-10, 130)
ax.set_ylim(15, 55)
ax.set_aspect('equal')
ax.axis('off')

# --- FIX: READ .SHP DIRECTLY ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# UPDATE: Point this to the .shp file you see in your folder.
# Standard Natural Earth name is "ne_110m_admin_0_countries.shp"
shapefile_name = "ne_110m_admin_0_countries.shp"
shp_path = os.path.join(script_dir, shapefile_name)

print(f"Looking for map at: {shp_path}")

if os.path.exists(shp_path):
    world = gpd.read_file(shp_path)
    world.plot(
        ax=ax,
        color='#1a1a1a', 
        edgecolor='#2a2a2a', 
        linewidth=0.5, 
        zorder=1
    )
else:
    print(f"ERROR: Could not find '{shapefile_name}'")
    print("Please check the folder. Do you see a file ending in .shp?")
    # Fallback: List files to help debug
    print(f"Files found in {script_dir}:")
    print(os.listdir(script_dir))
    exit()

# --- 4. RENDERING THE "GLOW" ---
# Layer 1
gdf_routes.plot(ax=ax, color='#FF8C00', linewidth=8, alpha=0.15, zorder=2)
# Layer 2
gdf_routes.plot(ax=ax, color='#FFD700', linewidth=4, alpha=0.4, zorder=3)
# Layer 3
gdf_routes.plot(ax=ax, color='#FFFFFF', linewidth=1.2, alpha=0.9, zorder=4)

# --- 5. PLOT CITIES ---
gdf_cities.plot(ax=ax, color='#FFFFFF', markersize=15, alpha=1, zorder=5, edgecolor='#FFD700', linewidth=2)

major_hubs = ["Chang'an (Xi'an)", "Samarkand", "Baghdad", "Constantinople", "Rome"]
for x, y, label in zip(gdf_cities.geometry.x, gdf_cities.geometry.y, gdf_cities['city']):
    if label in major_hubs:
        plt.text(x, y+0.8, label, color='#FFD700', fontsize=10, ha='center', fontweight='bold', zorder=6)

plt.title("THE COMPLEX SILK ROAD NETWORK", color='#FFD700', fontsize=22, pad=20)
plt.tight_layout()

# --- 6. EXPORT ---
output_png = os.path.join(script_dir, "silk_road_complex.png")
output_svg = os.path.join(script_dir, "silk_road_complex.svg")

print(f"Exporting PNG to: {output_png}")
plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())

print(f"Exporting SVG to: {output_svg}")
plt.savefig(output_svg, format='svg', bbox_inches='tight', facecolor=fig.get_facecolor())

print("Done!")