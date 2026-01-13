import folium

# 1. Define key Silk Road cities with coordinates (Lat, Lon)
cities = {
    "Xi'an (Chang'an)": [34.3416, 108.9398],
    "Dunhuang": [40.1421, 94.6620],
    "Kashgar": [39.4677, 75.9938],
    "Samarkand": [39.6270, 66.9750],
    "Tehran (Rayy)": [35.6892, 51.3890],
    "Baghdad": [33.3152, 44.3661],
    "Antioch": [36.2021, 36.1606],
    "Constantinople (Istanbul)": [41.0082, 28.9784]
}

# 2. Create a map centered roughly on the route
m = folium.Map(location=[38.0, 70.0], zoom_start=4, tiles="CartoDB dark_matter")

# 3. Add markers for each city
route_coords = []

for name, coords in cities.items():
    folium.Marker(
        location=coords,
        popup=name,
        icon=folium.Icon(color="beige", icon="info-sign")
    ).add_to(m)
    route_coords.append(coords)

# 4. Draw the trade route line connecting them
folium.PolyLine(
    route_coords,
    color="gold",
    weight=2.5,
    opacity=0.8,
    dash_array='10'
).add_to(m)

# 5. Save the map
output_file = "playground/silkRoad/silk_road_map.html"
m.save(output_file)
print(f"Map created! Open '{output_file}' in your browser to see the Silk Road.")