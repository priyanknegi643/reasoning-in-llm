import folium
import pickle
import argparse
import osmnx as ox
import networkx as nx
import os
import numpy as np
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from PIL import Image


def create_curved_edge(lat1, lon1, lat2, lon2, direction, num_points=50):
    # Calculate the control point for the quadratic Bezier curve
    delta_lat = 0
    delta_lon = 0
    if "N" in direction:
        delta_lat = 0.002
    elif "S" in direction:
        delta_lat = -0.002
    if "E" in direction:
        delta_lon = 0.002
    elif "W" in direction:
        delta_lon = -0.002

    control_lat = lat1 + delta_lat
    control_lon = lon1 + delta_lon

    # Generate points along the Bezier curve
    t = np.linspace(0, 1, num_points)
    lat_curve = (1 - t) ** 2 * lat1 + 2 * (1 - t) * t * control_lat + t**2 * lat2
    lon_curve = (1 - t) ** 2 * lon1 + 2 * (1 - t) * t * control_lon + t**2 * lon2

    # Create a list of coordinate tuples for the curved edge
    edge_coordinates = list(zip(lat_curve, lon_curve))
    return edge_coordinates


# def white_bg_tile(x, y, z):
# return 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAQAAAAEAAQMAAABmvDolAAAAA1BMVEUAAACnej3aAAAAAXRSTlMAQObYZgAAACJJREFUaIHtwTEBAAAAwqD1T20ND6AAAAAAAAAAAAAA4N8AKvgAAUFIrrEAAAAASUVORK5CYII='


def make_map(graph, lat_long_to_intersection_id, far):
    # Create a Folium map centered on Midtown
    tile = "OpenStreetMap"  # Use OpenStreetMap tiles instead of blank canvas
    # tile = folium.raster_layers.TileLayer(opacity=0.2)
    if far:
        lat_range = (40.702, 40.800)
        long_range = (-74.022, -73.933)
        zoom = 13
        weight = 0.75
        radius = 1
        cmap_new = ["black", "black"]
        cmap_true = ["black", "black"]
        alpha = 0.3
        # tile = None
    else:
        lat_range = (40.702, 40.800)
        long_range = (-74.022, -73.933)
        zoom = 13
        weight = 1.5
        radius = 2
        cmap_new = ["lightsalmon", "firebrick"]
        cmap_true = ["black", "black"]
        alpha = 0.7
        # tile = None

    center = (sum(lat_range) / 2, sum(long_range) / 2)
    map_nyc = folium.Map(
        location=center, zoom_start=zoom, tiles=None, zoom_control=False
    )

    # Add OpenStreetMap tiles with reduced opacity
    folium.TileLayer(
        tiles="OpenStreetMap",
        opacity=0.3,  # Reduced opacity so black lines are visible
        name="Street Map",
    ).add_to(map_nyc)

    # Add edges as lines to the map with labels
    for u, v, k, data in graph.edges(keys=True, data=True):
        lat1, long1 = intersection_id_to_lat_long[u]
        lat2, long2 = intersection_id_to_lat_long[v]
        etype = data["edge_type"]
        direction = data["direction"]
        if etype == "true_unused":
            continue
        elif etype == "true":
            edge_locations = [[lat1, long1], [lat2, long2]]
            edge_line = folium.ColorLine(
                positions=edge_locations,
                weight=1,
                colors=np.arange(len(edge_locations) - 1),
                colormap=cmap_true,
                alpha=0.3,
            )
        else:
            curved_edge_locations = create_curved_edge(
                lat1, long1, lat2, long2, direction, num_points=50
            )
            edge_line = folium.ColorLine(
                positions=curved_edge_locations,
                weight=weight,
                colors=np.arange(len(curved_edge_locations) - 1),
                colormap=cmap_new,
                alpha=alpha,
            )

        edge_line.add_to(map_nyc)
    return map_nyc


def take_map_screenshot(
    html_file_path, output_png_path, size=(1000, 1000), crop_margin=50
):
    """Take a screenshot of a folium map HTML file and crop off bottom-right watermark."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(f"--window-size={size[0]},{size[1]}")

    driver = webdriver.Chrome(options=chrome_options)

    try:
        driver.get(f"file://{os.path.abspath(html_file_path)}")
        time.sleep(3)

        temp_png_path = output_png_path.replace(".png", "_temp.png")
        driver.save_screenshot(temp_png_path)

        with Image.open(temp_png_path) as img:
            width, height = img.size

            # Remove 'crop_margin' pixels from bottom and right to remove watermark
            left = 0
            top = 0
            right = width - crop_margin
            bottom = height - crop_margin

            # Perform the crop
            cropped_img = img.crop((left, top, right, bottom))

            # (Optional) ensure square crop from center
            min_dim = min(cropped_img.size)
            w, h = cropped_img.size
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            cropped_img = cropped_img.crop((left, top, left + min_dim, top + min_dim))

            cropped_img.save(output_png_path)

        os.remove(temp_png_path)
        print(f"Screenshot saved to: {output_png_path}")

    finally:
        driver.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--screenshot", action="store_true", help="Take screenshots of the maps"
    )
    args = parser.parse_args()
    place_name = "Manhattan, New York City, New York, USA"
    historical_date = datetime(2024, 5, 5, 0, 0, 0)
    ox.settings.overpass_settings = (
        f'[out:json][timeout:180][date:"{historical_date.isoformat()}Z"]'
    )
    ox_graph = ox.graph_from_place(place_name, network_type="drive")
    nodes = list(ox_graph.nodes())

    intersection_id_to_lat_long = {}
    for node in nodes:
        intersection_id_to_lat_long[node] = (
            ox_graph.nodes[node]["y"],
            ox_graph.nodes[node]["x"],
        )
    lat_long_to_intersection_id = {v: k for k, v in intersection_id_to_lat_long.items()}

    graph_names = [x for x in os.listdir("graphs/") if x[0] != "."]

    # Create the map directory if it doesn't exist
    os.makedirs("maps", exist_ok=True)

    for graph_name in graph_names:
        with open("graphs/" + graph_name, "rb") as f:
            graph = pickle.load(f)
        print(graph_name)
        print(graph)
        far_map = make_map(graph, lat_long_to_intersection_id, True)
        close_map = make_map(graph, lat_long_to_intersection_id, False)

        # Save HTML files
        far_html_path = "maps/far_{}.html".format(graph_name[:-4])
        close_html_path = "maps/close_{}.html".format(graph_name[:-4])
        far_map.save(far_html_path)
        close_map.save(close_html_path)

        # Take screenshots
        far_png_path = "maps/far_{}.png".format(graph_name[:-4])
        close_png_path = "maps/close_{}.png".format(graph_name[:-4])

        if args.screenshot:
            print(f"Taking screenshot of {far_html_path}...")
            take_map_screenshot(far_html_path, far_png_path, size=(1000, 1000))

            print(f"Taking screenshot of {close_html_path}...")
            take_map_screenshot(close_html_path, close_png_path, size=(1000, 1000))
