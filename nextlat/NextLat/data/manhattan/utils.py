import networkx as nx
from datetime import datetime
import math
import random
from collections import defaultdict

import torch
import numpy as np
import osmnx as ox
from datetime import datetime
from sklearn.cluster import KMeans


def distance_lat_long_to_miles(latlong1, latlong2):
    lat1 = latlong1[0]
    lon1 = latlong1[1]
    lat2 = latlong2[0]
    lon2 = latlong2[1]
    # Radius of the Earth in miles
    radius = 3958.8

    # Convert degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = radius * c

    return distance


# def validate_turns_sequence(
#     node_and_direction_to_neighbor, valid_turns, sequence, verbose=True
# ):
#     source, dest = sequence[:2]
#     directions = sequence[2:]
#     cur_node = source
#     for i, direction in enumerate(directions):
#         if direction in valid_turns[cur_node]:
#             cur_node = node_and_direction_to_neighbor[(cur_node, direction)]
#         else:
#             if verbose:
#                 print(
#                     "Path Failure (turns): ", source, dest, i, directions[i], directions
#                 )
#             return False
#     if cur_node != dest:
#         if verbose:
#             print("Dest Failure (turns)", source, dest, directions)
#         return False
#     return True


def validate_graph_sequences(graph, sequences, verbose=True):
    successes = []
    path_failures = []
    dest_failures = []
    for sequence in sequences:
        source, dest = sequence[:2]
        directions = sequence[2:]
        cur_node = source
        failure = False
        for i, direction in enumerate(directions):
            next_node = -1
            for u, v, k, d in graph.out_edges(cur_node, data="direction", keys=True):
                if d == direction:
                    next_node = v
                    break
            if next_node == -1:
                path_failures.append(sequence)
                failure = True
                break
            cur_node = next_node
        if failure:
            continue
        if cur_node != dest:
            dest_failures.append(sequence)
        else:
            successes.append(sequence)
    if verbose:
        print("# Sequences:", len(sequences))
        print(
            "# Successes: {} ({:.1f}%)".format(
                len(successes), len(successes) / len(sequences) * 100
            )
        )
        print(
            "# Path Failures: {} ({:.1f}%)".format(
                len(path_failures), len(path_failures) / len(sequences) * 100
            )
        )
        print(
            "# Dest Failures: {} ({:.1f}%)".format(
                len(dest_failures), len(dest_failures) / len(sequences) * 100
            )
        )
    return successes, path_failures, dest_failures


def sample2sequence(sample, verbose=True):
    tokens = sample.split(" ")
    if len(tokens) <= 3:
        if verbose:
            print("Ignoring sample ", tokens)
        return []

    # Find the first 'end' token
    try:
        end_index = tokens.index("end")
    except ValueError:
        # No 'end' token found
        if verbose:
            print("No 'end' token found in sample ", tokens)
        tokens.append("end")
        end_index = tokens.index("end")

    # Truncate sequence up to and including the first 'end' token
    truncated_tokens = tokens[: end_index + 1]

    source = int(truncated_tokens[0])
    dest = int(truncated_tokens[1])
    return [source, dest] + truncated_tokens[2:-1]


def get_timestamp():
    timestamp = datetime.now()
    # Format the timestamp as a string
    return timestamp.strftime("%Y-%m-%d_%H-%M-%S")


def annotate_error_types(graph):
    """Given a graph with true and new edges, annotate what type of failures the new edges are"""
    pass


def sample_length_k_prefix_from_state(
    current_state,
    end_state,
    k,
    valid_previous_turns,
    node_and_previous_direction_to_neighbors,
):
    # Perform random walk
    state = current_state
    direction_list = []
    for _ in range(k):
        valid_directions = valid_previous_turns[state]
        direction = random.choice(valid_directions)
        state = random.choice(
            node_and_previous_direction_to_neighbors[(state, direction)]
        )
        direction_list.append(direction)
    direction_list.append(str(end_state))
    direction_list.append(str(state))
    direction_list = direction_list[::-1]
    return direction_list


def create_reverse_maps(valid_turns, node_and_direction_to_neighbor):
    valid_previous_turns = defaultdict(list)
    node_and_previous_direction_to_neighbors = defaultdict(list)
    for node, moves in valid_turns.items():
        for move in moves:
            next_move = node_and_direction_to_neighbor[(node, move)]
            valid_previous_turns[next_move].append(move)
            node_and_previous_direction_to_neighbors[(next_move, move)].append(node)
    return valid_previous_turns, node_and_previous_direction_to_neighbors


def get_node_coordinates():
    """Get node coordinates from OSMnx for Manhattan."""
    place_name = "Manhattan, New York City, New York, USA"
    historical_date = datetime(2024, 5, 5, 0, 0, 0)
    ox.settings.overpass_settings = (
        f'[out:json][timeout:180][date:"{historical_date.isoformat()}Z"]'
    )
    ox_graph = ox.graph_from_place(place_name, network_type="drive")

    node_to_coords = {}
    for node in ox_graph.nodes():
        node_to_coords[node] = (ox_graph.nodes[node]["y"], ox_graph.nodes[node]["x"])

    return node_to_coords


def create_geographic_regions(node_to_coords, all_pairs, n_regions=16):
    """Divide Manhattan into geographic regions and assign pairs to regions."""
    # Get coordinates for all nodes that appear in pairs
    pair_nodes = set()
    for pair in all_pairs:
        pair_nodes.update(pair)

    # Filter to only include nodes we have coordinates for
    valid_nodes = [node for node in pair_nodes if node in node_to_coords]
    coords_array = np.array([node_to_coords[node] for node in valid_nodes])

    # Use K-means to create geographic regions
    kmeans = KMeans(n_clusters=n_regions, random_state=42, n_init=10)
    region_labels = kmeans.fit_predict(coords_array)

    # Create mapping from node to region
    node_to_region = {}
    for i, node in enumerate(valid_nodes):
        node_to_region[node] = region_labels[i]

    # Group pairs by region
    region_pairs = {}
    for pair in all_pairs:
        node1, node2 = pair
        if node1 in node_to_region and node2 in node_to_region:
            region1 = node_to_region[node1]
            region2 = node_to_region[node2]
            # Use the region of the start node, or create cross-region pairs
            region_key = region1
            if region_key not in region_pairs:
                region_pairs[region_key] = []
            region_pairs[region_key].append(pair)

    print(f"Created {n_regions} geographic regions")
    for region_id, pairs in region_pairs.items():
        print(f"Region {region_id}: {len(pairs)} pairs")

    return region_pairs, node_to_region


def sample_pairs_evenly_by_region(region_pairs, batch_size):
    """Sample pairs evenly across geographic regions."""
    region_ids = list(region_pairs.keys())
    pairs_per_region = batch_size // len(region_ids)
    remaining_pairs = batch_size % len(region_ids)

    sampled_pairs = []

    for i, region_id in enumerate(region_ids):
        region_pairs_list = region_pairs[region_id]
        if len(region_pairs_list) == 0:
            continue

        # Sample pairs from this region
        n_samples = pairs_per_region + (1 if i < remaining_pairs else 0)
        n_samples = min(n_samples, len(region_pairs_list))

        if n_samples > 0:
            sampled_indices = np.random.choice(
                len(region_pairs_list), size=n_samples, replace=False
            )
            sampled_pairs.extend([region_pairs_list[idx] for idx in sampled_indices])

    # If we don't have enough pairs, fill with random sampling
    if len(sampled_pairs) < batch_size:
        all_remaining_pairs = []
        for pairs in region_pairs.values():
            all_remaining_pairs.extend(pairs)

        needed = batch_size - len(sampled_pairs)
        additional_pairs = np.random.choice(
            len(all_remaining_pairs), size=needed, replace=True
        )
        sampled_pairs.extend([all_remaining_pairs[idx] for idx in additional_pairs])

    return sampled_pairs
