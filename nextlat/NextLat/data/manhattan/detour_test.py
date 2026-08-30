import torch
import numpy as np
import argparse
from tqdm import tqdm
import pickle
import os
import sys
import yaml
import osmnx as ox
from datetime import datetime
from sklearn.cluster import KMeans

import lightning as L
from argparse import ArgumentParser
from omegaconf import OmegaConf, DictConfig
from collections import defaultdict

# Add the root directory to Python path to import core_train
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from core_train import initialize_model
from data.manhattan import is_valid_sequence
import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a model checkpoint")
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to the checkpoint file"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        required=True,
        help="Path to write output - if missing only stout will be used",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        required=True,
        help="Name of the model for naming sample trajectories",
    )
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument(
        "--n_regions",
        type=int,
        default=16,
        help="Number of geographic regions for sampling (default: 16)",
    )
    parser.add_argument("--detour-prob", type=float, default=0.75)
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument(
        "--detour-type",
        type=str,
        default="random_valid",
        help="Options: least_likely, random_valid, second_most_likely",
    )
    args, conf_cli = parser.parse_known_args()

    default = OmegaConf.load("defaults.yaml")
    overrides = OmegaConf.load(args.config)
    cli = OmegaConf.from_dotlist(conf_cli)
    config = OmegaConf.merge(default, overrides, cli)

    # Initialize fabric
    fabric = L.Fabric()
    assert fabric.world_size == 1, "This script only supports single GPU"
    fabric.seed_everything(1234)
    print(yaml.dump(OmegaConf.to_container(config)))

    # Load model from checkpoint
    checkpoint_path = args.checkpoint_path
    assert os.path.isfile(
        checkpoint_path
    ), f"Checkpoint file {checkpoint_path} not found"
    with open(f"{args.data_dir}/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    config.model.vocab_size = len(tokenizer.word_to_id)
    model = initialize_model(
        fabric,
        config,
        tokenizer,
        initialize_optimizer=False,
        checkpoint_path=checkpoint_path,
    )

    valid_turns = tokenizer.valid_turns
    node_and_direction_to_neighbor = tokenizer.node_and_direction_to_neighbor
    eos_token_id = tokenizer.word_to_id["end"]
    num_special_tokens = 3  # <start node>, <end node>, ...,  <end>

    # These are pairs that are (a) seen in the training data and (b) have
    # legal traversals up to the max length of the training data
    with open(f"{args.data_dir}/eval_pairs_dist50.pkl", "rb") as f:
        all_pairs = pickle.load(f)

    with open(f"data/manhattan/random_walks/shortest_paths.pkl", "rb") as f:
        shortest_paths = pickle.load(f)

    all_pairs = np.array(all_pairs)
    device = fabric.device
    print(f"Device: {device}")

    bar = tqdm(range(args.num_trials))
    total_nodes = 0
    success_nodes = 0
    for index in bar:
        pairs = all_pairs[np.random.choice(len(all_pairs), size=1)]
        input_ids = torch.tensor(
            [tokenizer.encode(" ".join([str(x) for x in list(pair)])) for pair in pairs]
        ).to(device)
        origin, destination = pairs[0][0], pairs[0][1]
        state = int(origin)
        for i in range(2, 100):  # sample 98 tokens
            turn_options = valid_turns[state]
            turn_states = [
                node_and_direction_to_neighbor[(state, turn)] for turn in turn_options
            ]
            turn_dists = [
                (
                    shortest_paths[(turn_state, destination)]
                    if (turn_state, destination) in shortest_paths
                    else np.inf
                )
                for turn_state in turn_states
            ]
            # Only include turns that can get us to the destination in less than 100 - i moves
            turn_options = [
                turn_options[turn_ind]
                for turn_ind in range(len(turn_options))
                if turn_dists[turn_ind] < 100 - num_special_tokens - i
            ]
            with torch.no_grad():
                logits = model.model(input_ids)
                probs = torch.softmax(logits, -1)
            relevant_probs = probs[
                0, -1, [tokenizer.word_to_id[turn] for turn in turn_options]
            ]
            transformer_pred = torch.argmax(probs[0, -1, :])
            # Insert detour with probability detour_prob
            if (
                np.random.rand() < args.detour_prob
                and state != destination
                and len(turn_options) > 0
            ):
                if args.detour_type == "least_likely":
                    # Force model to take least likely one.
                    # next_token = transformer_pred[None, None]
                    next_token = (
                        torch.tensor(
                            tokenizer.encode(turn_options[torch.argmin(relevant_probs)])
                        )
                        .unsqueeze(0)
                        .to(model.model.device)
                    )
                elif args.detour_type == "random_valid":
                    # Sample a valid one that's different from the transformer pred
                    turn_options_except_pred = [
                        turn
                        for turn in turn_options
                        if turn != tokenizer.decode(transformer_pred)
                    ]
                    if len(turn_options_except_pred) == 0:
                        next_token = (
                            torch.tensor(tokenizer.encode(turn_options[0]))
                            .unsqueeze(0)
                            .to(model.model.device)
                        )
                    else:
                        next_token = (
                            torch.tensor(
                                tokenizer.encode(
                                    np.random.choice(turn_options_except_pred)
                                )
                            )
                            .unsqueeze(0)
                            .to(model.model.device)
                        )
                elif args.detour_type == "second_most_likely":
                    # Force model to take highest rated turn option except pred
                    turn_options_except_pred = [
                        turn
                        for turn in turn_options
                        if turn != tokenizer.decode(transformer_pred)
                    ]
                    if len(turn_options_except_pred) == 0:
                        next_token = (
                            torch.tensor(tokenizer.encode(turn_options[0]))
                            .unsqueeze(0)
                            .to(model.model.device)
                        )
                    else:
                        relevant_probs = probs[
                            0,
                            -1,
                            [
                                tokenizer.word_to_id[turn]
                                for turn in turn_options_except_pred
                            ],
                        ]
                        next_token = (
                            torch.tensor(
                                tokenizer.encode(
                                    turn_options_except_pred[
                                        torch.argmax(relevant_probs)
                                    ]
                                )
                            )
                            .unsqueeze(0)
                            .to(model.model.device)
                        )
                else:
                    raise ValueError(f"Invalid detour type: {args.detour_type}")
            else:
                next_token = transformer_pred[None, None]
            input_ids = torch.cat((input_ids, next_token), dim=-1)
            if next_token == eos_token_id:
                break
            try:
                state = node_and_direction_to_neighbor[
                    (state, tokenizer.decode(next_token[0]))
                ]
            except:
                # Getting here means the model has suggested a token that's not in the valid_turns dict.
                pass

        path = tokenizer.decode(input_ids[0])
        reached_end, is_valid = is_valid_sequence(
            path, valid_turns, node_and_direction_to_neighbor
        )
        print(f"Reached end: {reached_end}, is valid: {is_valid}")
        print(f"Path: {path}")
        total_nodes += 1
        if is_valid:
            success_nodes += 1
        success_rate = success_nodes / total_nodes
        std = np.sqrt(success_rate * (1 - success_rate) / total_nodes)
        bar.set_description(
            f"Fraction successful {args.data_dir} ({args.detour_type} detours, p={args.detour_prob}): {success_rate:.3f} ({std:.3f})"
        )
