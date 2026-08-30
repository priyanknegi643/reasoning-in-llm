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

# Add the root directory to Python path to import core_train
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from core_train import initialize_model
from data.manhattan_dataset import is_valid_sequence
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a model checkpoint")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        required=False,
        help="Number of loops of generation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        required=False,
        help="Batch size for each loop of generation",
    )
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

    # These are pairs that are (a) seen in the training data and (b) have
    # legal traversals up to the max length of the training data
    with open(f"{args.data_dir}/eval_pairs_dist50.pkl", "rb") as f:
        all_pairs = pickle.load(f)

    all_pairs = np.array(all_pairs)
    device = fabric.device
    print(f"Device: {device}")

    samples = []
    num_successful = 0
    num_valid = 0
    total_nodes = 0
    bar = tqdm(range(args.num_samples))
    for _ in bar:
        pairs = all_pairs[np.random.choice(len(all_pairs), size=args.batch_size)]
        prefix = torch.tensor(
            [tokenizer.encode(" ".join([str(x) for x in list(pair)])) for pair in pairs]
        ).to(device)
        generated_ids = model.generate(
            prefix,
            max_new_tokens=100,
            temperature=getattr(config.trainer, "temperature", 1.0),
            top_k=1,  # sample
        )
        batch_samples = [
            tokenizer.decode(generated_id) for generated_id in generated_ids
        ]
        samples.extend(batch_samples)
        for sample in batch_samples:
            total_nodes += 1
            reached_end, is_valid = is_valid_sequence(
                sample, valid_turns, node_and_direction_to_neighbor
            )
            if reached_end:
                num_successful += 1
            if is_valid:
                num_valid += 1
        bar.set_description(
            f"Fraction successful: {num_successful/total_nodes:.2f} ({num_successful}/{total_nodes}), Fraction valid: {num_valid/total_nodes:.2f} ({num_valid}/{total_nodes})"
        )

    # Save samples
    os.makedirs(f"{args.data_dir}/samples", exist_ok=True)
    with open(f"{args.data_dir}/samples/{args.model_name}.txt", "w") as f:
        for sample in samples:
            f.write(sample)
            f.write("\n")
