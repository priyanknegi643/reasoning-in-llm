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
from data.manhattan_dataset import (
    is_valid_sequence,
    tokenize_sequences,
    ManhattanDataset,
)
from data.utils import ConstantLengthDataset

from models.model_gpt import GPT
from models.model_bst import BST
from models.model_nextlat import NextLat
from models.model_mtp_jtp import JTP


def calculate_participation_ratio(hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Computes the participation ratio (effective dimensionality)
    of hidden representations across the batch and sequence.
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape

    # Flatten batch and sequence into one dimension
    X = hidden_states.reshape(-1, hidden_dim)  # (B*T, D)

    # Center the representations
    X = X - X.mean(dim=0, keepdim=True)

    # Compute covariance (D x D)
    cov = X.T @ X / (X.shape[0] - 1)

    # Eigenvalues of covariance (variances along principal axes)
    eigvals = torch.linalg.eigvalsh(cov)

    # Keep only real part (should be real anyway)
    eigvals = eigvals.real

    # Avoid numerical issues
    eigvals = torch.clamp(eigvals, min=1e-12)

    # Compute participation ratio
    pr = (eigvals.sum() ** 2) / (eigvals.pow(2).sum())

    return pr


def calculate_effective_rank(hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Computes the effective rank (spectral entropy) of hidden representations
    across the batch and sequence.
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape

    # Flatten batch and sequence into one dimension
    X = hidden_states.reshape(-1, hidden_dim)  # (B*T, D)
    U_batch, S_batch, V_batch = torch.svd(X)

    # Effective rank (Shannon entropy of normalized singular values)
    normalized_s_batch = S_batch / S_batch.sum()
    normalized_s_batch = torch.clamp(normalized_s_batch, min=1e-12)
    effective_rank = torch.exp(
        -torch.sum(normalized_s_batch * torch.log(normalized_s_batch))
    )

    return effective_rank


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
        default=256,
        required=False,
        help="Batch size for each loop of generation",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=100,
        required=False,
        help="Number of batches to run",
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
    parser.add_argument("--num_sequences", type=int, default=100)
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
    with open(f"{args.data_dir}/heldout_sequences.txt", "r") as f:
        heldout_sequences = f.read().split("\n")

    # Validate on X heldout sequences
    heldout_subsample_size = (
        args.num_batches * args.batch_size * 10
    )  # roughly 10 trajectories packed into a sequence
    heldout_sequences = [
        heldout_sequences[i]
        for i in np.random.choice(
            len(heldout_sequences), heldout_subsample_size, replace=False
        )
    ]

    tokenized_valid = tokenize_sequences(
        heldout_sequences, tokenizer.word_to_id, num_workers=1, batch_size=10000
    )
    val_dataset = ManhattanDataset(tokenized_valid, tokenizer)
    device = fabric.device
    print(f"Device: {device}")

    val_dataset = ConstantLengthDataset(
        dataset=val_dataset,
        tokenizer=tokenizer,
        formatting_func=lambda x: x,
        seq_length=config.model.block_size,
        append_concat_token=False,  # the dataset already has an eos token at the end of each sequence
        pretokenized=True,
        no_invalid_starts=True,
    )
    dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )
    dataloader = fabric.setup_dataloaders(dataloader)

    participation_ratio = torch.zeros(1, device=fabric.device)
    effective_rank = torch.zeros(1, device=fabric.device)
    actual_num_batches = 0
    for batch in tqdm(
        dataloader,
        desc="Validation",
        total=len(dataloader),
        leave=False,
    ):
        batch = batch["input_ids"]
        inputs = batch.to(device)
        with torch.no_grad():
            if isinstance(model, NextLat):
                hidden_states, _ = model.model(inputs, return_hidden_states=True)
            elif isinstance(model, BST):
                hidden_states, _ = model.encoder(
                    inputs, compute_forward=True, compute_backward=False
                )
            else:
                _, hidden_states = model.model(inputs, return_hidden_states=True)

        participation_ratio += calculate_participation_ratio(hidden_states)
        effective_rank += calculate_effective_rank(hidden_states)
        actual_num_batches += 1

    participation_ratio /= actual_num_batches
    effective_rank /= actual_num_batches
    print(f"Mean participation ratio: {participation_ratio.item()}")
    print(f"Mean effective rank: {effective_rank.item()}")
