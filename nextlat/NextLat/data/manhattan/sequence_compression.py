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
from data.manhattan.utils import create_reverse_maps, sample_length_k_prefix_from_state


def get_conditional_probability_of_suffixes_after_prefix(
    prefix, sequences, model, tokenizer, batch_size=32
):
    prefix_len = len(prefix)
    # Pad input_ids to the same length
    padded_input_ids = []
    max_suffix_len = max(len(sequence) for sequence in sequences)
    for ids in sequences:
        padding_length = max_suffix_len - len(ids)
        padded_ids = ids + [tokenizer.pad_token_id] * padding_length
        padded_input_ids.append(padded_ids)
    padded_input_ids = torch.tensor(padded_input_ids).to(model.model.device)
    num_batches = (len(padded_input_ids) - 1) // batch_size + 1
    logits_list = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        with torch.no_grad():
            logits = model.model(batch=padded_input_ids[start_idx:end_idx])
        logits_list.append(logits)
    logits = torch.cat(logits_list, dim=0)
    probs = torch.softmax(logits, -1)
    next_token_probs = torch.gather(
        probs[:, :-1], dim=-1, index=padded_input_ids[:, 1:].unsqueeze(-1)
    )[:, :, 0]
    num_suffixes = len(sequences)
    # Adjust the indexing to account for different suffix lengths
    suffix_probs = []
    for j in range(num_suffixes):
        suffix_probs.append(
            next_token_probs[j, (prefix_len - 1) : (len(sequences[j]) - 1)]
            .cpu()
            .numpy()
        )
    print("prefix length", prefix_len)
    print("next token probs length", len(next_token_probs[0]))
    print("next token probs", next_token_probs[0])
    print("suffix length", len(suffix_probs[0]))
    print("suffix probs", suffix_probs[0])
    return suffix_probs


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
    parser.add_argument("--num-suffix-samples", type=int, default=30)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--num-trials", type=int, default=100)
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

    all_start_nodes = set([pair[0] for pair in all_pairs])
    all_end_nodes = set([pair[1] for pair in all_pairs])
    all_nodes = all_start_nodes.union(all_end_nodes)
    all_nodes = np.array([node for node in all_nodes if len(valid_turns[node]) > 0])

    for node in all_nodes:
        node_and_direction_to_neighbor[(node, "end")] = "end"
    node_and_direction_to_neighbor[("end", "end")] = "end"
    node_and_neighbor_to_direction = {
        (node, neighbor): direction
        for (node, direction), neighbor in node_and_direction_to_neighbor.items()
    }

    (
        valid_previous_turns,
        node_and_previous_direction_to_neighbors,
    ) = create_reverse_maps(valid_turns, node_and_direction_to_neighbor)

    def perform_single_compression_test():
        prefix1 = None
        prefix2 = None
        while prefix1 == prefix2 or prefix1 is None or prefix2 is None:
            state_ind = np.random.choice(len(all_pairs))
            start_node, end_node = all_pairs[state_ind]
            shortest_path = shortest_paths[(start_node, end_node)]
            # Make sure we can get to the end node in 100 moves.
            prefix_len = np.random.choice(
                range(1, 100 - shortest_path - num_special_tokens)
            )
            prefix1 = sample_length_k_prefix_from_state(
                start_node,
                end_node,
                prefix_len,
                valid_previous_turns,
                node_and_previous_direction_to_neighbors,
            )
            prefix2 = sample_length_k_prefix_from_state(
                start_node,
                end_node,
                prefix_len,
                valid_previous_turns,
                node_and_previous_direction_to_neighbors,
            )
        suffixes1 = []
        prefix = torch.tensor(
            [tokenizer.encode(" ".join([str(x) for x in prefix1]))]
        ).to(device)
        for _ in range(args.num_suffix_samples):
            generated_ids = model.generate(
                prefix,
                max_new_tokens=100 - len(prefix[0]),
                temperature=getattr(config.trainer, "temperature", 1.0),
                top_k=None,  # sample
            )[0]
            generated_ids = generated_ids.tolist()
            # Find the first 'end' token
            try:
                end_index = generated_ids.index(tokenizer.word_to_id["end"])
            except ValueError:
                end_index = len(generated_ids)

            # Truncate sequence up to and including the first 'end' token
            truncated_tokens = generated_ids[: end_index + 1]
            suffixes1.append(truncated_tokens)

        suffix1_probs_prefix2 = get_conditional_probability_of_suffixes_after_prefix(
            prefix2, suffixes1, model, tokenizer
        )
        precision = all(
            [
                all(suffix1_probs_prefix2[i] > args.epsilon)
                for i in range(args.num_suffix_samples)
            ]
        )
        return float(precision), tuple(prefix1), tuple(prefix2), start_node, end_node

    state_pair_to_prefixes_to_score = defaultdict(lambda: defaultdict(list))
    bar = tqdm(range(args.num_trials))
    for trial in bar:
        # Retry until we get a successful trial
        while True:
            try:
                (
                    precision,
                    prefix1,
                    prefix2,
                    start_node,
                    end_node,
                ) = perform_single_compression_test()
                state_pair_to_prefixes_to_score[(start_node, end_node)][
                    (prefix1, prefix2)
                ].append(precision)
                average_precisions = [
                    [np.mean(v) for k, v in inner_dict.items()]
                    for k1, inner_dict in state_pair_to_prefixes_to_score.items()
                ]
                mean_precision = np.mean(average_precisions)
                std = np.std(average_precisions) / np.sqrt(len(average_precisions))
                bar.set_description(
                    f"Mean compression precision: {mean_precision:.3f} ({std:.3f})"
                )
                break  # Success, exit retry loop
            except Exception as e:
                # Retry on any error - don't print to avoid spam
                continue

    # Print final results
    print(f"\nFinal Results:")
    print(f"Mean compression precision: {mean_precision:.3f} ({std:.3f})")
    print(f"Total trials completed: {len(state_pair_to_prefixes_to_score)}")

    # Save samples
    os.makedirs(f"{args.data_dir}/samples", exist_ok=True)
    with open(
        f"{args.data_dir}/samples/{args.model_name}_compression_test.txt", "w"
    ) as f:
        f.write(f"Mean compression precision: {mean_precision:.3f} ({std:.3f})")
        f.write("\n")
