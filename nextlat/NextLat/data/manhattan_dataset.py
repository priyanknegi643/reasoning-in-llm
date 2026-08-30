import torch
import os
import numpy as np
import pickle
import lightning as L
from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool
from datasets import Dataset as HFDataset, load_dataset
from huggingface_hub import hf_hub_download

from data.utils import ConstantLengthDataset

PAD_TOKEN_ID = 0


def _tokenize_single_sequence(sequence_word_id_pair):
    """Helper function for multiprocessing tokenization"""
    sequence, word_to_id = sequence_word_id_pair
    return [word_to_id.get(word, word_to_id["<pad>"]) for word in sequence.split()]


def tokenize_sequences(sequences, word_to_id, num_workers=1, batch_size=10000):
    """Tokenize sequences using multiprocessing with batching to prevent OOM"""
    if num_workers <= 1:
        # Process in batches even for single worker to avoid memory issues
        tokenized_sequences = []
        for i in tqdm(
            range(0, len(sequences), batch_size), desc="Tokenizing sequences"
        ):
            batch = sequences[i : i + batch_size]
            batch_tokenized = [
                [word_to_id.get(word, word_to_id["<pad>"]) for word in seq.split()]
                for seq in batch
            ]
            tokenized_sequences.extend(batch_tokenized)
        return tokenized_sequences

    # Use batched multiprocessing to prevent memory explosion
    tokenized_sequences = []
    for i in tqdm(range(0, len(sequences), batch_size), desc="Tokenizing sequences"):
        batch = sequences[i : i + batch_size]
        # Prepare arguments for this batch only
        args = [(seq, word_to_id) for seq in batch]

        with Pool(processes=num_workers) as pool:
            batch_tokenized = pool.map(_tokenize_single_sequence, args)
            tokenized_sequences.extend(batch_tokenized)

    return tokenized_sequences


def next_token_test(logits, sequence_str, tokenizer):
    generated_list = sequence_str.split(" ")
    try:
        start_node, end_node = int(generated_list[0]), int(generated_list[1])
    except ValueError:
        print("Invalid sequence: ", sequence_str)
        return 0, 0
    current_state = start_node
    num_total = 0
    num_success = 0
    for length_of_partial_sequence in range(2, len(generated_list)):
        top_pred = logits[length_of_partial_sequence - 1]
        top_pred_str = tokenizer.decode(top_pred)
        num_total += 1
        next_str = generated_list[length_of_partial_sequence]
        if top_pred_str in tokenizer.valid_turns[current_state]:
            num_success += 1
        elif top_pred_str == "end" and current_state == end_node:
            num_success += 1
        if next_str != "end":
            current_state = tokenizer.node_and_direction_to_neighbor[
                (current_state, next_str)
            ]
    return num_success, num_total


def is_valid_sequence(sample, valid_turns, node_and_direction_to_neighbor):
    """
    Returns:
        reached_end: True if the sequence reached the end node, False otherwise
        is_correct: True if every turn in the sequence is valid, False otherwise
    """
    generated_list = sample.split(" ")
    try:
        start_node, end_node = int(generated_list[0]), int(generated_list[1])
    except ValueError:
        print("Invalid sequence: ", sample)
        return False, False
    directions = generated_list[2:]
    current_state = start_node
    state_seq = [current_state]
    for _, direction in enumerate(directions):
        if direction != "end":
            if direction in valid_turns[current_state]:
                current_state = node_and_direction_to_neighbor[
                    (current_state, direction)
                ]
                state_seq.append(current_state)
            else:
                return False, False
        else:
            if current_state == end_node:
                return True, True
            else:
                return False, True
    # if we get here, we didn't reach the end node, but every turn was valid
    return False, True


class SimpleTokenizer:
    def __init__(self, sentences, node_and_direction_to_neighbor, valid_turns):
        words = set()
        for sentence in sentences:
            words.update(sentence.split())
        # Reverse sorted so cardinal directions are one or two digit so saved file is smaller
        self.word_to_id = {
            word: idx + 1 for idx, word in enumerate(sorted(words, reverse=True))
        }
        self.id_to_word = {id: word for word, id in self.word_to_id.items()}
        self.pad_token_id = PAD_TOKEN_ID
        self.word_to_id["<pad>"] = self.pad_token_id
        self.id_to_word[self.pad_token_id] = "<pad>"
        self.node_and_direction_to_neighbor = node_and_direction_to_neighbor
        self.valid_turns = valid_turns
        self.eos_token_id = self.word_to_id["end"]

    def encode(self, sentence):
        return [
            self.word_to_id.get(word, self.word_to_id["<pad>"])
            for word in sentence.split()
        ]

    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy()
        if token_ids.ndim == 0:
            token_ids = np.array([token_ids])
        return " ".join(
            self.id_to_word[id] for id in token_ids if id != self.pad_token_id
        )


class ManhattanDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, all_pairs=None):
        self.data = data
        self.tokenizer = tokenizer
        self.all_pairs = all_pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if isinstance(sample, dict) and "input_ids" in sample:
            return sample["input_ids"]
        return sample

    def __iter__(self):
        for sample in self.data:
            if isinstance(sample, dict) and "input_ids" in sample:
                yield sample["input_ids"]
            else:
                yield sample

    def shard(self, data_size_per_shard, index):
        print(
            "Sharding data...data_size_per_shard:", data_size_per_shard, "index:", index
        )
        start = index * data_size_per_shard
        end = (index + 1) * data_size_per_shard
        if isinstance(self.data, HFDataset):
            self.data = self.data.select(range(start, end))
        else:
            self.data = self.data[start:end]
        print("Sharded data...len:", len(self.data))

    def evaluate_manhattan(
        self, model, dataloader, config, fabric, show_progress_bar=True
    ):
        """
        Evaluate Manhattan sequence accuracy. Returns overall accuracy.
        """
        model.eval()
        test_logs = {}

        # correct destination
        total_sequences = 0
        total_reached_end = 0
        total_valid = 0

        # correct directions
        total_correct_directions = 0
        total_nodes = 0

        batch_count = 0

        disable_pbar = True if not show_progress_bar else None
        n_test_batches = None
        if hasattr(config.trainer, "test_batches") and config.trainer.test_batches > 0:
            n_test_batches = config.trainer.test_batches // fabric.world_size

        for batch in tqdm(
            dataloader,
            desc="Manhattan Evaluation",
            total=n_test_batches or len(dataloader),
            leave=False,
            disable=disable_pbar,
        ):
            input_ids = batch["input_ids"]
            batch_size = input_ids.size(0)
            prefix = input_ids[:, :2]

            with torch.inference_mode():
                predictions = model.generate(
                    prefix,
                    max_new_tokens=128,
                    temperature=getattr(config.trainer, "temperature", 1.0),
                    top_k=1,  # greedy decoding
                )

            batch_samples = [self.tokenizer.decode(pred) for pred in predictions]
            print("batch_samples: ", batch_samples[:5])
            for sample in batch_samples:
                total_sequences += 1
                reached_end, is_valid = is_valid_sequence(
                    sample,
                    self.tokenizer.valid_turns,
                    self.tokenizer.node_and_direction_to_neighbor,
                )
                if reached_end:
                    total_reached_end += 1
                if is_valid:
                    total_valid += 1

            logits = model.model(input_ids).argmax(dim=-1)
            for i in range(batch_size):
                sequence_str = self.tokenizer.decode(input_ids[i])
                correct_directions, node_count = next_token_test(
                    logits[i], sequence_str, self.tokenizer
                )
                total_correct_directions += correct_directions
                total_nodes += node_count
            batch_count += 1
            if n_test_batches is not None and batch_count >= n_test_batches:
                break

        overall_reached_end_accuracy = total_reached_end / total_sequences
        overall_valid_accuracy = total_valid / total_sequences
        overall_accuracy_directions = total_correct_directions / total_nodes
        test_logs["val/reached_end_accuracy"] = overall_reached_end_accuracy
        test_logs["val/valid_accuracy"] = overall_valid_accuracy
        test_logs["val/next_token_accuracy"] = overall_accuracy_directions
        fabric.print(
            f"{datetime.now()} Manhattan Test Reached End Accuracy: {overall_reached_end_accuracy:.2%}, Manhattan Test Valid Accuracy: {overall_valid_accuracy:.2%}, \
                Manhattan Test Directions Accuracy: {overall_accuracy_directions:.2%}"
        )
        return test_logs

    def evaluate_manhattan_generalization(
        self, model, dataloader, config, fabric, show_progress_bar=True
    ):
        """
        Evaluate Manhattan out-of-distribution accuracy. Returns overall accuracy.
        """
        assert self.all_pairs is not None, "all_pairs must be provided"
        model.eval()
        test_logs = {}
        total_sequences = 0
        total_reached_end = 0
        total_valid = 0

        num_samples = config.data.generalization_num_samples
        batch_size = config.data.device_batch_size

        for _ in range(num_samples):
            pairs = self.all_pairs[
                np.random.choice(len(self.all_pairs), size=batch_size)
            ]
            prefix = torch.tensor(
                [
                    self.tokenizer.encode(" ".join([str(x) for x in list(pair)]))
                    for pair in pairs
                ]
            ).to(fabric.device)
            with torch.inference_mode():
                generations = model.generate(
                    prefix,
                    max_new_tokens=128,
                    temperature=getattr(config.trainer, "temperature", 1.0),
                    top_k=1,  # greedy decoding
                )

            batch_samples = [self.tokenizer.decode(gen) for gen in generations]

            for sample in batch_samples:
                total_sequences += 1
                reached_end, is_valid = is_valid_sequence(
                    sample,
                    self.tokenizer.valid_turns,
                    self.tokenizer.node_and_direction_to_neighbor,
                )
                if reached_end:
                    total_reached_end += 1
                if is_valid:
                    total_valid += 1

        overall_reached_end_accuracy = total_reached_end / total_sequences
        overall_valid_accuracy = total_valid / total_sequences
        test_logs["generalization/test_accuracy"] = overall_reached_end_accuracy
        test_logs["generalization/valid_accuracy"] = overall_valid_accuracy
        fabric.print(
            f"{datetime.now()} Manhattan Generalization Test Reached End Accuracy: {overall_reached_end_accuracy:.2%}, \
                Manhattan Generalization Test Valid Accuracy: {overall_valid_accuracy:.2%}"
        )
        return test_logs


class ManhattanDataModule:
    def __init__(self, fabric: L.Fabric, config):
        super().__init__()
        self.fabric = fabric
        self.data_path = config.data.data_path
        self.batch_size = config.data.device_batch_size
        self.num_workers = config.data.num_workers
        self.use_pretokenized = config.data.use_pretokenized
        self.pretokenized_hf_repo = "JaydenTeoh/manhattan"

        assert self.num_workers >= 0, "num_workers must be >= 0"
        # Note: num_workers > 1 is allowed for tokenization, but ConstantLengthDataset requires num_workers <= 1

        fabric.print("Loading datasets...")
        # Load graph artifacts used by evaluation/generalization.
        with self.fabric.rank_zero_first(local=True):
            if self.use_pretokenized:

                def _load_pickle_from_hub(filename):
                    local_path = hf_hub_download(
                        repo_id=self.pretokenized_hf_repo,
                        filename=filename,
                        repo_type="dataset",
                    )
                    with open(local_path, "rb") as f:
                        return pickle.load(f)

                all_pairs = _load_pickle_from_hub("all_pairs.pkl")
                valid_turns = _load_pickle_from_hub("valid_turns.pkl")
                node_and_direction_to_neighbor = _load_pickle_from_hub(
                    "node_and_direction_to_neighbor.pkl"
                )
            else:
                with open(f"{self.data_path}/all_pairs.pkl", "rb") as f:
                    all_pairs = pickle.load(f)
                with open(f"{self.data_path}/valid_turns.pkl", "rb") as f:
                    valid_turns = pickle.load(f)
                with open(
                    f"{self.data_path}/node_and_direction_to_neighbor.pkl", "rb"
                ) as f:
                    node_and_direction_to_neighbor = pickle.load(f)

        if self.use_pretokenized:
            fabric.print(
                f"Loading pretokenized splits from HF dataset {self.pretokenized_hf_repo}"
            )
            parquet_data = load_dataset(self.pretokenized_hf_repo)
            tokenized_train = parquet_data["train"]
            tokenized_valid = parquet_data["heldout"]

            tokenizer_path = hf_hub_download(
                repo_id=self.pretokenized_hf_repo,
                filename="tokenizer.pkl",
                repo_type="dataset",
            )
            with open(tokenizer_path, "rb") as f:
                tokenizer = pickle.load(f)
            fabric.print(f"Loaded tokenizer from {tokenizer_path}")
        else:
            # Backward-compatible path: load raw text and tokenize.
            fabric.print(
                f"Loading raw text and tokenizing from scratch. Will take a while..."
            )
            with self.fabric.rank_zero_first(local=True):
                with open(f"{self.data_path}/train_sequences.txt", "r") as f:
                    train_sequences = f.read().split("\n")
                with open(f"{self.data_path}/heldout_sequences.txt", "r") as f:
                    heldout_sequences = f.read().split("\n")

            tokenizer = SimpleTokenizer(
                train_sequences, node_and_direction_to_neighbor, valid_turns
            )
            fabric.print("Tokenizer created, eos_token_id:", tokenizer.eos_token_id)

            # Save the tokenizer
            tokenizer_path = os.path.join(self.data_path, "tokenizer.pkl")
            with open(tokenizer_path, "wb") as f:
                pickle.dump(tokenizer, f)
            fabric.print(f"Tokenizer saved to {tokenizer_path}")

            # Validate on 4000 heldout sequences during training (in order to have at least one batch)
            heldout_subsample_size = 4000
            heldout_sequences = [
                heldout_sequences[i]
                for i in np.random.choice(
                    len(heldout_sequences), heldout_subsample_size, replace=False
                )
            ]
            fabric.print("...done!")

            fabric.print("Tokenizing sequences...")
            fabric.print(f"Tokenizing {len(train_sequences)} training sequences...")
            tokenized_train = tokenize_sequences(
                train_sequences,
                tokenizer.word_to_id,
                self.num_workers,
                batch_size=10000,
            )

            fabric.print(f"Tokenizing {len(heldout_sequences)} validation sequences...")
            tokenized_valid = tokenize_sequences(
                heldout_sequences,
                tokenizer.word_to_id,
                self.num_workers,
                batch_size=10000,
            )

        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.word_to_id)
        self.total_len = 128  # traversal sequences are never longer than 100 tokens
        fabric.print(f"Vocab size: {self.vocab_size}")

        all_pairs = np.array(all_pairs)

        train_dataset = ManhattanDataset(tokenized_train, tokenizer)
        val_dataset = ManhattanDataset(tokenized_valid, tokenizer)

        # Partition data between devices
        # Because we later create an iterable dataset, we first need to manually shard the data here
        data_size_per_shard = len(tokenized_train) // self.fabric.world_size
        fabric.print("Creating shards...data_size_per_shard:", data_size_per_shard)
        train_dataset.shard(
            data_size_per_shard=data_size_per_shard, index=self.fabric.global_rank
        )

        # Pack data
        train_dataset = ConstantLengthDataset(
            dataset=train_dataset,
            tokenizer=tokenizer,
            formatting_func=lambda x: x,
            seq_length=config.model.block_size,
            append_concat_token=False,  # the dataset already has an eos token at the end of each sequence
            pretokenized=True,
            no_invalid_starts=True,
        )
        val_dataset = ConstantLengthDataset(
            dataset=val_dataset,
            tokenizer=tokenizer,
            formatting_func=lambda x: x,
            seq_length=config.model.block_size,
            append_concat_token=False,  # the dataset already has an eos token at the end of each sequence
            pretokenized=True,
            no_invalid_starts=True,
        )
        assert (
            train_dataset.pretokenized == True
        ), "train_dataset should be marked as pretokenized"
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        if config.data.test_generalization:
            # only use 1000 of the heldout sequences for generalization
            valid_subset = (
                tokenized_valid.select(range(min(1000, len(tokenized_valid))))
                if isinstance(tokenized_valid, HFDataset)
                else tokenized_valid[:1000]
            )
            generalization_dataset = ManhattanDataset(
                valid_subset, tokenizer, all_pairs
            )
            self.generalization_dataset = generalization_dataset
        fabric.print("...done!")

    def train_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            pin_memory=True,
            drop_last=True,
        )
        return self.fabric.setup_dataloaders(dataloader)

    def val_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            pin_memory=True,
            drop_last=True,
        )
        return self.fabric.setup_dataloaders(dataloader)

    def generalization_dataloader(self):
        # this uses padded inputs
        dataloader = torch.utils.data.DataLoader(
            self.generalization_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.collate_fn,
        )
        return self.fabric.setup_dataloaders(dataloader)

    def update_config(self, config):
        config.model.vocab_size = self.vocab_size
        config.model.context_length = 1  # index of the second coordinate

    def get_tokenizer(self):
        return self.tokenizer

    def prepare_batch(self, batch):
        return batch["input_ids"]

    @staticmethod
    def collate_fn(batch):
        max_length = max(len(data) for data in batch)
        padded_inputs = []
        attention_masks = []
        labels = []
        for data in batch:
            padding_length = max_length - len(data)
            padded_input = data + [PAD_TOKEN_ID] * padding_length
            attention_mask = [1] * len(data) + [0] * padding_length
            padded_inputs.append(padded_input)
            attention_masks.append(attention_mask)
            labels.append(
                [-100] * 2 + padded_input[2:]
            )  # first two tokens are start and end node
        return {
            "input_ids": torch.tensor(padded_inputs),
            "attention_mask": torch.tensor(attention_masks),
            "labels": torch.tensor(labels),
        }
