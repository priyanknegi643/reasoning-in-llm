# Adapted from https://github.com/KellerJordan/modded-nanogpt
import os
import time
import torch
import glob
import random
from pathlib import Path
from torch.utils.data import IterableDataset, DataLoader
from transformers import GPT2TokenizerFast
import lightning as L  # Fabric
from typing import Any, Callable, Dict, List, Optional
from functools import partial

from models.model_speculative import SpeculativeModel
from utils.speculative_sampling import speculative_decode_v2


def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens

    return tokens


class FineWebIterableDataset(IterableDataset):
    def __init__(self, files, seq_len, rank, world_size, shuffle=True):
        self.files = files
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.epoch = 0  # default

        self.tokens_per_file = 100_000_000
        self.total_tokens = len(self.files) * self.tokens_per_file

    def __len__(self):
        num_sequences = self.total_tokens // self.seq_len
        sequences_per_rank = num_sequences // self.world_size
        return sequences_per_rank

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        files = self.files.copy()
        rng = random.Random(self.epoch)
        if self.shuffle:
            rng.shuffle(files)

        for file in files:
            tokens = _load_data_shard(file)
            num_possible_offsets = (len(tokens) - self.seq_len) // self.seq_len
            if num_possible_offsets <= 0:
                continue

            all_offsets = list(range(num_possible_offsets))
            rng.shuffle(all_offsets)
            local_offsets = all_offsets[self.rank :: self.world_size]

            for offset_index in local_offsets:
                start = offset_index * self.seq_len
                end = start + self.seq_len
                chunk = tokens[start:end]
                if len(chunk) < self.seq_len:
                    continue
                example = chunk.to(torch.long)
                yield {"input_ids": example, "labels": example}

    @torch.inference_mode()
    def evaluate_speculative_fineweb(
        self,
        model_wrapper: Any,
        val_dataloader: Any,
        config: Any,
        fabric: L.Fabric,
        prepare_batch_func: Optional[Callable] = None,
    ) -> Dict[str, float]:
        return evaluate_speculative_fineweb(
            model_wrapper=model_wrapper,
            val_dataloader=val_dataloader,
            fabric=fabric,
            trainer_config=config.trainer,
            model_config=config.model,
            data_config=config.data,
            prepare_batch_func=prepare_batch_func,
        )


class FineWebDataModule:
    """
    Data Module for FineWebEdu dataset.
    """

    def __init__(self, fabric: L.Fabric, config):
        self.fabric = fabric
        self.data_dir = Path(config.data.fineweb_data_path)
        self.num_train_files = config.data.num_train_files
        self.batch_size = config.data.device_batch_size
        self.seq_len = config.model.block_size

        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.vocab_size = self.tokenizer.vocab_size

        self.train_files = sorted(self.data_dir.glob("fineweb_train_*.bin"))
        self.train_files = self.train_files[
            : self.num_train_files
        ]  # Each train file has 100M tokens
        assert len(self.train_files) > 0, "No train bin files found."

        val_path = self.data_dir / "fineweb_val_000000.bin"
        self.val_files = [val_path] if val_path.exists() else []
        assert len(self.val_files) > 0, "No val bin files found"

        self.train_dataset = None
        self.val_dataset = None

        self.rank = self.fabric.global_rank
        self.world_size = self.fabric.world_size
        self.epoch = 0  # default epoch
        self.num_workers = config.data.num_workers

        assert self.num_workers >= 0, "num_workers must be >= 0"

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        if self.train_dataset is not None:
            self.train_dataset.set_epoch(epoch)
        if self.val_dataset is not None:
            self.val_dataset.set_epoch(epoch)

    def train_dataloader(self):
        self.train_dataset = FineWebIterableDataset(
            files=self.train_files,
            seq_len=self.seq_len,
            rank=self.rank,
            world_size=self.world_size,
            shuffle=True,
        )
        self.train_dataset.set_epoch(self.epoch)
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return self.fabric.setup_dataloaders(dataloader)

    def val_dataloader(self):
        if not self.val_files:
            return None
        self.val_dataset = FineWebIterableDataset(
            files=self.val_files,
            seq_len=self.seq_len,
            rank=self.rank,
            world_size=self.world_size,
            shuffle=False,
        )
        self.val_dataset.set_epoch(self.epoch)
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return self.fabric.setup_dataloaders(dataloader)

    def update_config(self, config):
        config.model.vocab_size = self.vocab_size

    def get_tokenizer(self):
        return self.tokenizer

    def prepare_batch(self, batch):
        return batch["input_ids"]


@torch.inference_mode()
def evaluate_speculative_fineweb(
    model_wrapper: Any,
    val_dataloader: Any,
    fabric: L.Fabric,
    trainer_config: Any,
    model_config: Any,
    data_config: Any,
    prepare_batch_func: Optional[Callable] = None,
) -> Dict[str, float]:
    if not isinstance(model_wrapper, SpeculativeModel):
        return {}
    prompts = collect_speculative_prompts_from_val_dataloader(
        val_dataloader=val_dataloader,
        trainer_config=trainer_config,
        model_config=model_config,
        data_config=data_config,
        world_size=fabric.world_size,
        prepare_batch_func=prepare_batch_func,
    )
    if not prompts:
        return {}

    gamma = int(
        trainer_config.get(
            "spec_eval_gamma", max(1, int(getattr(model_config, "mtp_horizon", 4)))
        )
    )
    max_new_tokens = int(trainer_config.get("spec_eval_max_new_tokens", 64))
    temperature = float(trainer_config.get("spec_eval_temperature", 1.0))
    top_k = trainer_config.get("spec_eval_top_k", 1)
    top_p = trainer_config.get("spec_eval_top_p", None)
    if top_k is not None:
        top_k = int(top_k)
    if top_p is not None:
        top_p = float(top_p)

    propose_fn = model_wrapper.build_speculative_propose_fn(
        temperature=temperature, top_k=top_k, top_p=top_p
    )
    target_probs_fn = model_wrapper.build_speculative_target_probs_fn(
        temperature=temperature, top_k=top_k, top_p=top_p
    )

    ar_seconds_local = 0.0
    spec_seconds_local = 0.0
    local_samples = len(prompts)
    accepted_local = 0.0
    proposed_local = 0.0
    accepted_pos2plus_local = 0.0
    proposed_pos2plus_all_drafted_local = 0.0
    target_samples_local = 0.0
    resamples_local = 0.0
    pos_proposed_local = [0.0 for _ in range(gamma)]
    pos_accepted_local = [0.0 for _ in range(gamma)]

    for prompt in prompts:
        idx = torch.tensor([prompt], dtype=torch.long, device=fabric.device)

        if fabric.device.type == "cuda":
            torch.cuda.synchronize(fabric.device)
        t0 = time.perf_counter()
        _ = model_wrapper.generate(
            idx.clone(),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        if fabric.device.type == "cuda":
            torch.cuda.synchronize(fabric.device)
        ar_seconds_local += time.perf_counter() - t0

        if fabric.device.type == "cuda":
            torch.cuda.synchronize(fabric.device)
        t1 = time.perf_counter()
        _, stats = speculative_decode_v2(
            idx=idx.clone(),
            max_new_tokens=max_new_tokens,
            gamma=gamma,
            propose_fn=propose_fn,
            target_probs_fn=target_probs_fn,
        )
        if fabric.device.type == "cuda":
            torch.cuda.synchronize(fabric.device)
        spec_seconds_local += time.perf_counter() - t1

        accepted_local += float(stats.get("accepted_tokens", 0.0))
        proposed_local += float(stats.get("proposed_tokens", 0.0))
        accepted_pos2plus_local += float(stats.get("accepted_tokens_pos2plus", 0.0))
        proposed_pos2plus_all_drafted_local += float(
            stats.get("proposed_tokens_pos2plus_all_drafted", 0.0)
        )
        target_samples_local += float(stats.get("target_samples", 0.0))
        resamples_local += float(stats.get("resamples", 0.0))
        for pos in range(1, gamma):
            pos_idx = pos + 1
            pos_proposed_local[pos] += float(
                stats.get(f"proposed_tokens_pos_{pos_idx}", 0.0)
            )
            pos_accepted_local[pos] += float(
                stats.get(f"accepted_tokens_pos_{pos_idx}", 0.0)
            )

    device = fabric.device
    ar_seconds = fabric.all_reduce(
        torch.tensor(ar_seconds_local, device=device), reduce_op="sum"
    ).item()
    spec_seconds = fabric.all_reduce(
        torch.tensor(spec_seconds_local, device=device), reduce_op="sum"
    ).item()
    samples = int(
        fabric.all_reduce(
            torch.tensor(float(local_samples), device=device), reduce_op="sum"
        ).item()
    )
    accepted = fabric.all_reduce(
        torch.tensor(accepted_local, device=device), reduce_op="sum"
    ).item()
    proposed = fabric.all_reduce(
        torch.tensor(proposed_local, device=device), reduce_op="sum"
    ).item()
    accepted_pos2plus = fabric.all_reduce(
        torch.tensor(accepted_pos2plus_local, device=device), reduce_op="sum"
    ).item()
    proposed_pos2plus_all_drafted = fabric.all_reduce(
        torch.tensor(proposed_pos2plus_all_drafted_local, device=device),
        reduce_op="sum",
    ).item()
    target_samples = fabric.all_reduce(
        torch.tensor(target_samples_local, device=device), reduce_op="sum"
    ).item()
    resamples = fabric.all_reduce(
        torch.tensor(resamples_local, device=device), reduce_op="sum"
    ).item()
    pos_proposed = [
        fabric.all_reduce(torch.tensor(v, device=device), reduce_op="sum").item()
        for v in pos_proposed_local
    ]
    pos_accepted = [
        fabric.all_reduce(torch.tensor(v, device=device), reduce_op="sum").item()
        for v in pos_accepted_local
    ]

    total_new_tokens = float(samples * max_new_tokens)
    ar_tps = total_new_tokens / max(ar_seconds, 1e-8)
    spec_tps = total_new_tokens / max(spec_seconds, 1e-8)
    speedup = spec_tps / max(ar_tps, 1e-8)
    # Pos-2+ only; excludes position 1 (next-token prediction).
    acceptance_rate = accepted_pos2plus / max(proposed_pos2plus_all_drafted, 1e-8)
    # A speculative step ends in either target sampling or resampling.
    total_spec_steps = target_samples + resamples
    avg_accepted_tokens_per_step = accepted_pos2plus / max(total_spec_steps, 1e-8)

    logs: Dict[str, float] = {
        "spec_eval/fineweb_val/samples": float(samples),
        "spec_eval/fineweb_val/ar_tokens_per_sec": float(ar_tps),
        "spec_eval/fineweb_val/spec_tokens_per_sec": float(spec_tps),
        "spec_eval/fineweb_val/speedup": float(speedup),
        # Pos-2+ only; excludes position 1 (next-token prediction).
        "spec_eval/fineweb_val/acceptance_rate": float(acceptance_rate),
        "spec_eval/fineweb_val/avg_accepted_tokens_per_step": float(
            avg_accepted_tokens_per_step
        ),
        "spec_eval/fineweb_val/proposed_tokens": float(proposed),
        "spec_eval/fineweb_val/accepted_tokens": float(accepted),
    }
    for pos in range(1, gamma):
        pos_idx = pos + 1
        proposed_pos = float(pos_proposed[pos])
        accepted_pos = float(pos_accepted[pos])
        logs[f"spec_eval/fineweb_val/proposed_tokens_pos_{pos_idx}"] = proposed_pos
        logs[f"spec_eval/fineweb_val/accepted_tokens_pos_{pos_idx}"] = accepted_pos
        logs[f"spec_eval/fineweb_val/acceptance_rate_pos_{pos_idx}"] = (
            accepted_pos / proposed_pos if proposed_pos > 0.0 else 0.0
        )

    return logs


def collect_speculative_prompts_from_val_dataloader(
    val_dataloader: Any,
    trainer_config: Any,
    model_config: Any,
    data_config: Any,
    world_size: int,
    prepare_batch_func: Optional[Callable] = None,
) -> list[list[int]]:
    """
    Collect prompt slices from validation dataloader for speculative eval.
    """
    spec_eval_num_samples = int(trainer_config.get("spec_eval_num_samples", 0))
    per_rank_spec_samples = (
        (spec_eval_num_samples + max(1, world_size) - 1) // max(1, world_size)
        if spec_eval_num_samples > 0
        else 0
    )
    existing_prompts: list[list[int]] = []
    if per_rank_spec_samples <= 0 or len(existing_prompts) >= per_rank_spec_samples:
        return existing_prompts

    spec_eval_prompt_tokens = int(
        trainer_config.get("spec_eval_prompt_tokens", model_config.context_length)
    )
    spec_eval_max_new_tokens = int(trainer_config.get("spec_eval_max_new_tokens", 64))
    max_prompt_tokens = max(
        1,
        min(
            spec_eval_prompt_tokens,
            int(model_config.block_size) - max(1, spec_eval_max_new_tokens),
        ),
    )

    gradient_accum_steps = int(getattr(data_config, "gradient_accum_steps", 1))
    micro_batch_size = int(getattr(data_config, "micro_batch_size", 1))

    for batch in val_dataloader:
        if prepare_batch_func is not None:
            batch = prepare_batch_func(batch)
        for accum_step in range(gradient_accum_steps):
            start_idx = accum_step * micro_batch_size
            end_idx = (accum_step + 1) * micro_batch_size
            sub_batch = batch[start_idx:end_idx]
            for row in sub_batch:
                row_list = row.tolist()
                if len(row_list) < max_prompt_tokens:
                    continue
                existing_prompts.append(row_list[:max_prompt_tokens])
                if len(existing_prompts) >= per_rank_spec_samples:
                    return existing_prompts
        if len(existing_prompts) >= per_rank_spec_samples:
            break

    return existing_prompts
