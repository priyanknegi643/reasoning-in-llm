#!/usr/bin/env python
"""
Benchmark speculative decoding for NextLat-style checkpoints.

This script compares baseline autoregressive decoding vs speculative decoding
across:
  - Wikipedia prompts,
  - Books prompts,
  - Code prompts,
  - Math prompts,
  - FineWeb-Edu validation prompts from local data.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import lightning as L
import torch
import torch.distributed as dist
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer

# Allow running this script from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core_train import initialize_model
from data.fineweb import FineWebDataModule
from models.model_speculative import SpeculativeModel
from utils.speculative_sampling import (
    normalize_logits,
    sample_from_probs,
    speculative_decode_v2,
)

try:
    import wandb
except Exception:  # pragma: no cover - script still works without wandb
    wandb = None

try:
    from botocore import UNSIGNED
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import ClientError, NoCredentialsError
except Exception:  # pragma: no cover - botocore is optional until stack-edu is used
    UNSIGNED = None
    BotoConfig = None
    ClientError = Exception
    NoCredentialsError = Exception

_STACK_EDU_BUCKET = "softwareheritage"
_STACK_EDU_S3_CLIENT = None

CKPT_RE = re.compile(r"ckpt_iter_(\d+)(?:_loss_[^.]*)?\.pt$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoints with speculative decoding benchmarks."
    )
    parser.add_argument(
        "--train_config",
        required=True,
        help="Training config used for this run.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Checkpoint directory containing ckpt_iter_*.pt files.",
    )
    parser.add_argument(
        "--output_dir",
        default="output/speculative_eval",
        help="Directory for per-checkpoint and aggregate JSON outputs.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda"],
        help="Evaluation device.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        default="gpt2",
        help="Tokenizer for text datasets (Pile/TinyStories).",
    )
    parser.add_argument(
        "--num_samples_per_dataset",
        type=int,
        default=128,
        help="Number of prompts per dataset benchmark.",
    )
    parser.add_argument(
        "--prompt_tokens",
        type=int,
        default=128,
        help="Prompt length (tokens) for each benchmark sample.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Number of new tokens to generate per sample.",
    )
    parser.add_argument(
        "--gamma",
        type=int,
        default=4,
        help="Speculative decoding gamma.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1,
        help="Top-k for sampling (1 is greedy-like).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Optional top-p for sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--wiki_dataset",
        default="wikimedia/wikipedia",
        help="HuggingFace dataset for Wikipedia prompts.",
    )
    parser.add_argument(
        "--wiki_dataset_config",
        default="20231101.en",
        help="Optional dataset config/name for wiki dataset.",
    )
    parser.add_argument(
        "--wiki_split",
        default="train",
        help="Split for Wikipedia dataset.",
    )
    parser.add_argument(
        "--books_dataset",
        default="lucadiliello/bookcorpusopen",
        help="HuggingFace dataset for books prompts.",
    )
    parser.add_argument(
        "--books_dataset_config",
        default=None,
        help="Optional dataset config/name for books dataset.",
    )
    parser.add_argument(
        "--books_split",
        default="train",
        help="Split for books dataset.",
    )
    parser.add_argument(
        "--code_dataset",
        default="HuggingFaceTB/stack-edu",
        help="HuggingFace dataset for code prompts.",
    )
    parser.add_argument(
        "--code_dataset_config",
        default="Python",
        help="Optional dataset config/name for code dataset.",
    )
    parser.add_argument(
        "--code_split",
        default="train",
        help="Split for code dataset.",
    )
    parser.add_argument(
        "--math_dataset",
        default="open-web-math/open-web-math",
        help="HuggingFace dataset for math prompts.",
    )
    parser.add_argument(
        "--math_dataset_config",
        default=None,
        help="Optional dataset config/name for math dataset.",
    )
    parser.add_argument(
        "--math_split",
        default="train",
        help="Split for math dataset.",
    )
    parser.add_argument(
        "--wandb_project",
        default=None,
        help="Optional W&B project for logging speculative eval metrics.",
    )
    parser.add_argument(
        "--wandb_run_name",
        default=None,
        help="Optional W&B run name.",
    )
    parser.add_argument(
        "--wandb_tags",
        default="",
        help="Comma-separated W&B tags.",
    )
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Disable W&B logging even if project is set.",
    )
    parser.add_argument(
        "--last_checkpoint_only",
        action="store_true",
        help="If set, evaluate only the latest checkpoint.",
    )
    return parser.parse_args()


def _extract_step(path: Path) -> int:
    match = CKPT_RE.search(path.name)
    if match is None:
        raise ValueError(f"Checkpoint does not match expected format: {path}")
    return int(match.group(1))


def _distributed_context() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def _shard_checkpoints(
    checkpoints: list[Path], rank: int, world_size: int
) -> list[Path]:
    if world_size <= 1:
        return checkpoints
    return [ckpt for i, ckpt in enumerate(checkpoints) if i % world_size == rank]


def _forward_logits(model_wrapper: Any, input_ids: torch.Tensor) -> torch.Tensor:
    logits = model_wrapper.model(input_ids)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise RuntimeError(
            f"Expected logits [B, T, V], got {type(logits)} with shape "
            f"{getattr(logits, 'shape', None)}"
        )
    return logits


def _normalize_sequence_logits(
    logits_bt_v: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
) -> torch.Tensor:
    probs: list[torch.Tensor] = []
    for t in range(logits_bt_v.size(1)):
        probs.append(
            normalize_logits(
                logits_bt_v[:, t, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        )
    return torch.stack(probs, dim=1)


def _build_fallback_propose_fn(
    model_wrapper: Any,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
) -> Callable[[torch.Tensor, int], Tuple[torch.Tensor, List[torch.Tensor]]]:
    def propose_fn(seq: torch.Tensor, steps_to_propose: int):
        cur = seq
        drafted: list[torch.Tensor] = []
        q_probs_steps: list[torch.Tensor] = []
        for _ in range(steps_to_propose):
            logits = _forward_logits(model_wrapper, cur)
            q_probs = normalize_logits(
                logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
            )
            tok = sample_from_probs(q_probs)
            drafted.append(tok)
            q_probs_steps.append(q_probs)
            cur = torch.cat((cur, tok), dim=1)
        return torch.cat(drafted, dim=1), q_probs_steps

    return propose_fn


def _build_fallback_target_probs_fn(
    model_wrapper: Any,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
) -> Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    def target_probs_fn(
        seq: torch.Tensor,
        draft_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        full = torch.cat((seq, draft_tokens), dim=1)
        logits = _forward_logits(model_wrapper, full)  # [B, T+S, V]
        prefix_len = seq.size(1)
        steps = draft_tokens.size(1)
        step_logits = logits[:, prefix_len - 1 : prefix_len - 1 + steps, :]
        next_logits = logits[:, prefix_len - 1 + steps, :]
        p_probs_steps = _normalize_sequence_logits(
            step_logits, temperature=temperature, top_k=top_k, top_p=top_p
        )
        p_next_probs = normalize_logits(
            next_logits, temperature=temperature, top_k=top_k, top_p=top_p
        )
        return p_probs_steps, p_next_probs

    return target_probs_fn


def _collect_text_prompts(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    tokenizer: Any,
    prompt_tokens: int,
    num_samples: int,
) -> list[list[int]]:
    if dataset_config:
        ds = load_dataset(
            dataset_name, name=dataset_config, split=split, streaming=True
        )
    else:
        ds = load_dataset(dataset_name, split=split, streaming=True)
    prompts: list[list[int]] = []

    def _get_stack_edu_s3_client():
        global _STACK_EDU_S3_CLIENT
        if _STACK_EDU_S3_CLIENT is None:
            try:
                import boto3
            except Exception as exc:  # pragma: no cover - env dependent
                raise RuntimeError(
                    "Stack-Edu rows require blob content download, but boto3 is not "
                    "installed. Install boto3/botocore or use a dataset with text/content."
                ) from exc
            # Stack-Edu content blobs are publicly readable; use anonymous
            # unsigned requests so AWS credentials are not required.
            if BotoConfig is not None and UNSIGNED is not None:
                _STACK_EDU_S3_CLIENT = boto3.client(
                    "s3", config=BotoConfig(signature_version=UNSIGNED)
                )
            else:
                _STACK_EDU_S3_CLIENT = boto3.client("s3")
        return _STACK_EDU_S3_CLIENT

    def _download_stack_edu_blob_text(blob_id: str) -> Optional[str]:
        if not blob_id:
            return None
        key = f"content/{blob_id}"
        s3_client = _get_stack_edu_s3_client()
        try:
            obj = s3_client.get_object(Bucket=_STACK_EDU_BUCKET, Key=key)
            with gzip.GzipFile(fileobj=obj["Body"]) as fin:
                return fin.read().decode("utf-8", errors="ignore")
        except NoCredentialsError:  # pragma: no cover - env dependent
            return None
        except ClientError as exc:  # pragma: no cover - network/data dependent
            err_code = ""
            try:
                err_code = str(exc.response.get("Error", {}).get("Code", ""))
            except Exception:
                err_code = ""
            if err_code == "NoSuchKey":
                return None
            return None

    for row in ds:
        text: Optional[str] = None
        if isinstance(row, dict):
            for key in ("text", "content", "story", "prompt", "raw_content"):
                value = row.get(key)
                if isinstance(value, str) and value.strip():
                    text = value
                    break
            # Stack-Edu stores blob identifiers, not file contents.
            if text is None and isinstance(row.get("blob_id"), str):
                text = _download_stack_edu_blob_text(row["blob_id"])
            if text is None:
                for value in row.values():
                    if isinstance(value, str) and value.strip():
                        text = value
                        break
        if not text:
            continue
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < prompt_tokens:
            continue
        prompts.append(token_ids[:prompt_tokens])
        if len(prompts) >= num_samples:
            break
    return prompts


def _collect_domain_prompts(
    domain_name: str,
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    tokenizer: Any,
    prompt_tokens: int,
    num_samples: int,
) -> list[list[int]]:
    prompts = _collect_text_prompts(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        tokenizer=tokenizer,
        prompt_tokens=prompt_tokens,
        num_samples=num_samples,
    )
    print(
        f"[{domain_name}] loaded {len(prompts)} prompts from "
        f"{dataset_name}"
        + (f" ({dataset_config})" if dataset_config else "")
        + f":{split}"
    )
    return prompts


def _collect_wiki_prompts(
    args: argparse.Namespace,
    tokenizer: Any,
) -> list[list[int]]:
    return _collect_domain_prompts(
        domain_name="wiki",
        dataset_name=args.wiki_dataset,
        dataset_config=args.wiki_dataset_config,
        split=args.wiki_split,
        tokenizer=tokenizer,
        prompt_tokens=args.prompt_tokens,
        num_samples=args.num_samples_per_dataset,
    )


def _collect_books_prompts(
    args: argparse.Namespace,
    tokenizer: Any,
) -> list[list[int]]:
    return _collect_domain_prompts(
        domain_name="books",
        dataset_name=args.books_dataset,
        dataset_config=args.books_dataset_config,
        split=args.books_split,
        tokenizer=tokenizer,
        prompt_tokens=args.prompt_tokens,
        num_samples=args.num_samples_per_dataset,
    )


def _collect_code_prompts(
    args: argparse.Namespace,
    tokenizer: Any,
) -> list[list[int]]:
    return _collect_domain_prompts(
        domain_name="code",
        dataset_name=args.code_dataset,
        dataset_config=args.code_dataset_config,
        split=args.code_split,
        tokenizer=tokenizer,
        prompt_tokens=args.prompt_tokens,
        num_samples=args.num_samples_per_dataset,
    )


def _collect_math_prompts(
    args: argparse.Namespace,
    tokenizer: Any,
) -> list[list[int]]:
    return _collect_domain_prompts(
        domain_name="math",
        dataset_name=args.math_dataset,
        dataset_config=args.math_dataset_config,
        split=args.math_split,
        tokenizer=tokenizer,
        prompt_tokens=args.prompt_tokens,
        num_samples=args.num_samples_per_dataset,
    )


def _collect_fineweb_val_prompts(
    fabric: L.Fabric,
    cfg: Any,
    prompt_tokens: int,
    num_samples: int,
) -> list[list[int]]:
    cfg_local = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if "device_batch_size" not in cfg_local.data:
        cfg_local.data.device_batch_size = 1
    if "num_workers" not in cfg_local.data:
        cfg_local.data.num_workers = 0
    datamodule = FineWebDataModule(fabric, cfg_local)
    val_loader = datamodule.val_dataloader()
    if val_loader is None:
        return []

    prompts: list[list[int]] = []
    for batch in val_loader:
        input_ids = batch["input_ids"] if isinstance(batch, dict) else batch
        for row in input_ids:
            row_list = row.tolist()
            if len(row_list) < prompt_tokens:
                continue
            prompts.append(row_list[:prompt_tokens])
            if len(prompts) >= num_samples:
                return prompts
    return prompts


@torch.inference_mode()
def _evaluate_dataset(
    model_wrapper: Any,
    prompts: Sequence[list[int]],
    *,
    device: torch.device,
    rank: int,
    world_size: int,
    max_new_tokens: int,
    gamma: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    propose_fn: Callable[[torch.Tensor, int], Tuple[torch.Tensor, List[torch.Tensor]]],
    target_probs_fn: Callable[
        [torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
    ],
) -> dict[str, float]:
    eval_prompts = (
        [p for i, p in enumerate(prompts) if i % max(1, world_size) == rank]
        if world_size > 1
        else list(prompts)
    )
    total_samples = len(eval_prompts)
    pos_proposed = {i: 0.0 for i in range(2, gamma + 1)}
    pos_accepted = {i: 0.0 for i in range(2, gamma + 1)}
    if total_samples == 0 and not (world_size > 1):
        empty_result = {
            "samples": 0.0,
            "ar_tokens_per_sec": 0.0,
            "spec_tokens_per_sec": 0.0,
            "speedup": 0.0,
            # Pos-2+ only; excludes position 1 (next-token prediction).
            "acceptance_rate": 0.0,
            "avg_accepted_tokens_per_step": 0.0,
            "accepted_tokens": 0.0,
            "proposed_tokens": 0.0,
            "rejected_tokens": 0.0,
            "target_samples": 0.0,
            "resamples": 0.0,
        }
        for pos in range(2, gamma + 1):
            empty_result[f"proposed_tokens_pos_{pos}"] = 0.0
            empty_result[f"accepted_tokens_pos_{pos}"] = 0.0
            empty_result[f"acceptance_rate_pos_{pos}"] = 0.0
        return empty_result

    ar_seconds = 0.0
    spec_seconds = 0.0
    total_new_tokens = float(total_samples * max_new_tokens)
    agg_stats = {
        "accepted_tokens": 0.0,
        "proposed_tokens": 0.0,
        "rejected_tokens": 0.0,
        "target_samples": 0.0,
        "resamples": 0.0,
        "accepted_tokens_pos2plus": 0.0,
        "proposed_tokens_pos2plus_all_drafted": 0.0,
    }

    for prompt in eval_prompts:
        idx = torch.tensor([prompt], dtype=torch.long, device=device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        _ = model_wrapper.generate(
            idx.clone(),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ar_seconds += time.perf_counter() - t0

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        _, stats = speculative_decode_v2(
            idx=idx.clone(),
            max_new_tokens=max_new_tokens,
            gamma=gamma,
            propose_fn=propose_fn,
            target_probs_fn=target_probs_fn,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        spec_seconds += time.perf_counter() - t1
        for key in agg_stats:
            agg_stats[key] += float(stats.get(key, 0.0))
        for pos in range(2, gamma + 1):
            pos_proposed[pos] += float(stats.get(f"proposed_tokens_pos_{pos}", 0.0))
            pos_accepted[pos] += float(stats.get(f"accepted_tokens_pos_{pos}", 0.0))

    if world_size > 1 and dist.is_initialized():
        scalar_buf = torch.tensor(
            [
                float(total_samples),
                total_new_tokens,
                ar_seconds,
                spec_seconds,
                agg_stats["accepted_tokens"],
                agg_stats["proposed_tokens"],
                agg_stats["rejected_tokens"],
                agg_stats["target_samples"],
                agg_stats["resamples"],
                agg_stats["accepted_tokens_pos2plus"],
                agg_stats["proposed_tokens_pos2plus_all_drafted"],
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(scalar_buf, op=dist.ReduceOp.SUM)
        total_samples = int(scalar_buf[0].item())
        total_new_tokens = float(scalar_buf[1].item())
        ar_seconds = float(scalar_buf[2].item())
        spec_seconds = float(scalar_buf[3].item())
        agg_stats["accepted_tokens"] = float(scalar_buf[4].item())
        agg_stats["proposed_tokens"] = float(scalar_buf[5].item())
        agg_stats["rejected_tokens"] = float(scalar_buf[6].item())
        agg_stats["target_samples"] = float(scalar_buf[7].item())
        agg_stats["resamples"] = float(scalar_buf[8].item())
        agg_stats["accepted_tokens_pos2plus"] = float(scalar_buf[9].item())
        agg_stats["proposed_tokens_pos2plus_all_drafted"] = float(scalar_buf[10].item())

        pos_prop_buf = torch.tensor(
            [pos_proposed[pos] for pos in range(2, gamma + 1)],
            dtype=torch.float64,
            device=device,
        )
        pos_acc_buf = torch.tensor(
            [pos_accepted[pos] for pos in range(2, gamma + 1)],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(pos_prop_buf, op=dist.ReduceOp.SUM)
        dist.all_reduce(pos_acc_buf, op=dist.ReduceOp.SUM)
        for i, pos in enumerate(range(2, gamma + 1)):
            pos_proposed[pos] = float(pos_prop_buf[i].item())
            pos_accepted[pos] = float(pos_acc_buf[i].item())

    if total_samples == 0:
        empty_result = {
            "samples": 0.0,
            "ar_tokens_per_sec": 0.0,
            "spec_tokens_per_sec": 0.0,
            "speedup": 0.0,
            # Pos-2+ only; excludes position 1 (next-token prediction).
            "acceptance_rate": 0.0,
            "avg_accepted_tokens_per_step": 0.0,
            "accepted_tokens": 0.0,
            "proposed_tokens": 0.0,
            "rejected_tokens": 0.0,
            "target_samples": 0.0,
            "resamples": 0.0,
        }
        for pos in range(2, gamma + 1):
            empty_result[f"proposed_tokens_pos_{pos}"] = 0.0
            empty_result[f"accepted_tokens_pos_{pos}"] = 0.0
            empty_result[f"acceptance_rate_pos_{pos}"] = 0.0
        return empty_result

    ar_tps = total_new_tokens / max(ar_seconds, 1e-8)
    spec_tps = total_new_tokens / max(spec_seconds, 1e-8)
    speedup = spec_tps / max(ar_tps, 1e-8)
    # Pos-2+ only; excludes position 1 (next-token prediction).
    accepted_pos2plus = float(agg_stats["accepted_tokens_pos2plus"])
    proposed_pos2plus_all_drafted = float(
        agg_stats["proposed_tokens_pos2plus_all_drafted"]
    )
    acceptance_rate = accepted_pos2plus / max(proposed_pos2plus_all_drafted, 1e-8)
    total_spec_steps = float(agg_stats["target_samples"] + agg_stats["resamples"])
    avg_accepted_tokens_per_step = accepted_pos2plus / max(total_spec_steps, 1e-8)

    result = {
        "samples": float(total_samples),
        "ar_tokens_per_sec": float(ar_tps),
        "spec_tokens_per_sec": float(spec_tps),
        "speedup": float(speedup),
        "acceptance_rate": float(acceptance_rate),
        "avg_accepted_tokens_per_step": float(avg_accepted_tokens_per_step),
    }
    result.update({k: float(v) for k, v in agg_stats.items()})
    for pos in range(2, gamma + 1):
        proposed = float(pos_proposed[pos])
        accepted = float(pos_accepted[pos])
        result[f"proposed_tokens_pos_{pos}"] = proposed
        result[f"accepted_tokens_pos_{pos}"] = accepted
        result[f"acceptance_rate_pos_{pos}"] = (
            accepted / proposed if proposed > 0.0 else 0.0
        )
    return result


def _load_eval_config(train_config: Path, checkpoint_dir: Path) -> Any:
    default_cfg = OmegaConf.load("defaults.yaml")
    user_cfg = OmegaConf.load(train_config)
    cfg = OmegaConf.merge(default_cfg, user_cfg)
    materialized = checkpoint_dir / "materialized_config.yaml"
    if materialized.exists():
        cfg = OmegaConf.merge(default_cfg, OmegaConf.load(materialized))
    cfg.trainer.compile = False
    cfg.trainer.log_to_wandb = False
    cfg.trainer.log_to_file = False
    return cfg


def _wandb_metrics_from_result(result: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    datasets = result.get("datasets", {})
    speedups: list[float] = []
    acceptances: list[float] = []
    for ds_name, ds_metrics in datasets.items():
        if not isinstance(ds_metrics, dict):
            continue
        for metric_name, value in ds_metrics.items():
            if isinstance(value, (int, float)):
                metrics[f"spec_eval/{ds_name}/{metric_name}"] = float(value)
        if isinstance(ds_metrics.get("speedup"), (int, float)):
            speedups.append(float(ds_metrics["speedup"]))
        if isinstance(ds_metrics.get("acceptance_rate"), (int, float)):
            acceptances.append(float(ds_metrics["acceptance_rate"]))
    if speedups:
        metrics["spec_eval/mean_speedup"] = float(sum(speedups) / len(speedups))
    if acceptances:
        metrics["spec_eval/mean_acceptance_rate"] = float(
            sum(acceptances) / len(acceptances)
        )
    return metrics


def main() -> None:
    args = _parse_args()
    rank, world_size, local_rank = _distributed_context()
    train_config = Path(args.train_config)
    if not train_config.exists():
        raise FileNotFoundError(f"train_config not found: {train_config}")

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint_dir not found: {checkpoint_dir}")

    checkpoints = sorted(checkpoint_dir.glob("ckpt_iter_*.pt"), key=_extract_step)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    if args.last_checkpoint_only:
        checkpoints = [checkpoints[-1]]
    local_checkpoints = checkpoints

    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        args.device = "cpu"
    if args.device == "cuda" and world_size > 1:
        torch.cuda.set_device(local_rank)
    device = (
        torch.device(f"cuda:{local_rank}")
        if args.device == "cuda" and world_size > 1
        else torch.device(args.device)
    )

    cfg = _load_eval_config(train_config, checkpoint_dir)
    if args.prompt_tokens + args.max_new_tokens > int(cfg.model.block_size):
        raise ValueError(
            "prompt_tokens + max_new_tokens exceeds model block_size. "
            "Reduce prompt/new tokens for this benchmark."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    if getattr(tokenizer, "eos_token_id", None) is None:
        raise ValueError("Tokenizer must provide eos_token_id.")

    accelerator = "cuda" if device.type == "cuda" else "cpu"
    if world_size > 1:
        fabric = L.Fabric(accelerator=accelerator)
    else:
        fabric = L.Fabric(accelerator=accelerator, devices=1)
    fabric.launch()

    print(
        f"[rank {rank}] Collecting prompts..."
        + (f" on cuda:{local_rank}" if device.type == "cuda" else "")
    )
    wiki_prompts = _collect_wiki_prompts(args, tokenizer)
    books_prompts = _collect_books_prompts(args, tokenizer)
    code_prompts = _collect_code_prompts(args, tokenizer)
    math_prompts = _collect_math_prompts(args, tokenizer)
    fineweb_prompts = _collect_fineweb_val_prompts(
        fabric=fabric,
        cfg=cfg,
        prompt_tokens=args.prompt_tokens,
        num_samples=args.num_samples_per_dataset,
    )
    print(
        f"[rank {rank}] Collected prompts: wiki={len(wiki_prompts)} "
        f"books={len(books_prompts)} code={len(code_prompts)} "
        f"math={len(math_prompts)} fineweb_val={len(fineweb_prompts)}"
    )
    print(f"[rank {rank}] assigned {len(local_checkpoints)} checkpoint(s)")
    if world_size > 1:
        print(f"[rank {rank}] distributed_per_checkpoint enabled")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate: list[dict[str, Any]] = []
    wandb_run = None
    wandb_project = args.wandb_project or getattr(cfg.trainer, "wandb_project", None)
    enable_wandb_for_rank = not (world_size > 1 and rank != 0)
    if not args.disable_wandb and wandb_project and enable_wandb_for_rank:
        if wandb is None:
            print(
                "W&B logging requested but `wandb` is unavailable. "
                "Continuing without W&B."
            )
        else:
            # Prevent accidental resume
            os.environ.pop("WANDB_RUN_ID", None)
            os.environ.pop("WANDB_RESUME", None)
            tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
            run_name = args.wandb_run_name or f"speculative-eval-{checkpoint_dir.name}"
            run_id = f"spec-{int(time.time() * 1000)}-r{rank}-p{os.getpid()}"
            wandb_run = wandb.init(
                project=wandb_project,
                name=run_name,
                tags=tags,
                id=run_id,
                resume="never",
                reinit=True,
                job_type="speculative-eval",
                config={
                    "train_config": str(train_config),
                    "checkpoint_dir": str(checkpoint_dir),
                    "gamma": args.gamma,
                    "prompt_tokens": args.prompt_tokens,
                    "max_new_tokens": args.max_new_tokens,
                    "num_samples_per_dataset": args.num_samples_per_dataset,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                },
            )

    init_tok = SimpleNamespace(
        eos_token_id=int(tokenizer.eos_token_id),
        convert_tokens_to_ids=getattr(
            tokenizer, "convert_tokens_to_ids", lambda x: None
        ),
    )

    for ckpt in local_checkpoints:
        step = _extract_step(ckpt)
        print(f"\n[rank {rank}] === Evaluating checkpoint step={step}: {ckpt.name} ===")
        model_wrapper = initialize_model(
            fabric=fabric,
            config=cfg,
            tokenizer=init_tok,
            initialize_optimizer=False,
            checkpoint_path=str(ckpt),
        )
        model_wrapper.eval()

        if not isinstance(model_wrapper, SpeculativeModel):
            print(
                f"Skipping checkpoint step={step}: model does not implement "
                "SpeculativeModel interface."
            )
            del model_wrapper
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        propose_fn = model_wrapper.build_speculative_propose_fn(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        target_probs_fn = model_wrapper.build_speculative_target_probs_fn(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

        datasets = {
            "wiki": wiki_prompts,
            "books": books_prompts,
            "code": code_prompts,
            "math": math_prompts,
            "fineweb_val": fineweb_prompts,
        }
        ds_metrics: dict[str, dict[str, float]] = {}
        for ds_name, prompts in datasets.items():
            metrics = _evaluate_dataset(
                model_wrapper=model_wrapper,
                prompts=prompts,
                device=device,
                rank=rank,
                world_size=world_size,
                max_new_tokens=args.max_new_tokens,
                gamma=args.gamma,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                propose_fn=propose_fn,
                target_probs_fn=target_probs_fn,
            )
            ds_metrics[ds_name] = metrics
            print(
                f"[{ds_name}] speedup={metrics['speedup']:.3f} "
                f"acceptance={metrics['acceptance_rate']:.3f} "
                f"ar_tps={metrics['ar_tokens_per_sec']:.2f} "
                f"spec_tps={metrics['spec_tokens_per_sec']:.2f}"
            )

        result = {
            "checkpoint": str(ckpt),
            "step": step,
            "model_type": (
                "nextlat"
                if bool(getattr(cfg, "use_nextlat", False))
                else (
                    "mtp_gloeckle"
                    if bool(getattr(cfg, "use_mtp_gloeckle", False))
                    else (
                        "mtp_jtp" if bool(getattr(cfg, "use_mtp_jtp", False)) else "gpt"
                    )
                )
            ),
            "settings": {
                "gamma": args.gamma,
                "prompt_tokens": args.prompt_tokens,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "num_samples_per_dataset": args.num_samples_per_dataset,
            },
            "datasets": ds_metrics,
        }
        aggregate.append(result)
        out_path = output_dir / f"{ckpt.stem}_speculative_eval.json"
        should_write_rank = not (world_size > 1 and rank != 0)
        if should_write_rank:
            out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"Saved: {out_path}")
        if wandb_run is not None:
            wandb_metrics = _wandb_metrics_from_result(result)
            if wandb_metrics:
                wandb_run.log(wandb_metrics, step=step)
                print(f"Logged {len(wandb_metrics)} metric(s) to W&B at step={step}")

        del model_wrapper
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not (world_size > 1 and rank != 0):
        aggregate_filename = (
            "speculative_eval_all_checkpoints.json"
            if world_size <= 1
            else f"speculative_eval_all_checkpoints_rank{rank}.json"
        )
        aggregate_path = output_dir / aggregate_filename
        aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"\n[rank {rank}] Saved aggregate results: {aggregate_path}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
