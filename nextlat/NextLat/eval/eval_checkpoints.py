#!/usr/bin/env python
"""
Run lm-eval.py over saved checkpoints and log metrics to Weights & Biases.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Allow running this script from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from omegaconf import OmegaConf

try:
    import wandb
except Exception:  # pragma: no cover - script still works without wandb
    wandb = None


CKPT_RE = re.compile(r"ckpt_iter_(\d+)\.pt$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all checkpoints with lm-eval.py and log to W&B."
    )
    parser.add_argument(
        "--train_config",
        required=True,
        help="Training config used for the run (contains trainer.out_dir).",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated lm-eval tasks (e.g. mmlu,piqa,winogrande,arc_easy).",
    )
    parser.add_argument(
        "--eval_config",
        default=None,
        help="Optional shared eval yaml (tasks/device/tokenizer/wandb defaults).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Checkpoint directory containing ckpt_iter_*.pt files.",
    )
    parser.add_argument(
        "--lm_eval_script",
        default="eval/lm-eval.py",
        help="Path to lm-eval.py entry script.",
    )
    parser.add_argument(
        "--output_dir",
        default="output/lm_eval",
        help="Directory to save per-checkpoint JSON outputs.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        default=None,
        help="Tokenizer override passed to lm-eval.py.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device passed to lm-eval.py.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size passed to lm-eval.py.",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        help="Optional few-shot override for lm-eval.",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Optional sample limit for fast checks.",
    )
    parser.add_argument(
        "--wandb_project",
        default=None,
        help="W&B project for logging eval metrics.",
    )
    parser.add_argument(
        "--wandb_run_name",
        default=None,
        help="Optional W&B run name for this eval sweep.",
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


def _task_metrics_for_wandb(
    results_json: dict[str, Any], tasks: list[str]
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    results = results_json.get("results", {})
    acc_values: list[float] = []
    for task in tasks:
        task_metrics = results.get(task, {})
        if not isinstance(task_metrics, dict):
            continue
        for key, value in task_metrics.items():
            if isinstance(value, (int, float)):
                metrics[f"{task}/{key}"] = float(value)
        # Track task-level accuracy for an aggregate curve in W&B.
        for acc_key in (
            "acc_norm,none",
            "acc_norm",
            "accuracy_norm",
            "acc,none",
            "acc",
            "accuracy",
        ):
            acc_value = task_metrics.get(acc_key)
            if isinstance(acc_value, (int, float)):
                acc_values.append(float(acc_value))
                break
    summary = results_json.get("summary", {})
    if isinstance(summary, dict):
        avg = summary.get("average_accuracy")
        if isinstance(avg, (int, float)):
            metrics["summary/average_accuracy"] = float(avg)
    if "summary/average_accuracy" not in metrics and acc_values:
        metrics["summary/average_accuracy"] = sum(acc_values) / len(acc_values)
    return metrics


def _resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    cfg_data: dict[str, Any] = {}
    if args.eval_config is not None:
        eval_cfg_path = Path(args.eval_config)
        if not eval_cfg_path.exists():
            raise FileNotFoundError(f"eval_config not found: {eval_cfg_path}")
        loaded = OmegaConf.to_container(OmegaConf.load(eval_cfg_path), resolve=True)
        if isinstance(loaded, dict):
            cfg_data = loaded

    raw_tasks = args.tasks if args.tasks is not None else cfg_data.get("tasks")
    if isinstance(raw_tasks, str):
        tasks = [task.strip() for task in raw_tasks.split(",") if task.strip()]
    elif isinstance(raw_tasks, list):
        tasks = [str(task).strip() for task in raw_tasks if str(task).strip()]
    else:
        tasks = []

    tokenizer_name_or_path = (
        args.tokenizer_name_or_path
        if args.tokenizer_name_or_path is not None
        else cfg_data.get("tokenizer_name_or_path", "gpt2")
    )
    device = args.device if args.device is not None else cfg_data.get("device", "cuda")
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device '{device}'. Expected 'cpu' or 'cuda'.")

    batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(cfg_data.get("batch_size", 1))
    )
    num_fewshot = (
        args.num_fewshot
        if args.num_fewshot is not None
        else cfg_data.get("num_fewshot")
    )
    if num_fewshot is not None:
        num_fewshot = int(num_fewshot)
    limit = args.limit if args.limit is not None else cfg_data.get("limit")
    if limit is not None:
        limit = float(limit)

    wandb_project = (
        args.wandb_project
        if args.wandb_project is not None
        else cfg_data.get("wandb_project", "fineweb")
    )
    return {
        "tasks": tasks,
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "device": device,
        "batch_size": batch_size,
        "num_fewshot": num_fewshot,
        "limit": limit,
        "wandb_project": wandb_project,
    }


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


def main() -> None:
    args = _parse_args()
    settings = _resolve_settings(args)
    rank, world_size, local_rank = _distributed_context()

    train_config = Path(args.train_config)
    lm_eval_script = Path(args.lm_eval_script)
    if not train_config.exists():
        raise FileNotFoundError(f"train_config not found: {train_config}")
    if not lm_eval_script.exists():
        raise FileNotFoundError(f"lm_eval_script not found: {lm_eval_script}")

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint_dir not found: {checkpoint_dir}")

    eval_config = checkpoint_dir / "materialized_config.yaml"
    if not eval_config.exists():
        eval_config = train_config

    checkpoints = sorted(
        checkpoint_dir.glob("ckpt_iter_*.pt"),
        key=_extract_step,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    if args.last_checkpoint_only:
        checkpoints = [checkpoints[-1]]
    local_checkpoints = checkpoints

    tasks = settings["tasks"]
    if not tasks:
        raise ValueError("No tasks were provided.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run = None
    enable_wandb_for_rank = not (world_size > 1 and rank != 0)
    if wandb is not None and settings["wandb_project"] and enable_wandb_for_rank:
        run_name = args.wandb_run_name or f"lm-eval-{checkpoint_dir.name}"
        run = wandb.init(
            project=settings["wandb_project"], name=run_name, job_type="lm-eval"
        )

    if rank == 0:
        print(f"Checkpoint dir: {checkpoint_dir}")
        print(f"Eval config: {eval_config}")
        print(f"Found {len(checkpoints)} checkpoint(s)")
        if world_size > 1:
            print(f"Distributed eval: world_size={world_size}")
    print(
        f"[rank {rank}] assigned {len(local_checkpoints)} checkpoint(s)"
        + (f" on cuda:{local_rank}" if settings["device"] == "cuda" else "")
    )
    if world_size > 1:
        print(f"[rank {rank}] distributed_per_checkpoint enabled")

    for ckpt_path in local_checkpoints:
        step = _extract_step(ckpt_path)
        output_path = output_dir / f"{ckpt_path.stem}_{'_'.join(tasks)}.json"

        cmd = [
            sys.executable,
            str(lm_eval_script),
            "--config",
            str(eval_config),
            "--checkpoint_path",
            str(ckpt_path),
            "--tasks",
            ",".join(tasks),
            "--tokenizer_name_or_path",
            settings["tokenizer_name_or_path"],
            "--device",
            settings["device"],
            "--batch_size",
            str(settings["batch_size"]),
            "--output_path",
            str(output_path),
        ]
        if settings["num_fewshot"] is not None:
            cmd.extend(["--num_fewshot", str(settings["num_fewshot"])])
        if settings["limit"] is not None:
            cmd.extend(["--limit", str(settings["limit"])])

        print(f"\n[rank {rank}] === Evaluating step {step}: {ckpt_path.name} ===")
        # Keep torchrun-provided distributed env intact for child lm-eval.
        # Do not remap CUDA_VISIBLE_DEVICES per rank; child uses LOCAL_RANK.
        subprocess.run(cmd, check=True, env=os.environ.copy())

        if world_size > 1 and rank != 0:
            continue

        wait_secs = 60
        waited = 0.0
        while not output_path.exists() and waited < wait_secs:
            time.sleep(0.5)
            waited += 0.5
        if not output_path.exists():
            raise FileNotFoundError(f"Expected output not found: {output_path}")

        results_json = json.loads(output_path.read_text(encoding="utf-8"))
        metrics = _task_metrics_for_wandb(results_json, tasks)
        if run is not None and metrics:
            run.log(metrics, step=step)
            print(f"Logged {len(metrics)} metric(s) to W&B at step={step}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
