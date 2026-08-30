#!/usr/bin/env python
"""
Run EleutherAI lm-eval on NextLat custom model wrappers.

This script loads a NextLat-style checkpoint/config and evaluates via a custom
lm-eval adapter instead of HuggingFace PreTrainedModel APIs.
"""

from __future__ import annotations

import argparse
import json
import os
import datetime
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Generator
from typing import Any, Optional

import lightning as L
import torch
import torch.distributed as dist
from accelerate import Accelerator
from omegaconf import OmegaConf
from transformers import AutoTokenizer, PreTrainedTokenizerFast

# Allow running this script from any working directory/subprocess.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core_train import initialize_model

try:
    from lm_eval import simple_evaluate
    from lm_eval.api.model import LM
    from lm_eval.utils import get_rolling_token_windows as _get_rolling_token_windows
    from lm_eval.utils import make_disjoint_window as _make_disjoint_window
except ImportError as e:
    raise ImportError("lm-eval is required. Install with `pip install lm_eval`.") from e


@dataclass
class NextLatLMEvalConfig:
    config_path: str
    checkpoint_path: str
    tasks: list[str]
    output_path: Optional[str]
    device: str
    limit: Optional[float]
    num_fewshot: Optional[int]
    batch_size: int
    max_length: int
    max_gen_toks: int
    temperature: float
    top_k: Optional[int]
    tokenizer_name_or_path: Optional[str]
    trust_remote_code: bool
    progress_log_interval_sec: float
    progress_log_every_n_requests: int


def _load_tokenizer(cfg: Any, override_name_or_path: Optional[str] = None):
    tok_path = override_name_or_path
    if tok_path is None and "data" in cfg and "tokenizer_name_or_path" in cfg.data:
        tok_path = cfg.data.tokenizer_name_or_path

    if tok_path:
        return AutoTokenizer.from_pretrained(tok_path)

    raise ValueError(
        "Could not determine tokenizer. Pass --tokenizer_name_or_path or set "
        "data.tokenizer_name_or_path in config."
    )


class NextLatLMWrapper(LM):
    """
    Minimal lm-eval LM adapter for NextLat custom wrappers.
    Supports GPT/NextLat/JTP/MTP-like wrappers with `model(...)` + `generate(...)`.
    """

    def __init__(
        self,
        model_wrapper: Any,
        tokenizer: Any,
        *,
        device: torch.device,
        batch_size: int = 1,
        max_length: int = 1024,
        max_gen_toks: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = 1,
        progress_log_interval_sec: float = 30.0,
        progress_log_every_n_requests: int = 128,
    ):
        super().__init__()
        self.model_wrapper = model_wrapper
        self.tokenizer = tokenizer
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._max_gen_toks = max_gen_toks
        self._temperature = temperature
        self._top_k = top_k
        self._rank = 0
        self._world_size = 1
        self._progress_log_interval_sec = max(1.0, float(progress_log_interval_sec))
        self._progress_log_every_n_requests = max(1, int(progress_log_every_n_requests))

    def _maybe_log_progress(
        self,
        phase: str,
        done: int,
        total: int,
        start_time: float,
        *,
        force: bool = False,
    ) -> None:
        if self._rank != 0:
            return
        if done <= 0:
            return
        now = time.perf_counter()
        elapsed = now - start_time
        should_log = force
        if not should_log and (done % self._progress_log_every_n_requests == 0):
            should_log = True
        if not should_log and elapsed >= self._progress_log_interval_sec:
            should_log = True
        if not should_log:
            return
        # Reset timing window by mutating caller-owned start time through return convention.
        print(
            f"[rank {self._rank}] lm-eval {phase}: {done}/{total} requests "
            f"({100.0 * done / max(total, 1):.1f}%)"
        )

    @property
    def eot_token_id(self) -> int:
        return int(self.tokenizer.eos_token_id)

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        if hasattr(self.tokenizer, "encode"):
            try:
                return list(
                    self.tokenizer.encode(string, add_special_tokens=False, **kwargs)
                )
            except TypeError:
                return list(self.tokenizer.encode(string))
        raise TypeError("Tokenizer does not support encode().")

    def tok_decode(self, tokens: list[int]) -> str:
        if hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(tokens, skip_special_tokens=False)
        raise TypeError("Tokenizer does not support decode().")

    def _encode_loglikelihood_pair(
        self, context: str, continuation: str
    ) -> tuple[list[int], list[int]]:
        """
        Match lm_eval.api.model.TemplateLM._encode_pair for causal models: joint
        tokenization of context+continuation, continuation = suffix of whole encoding;
        trailing spaces stay on the continuation side (word-boundary alignment).
        """
        if context == "":
            continuation_enc = self.tok_encode(continuation, add_special_tokens=False)
            prefix_id = int(self.eot_token_id)
            if continuation_enc and int(continuation_enc[0]) == prefix_id:
                return continuation_enc[:1], continuation_enc[1:]
            return [prefix_id], continuation_enc
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole_enc = self.tok_encode(context + continuation, add_special_tokens=False)
        context_enc = self.tok_encode(context, add_special_tokens=False)
        return context_enc, whole_enc[len(context_enc) :]

    def _score_context_continuation(
        self, context_enc: list[int], continuation_enc: list[int]
    ) -> tuple[float, bool]:
        """
        Single forward, causal LM loglikelihood for encoded context/continuation,
        including left truncation when len(ctx)+len(cont) exceeds max_length+1
        (same windowing as the harness HFLM path).
        """
        if not continuation_enc:
            return 0.0, True
        if not context_enc:
            raise RuntimeError(
                "Internal error: empty context_enc in _score_context_continuation."
            )
        if len(continuation_enc) > self.max_length:
            raise RuntimeError(
                f"Continuation length {len(continuation_enc)} exceeds max_length={self.max_length}. "
                "Raise --max_length / model block_size."
            )
        total_length = len(context_enc) + len(continuation_enc)
        if total_length > self.max_length + 1 and self._rank == 0:
            print(
                f"[NextLatLMWrapper] loglikelihood: truncating "
                f"{total_length - self.max_length - 1} token(s) from the left "
                f"(combined len {total_length} > max_length+1={self.max_length + 1})."
            )
        inp_list = (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1]
        inp = torch.tensor([inp_list], dtype=torch.long, device=self.device)
        logits = self._forward_logits(inp)[0]
        logprobs = torch.log_softmax(logits, dim=-1)
        inplen = int(logits.shape[0])
        contlen = len(continuation_enc)
        padding_len_inp = inplen
        ctx_len = inplen + (logits.shape[0] - padding_len_inp)
        sl = logprobs[ctx_len - contlen : ctx_len]
        cont_t = torch.tensor(continuation_enc, dtype=torch.long, device=self.device)
        tok_lls = sl.gather(1, cont_t.unsqueeze(-1)).squeeze(-1)
        ll = float(tok_lls.sum().item())
        greedy = bool((sl.argmax(dim=-1) == cont_t).all().item())
        return ll, greedy

    def _forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Return logits [B, T, V] for the input sequence.
        """
        model = self.model_wrapper.model
        logits = model(input_ids)

        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise RuntimeError(
                "Model forward must return tensor logits with shape [B, T, V]. "
                "BST/Probe wrappers are currently unsupported in lm-eval.py."
            )
        return logits

    @torch.inference_mode()
    def loglikelihood(self, requests):
        results = []
        total = len(requests)
        progress_t0 = time.perf_counter()
        if self._rank == 0:
            print(f"[rank {self._rank}] lm-eval loglikelihood start: {total} requests")
        for idx, req in enumerate(requests, start=1):
            context, continuation = req.args
            if not continuation:
                results.append((0.0, True))
                continue
            context_enc, continuation_enc = self._encode_loglikelihood_pair(
                context, continuation
            )
            ll, greedy = self._score_context_continuation(context_enc, continuation_enc)
            results.append((ll, greedy))
            now = time.perf_counter()
            if (
                idx % self._progress_log_every_n_requests == 0
                or now - progress_t0 >= self._progress_log_interval_sec
                or idx == total
            ):
                self._maybe_log_progress(
                    "loglikelihood", idx, total, progress_t0, force=(idx == total)
                )
                progress_t0 = now
        return results

    @torch.inference_mode()
    def loglikelihood_rolling(self, requests):
        out = []
        total = len(requests)
        progress_t0 = time.perf_counter()
        if self._rank == 0:
            print(
                f"[rank {self._rank}] lm-eval loglikelihood_rolling start: "
                f"{total} requests"
            )
        for idx, req in enumerate(requests, start=1):
            (text,) = req.args
            toks = self.tok_encode(text, add_special_tokens=False)
            if len(toks) == 0:
                out.append(0.0)
                continue
            ll = 0.0
            for raw_window in _get_rolling_token_windows(
                toks,
                prefix_token=int(self.eot_token_id),
                max_seq_len=self.max_length,
                context_len=1,
            ):
                ctx_enc, pred_enc = _make_disjoint_window(raw_window)
                w_ll, _ = self._score_context_continuation(ctx_enc, pred_enc)
                ll += w_ll
            # One float per request; task code unpacks (loglikelihood,) from filtered resps.
            out.append(ll)
            now = time.perf_counter()
            if (
                idx % self._progress_log_every_n_requests == 0
                or now - progress_t0 >= self._progress_log_interval_sec
                or idx == total
            ):
                self._maybe_log_progress(
                    "loglikelihood_rolling",
                    idx,
                    total,
                    progress_t0,
                    force=(idx == total),
                )
                progress_t0 = now
        return out

    @torch.inference_mode()
    def generate_until(self, requests):
        generations = []
        total = len(requests)
        progress_t0 = time.perf_counter()
        if self._rank == 0:
            print(f"[rank {self._rank}] lm-eval generate_until start: {total} requests")
        for idx, req in enumerate(requests, start=1):
            context, gen_kwargs = req.args
            until = gen_kwargs.get("until", None)
            max_gen_req = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks))

            if isinstance(until, str):
                until_list = [until]
            elif until is None:
                until_list = []
            else:
                until_list = list(until)
            eos_text = self.tok_decode([self.eot_token_id])
            if not until_list and eos_text:
                until_list = [eos_text]

            if context:
                context_toks = self.tok_encode(context, add_special_tokens=False)
            else:
                context_toks = [self.eot_token_id]

            # Causal harness: reserve max_gen tokens; left-truncate prompt (HFLM behavior).
            max_gen = max_gen_req
            if max_gen >= self.max_length:
                max_gen = max(1, self.max_length - 1)
            max_ctx_len = max(1, self.max_length - max_gen)
            context_toks = context_toks[-max_ctx_len:]

            input_ids = torch.tensor(
                [context_toks], dtype=torch.long, device=self.device
            )

            generated = self.model_wrapper.generate(
                input_ids,
                max_new_tokens=max_gen,
                temperature=self._temperature,
                top_k=self._top_k,
            )
            new_tokens = generated[0, len(context_toks) :].tolist()
            text = self.tok_decode(new_tokens)

            if until_list:
                cut_idx = None
                for stop in until_list:
                    if not stop:
                        continue
                    pos = text.find(stop)
                    if pos != -1:
                        cut_idx = pos if cut_idx is None else min(cut_idx, pos)
                if cut_idx is not None:
                    text = text[:cut_idx]

            generations.append(text)
            now = time.perf_counter()
            if (
                idx % self._progress_log_every_n_requests == 0
                or now - progress_t0 >= self._progress_log_interval_sec
                or idx == total
            ):
                self._maybe_log_progress(
                    "generate_until", idx, total, progress_t0, force=(idx == total)
                )
                progress_t0 = now
        return generations


def _parse_args() -> NextLatLMEvalConfig:
    parser = argparse.ArgumentParser(
        description="Run lm-eval on NextLat custom model wrappers."
    )
    parser.add_argument("--config", required=True, help="Path to model config yaml.")
    parser.add_argument(
        "--checkpoint_path", required=True, help="Path to checkpoint .pt."
    )
    parser.add_argument("--tasks", required=True, help="Comma-separated lm-eval tasks.")
    parser.add_argument(
        "--output_path", default=None, help="Optional JSON file for full results."
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cpu", "cuda"], help="Evaluation device."
    )
    parser.add_argument("--limit", type=float, default=None, help="Task sample limit.")
    parser.add_argument(
        "--num_fewshot", type=int, default=None, help="Few-shot count override."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="lm-eval request batch size."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Max context length for adapter metadata.",
    )
    parser.add_argument(
        "--max_gen_toks", type=int, default=256, help="Default max generation tokens."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Generation temperature for generate_until.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1,
        help="Top-k for generate_until; use 1 for greedy-like decode.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        default=None,
        help="Optional tokenizer override (HF path or local folder).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow Hugging Face datasets with custom loading scripts "
            "(e.g., social_iqa). Use --no-trust_remote_code to disable."
        ),
    )
    parser.add_argument(
        "--progress_log_interval_sec",
        type=float,
        default=30.0,
        help="Print lm-eval request progress at least this often (seconds).",
    )
    parser.add_argument(
        "--progress_log_every_n_requests",
        type=int,
        default=128,
        help="Print lm-eval request progress every N requests.",
    )
    args, _ = parser.parse_known_args()

    return NextLatLMEvalConfig(
        config_path=args.config,
        checkpoint_path=args.checkpoint_path,
        tasks=[t.strip() for t in args.tasks.split(",") if t.strip()],
        output_path=args.output_path,
        device=args.device,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_gen_toks=args.max_gen_toks,
        temperature=args.temperature,
        top_k=args.top_k,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        trust_remote_code=bool(args.trust_remote_code),
        progress_log_interval_sec=args.progress_log_interval_sec,
        progress_log_every_n_requests=args.progress_log_every_n_requests,
    )


def _distributed_context() -> tuple[int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size


def _init_distributed_if_needed() -> tuple[int, int, bool]:
    rank, world_size = _distributed_context()
    initialized_here = False
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            timeout=datetime.timedelta(hours=2),
        )
        initialized_here = True
    return rank, world_size, initialized_here


def _pick_accuracy_metric(task_metrics: dict[str, Any]) -> float | None:
    # Prefer normalized accuracy when available.
    for key in (
        "acc_norm,none",
        "acc_norm",
        "accuracy_norm",
        "acc,none",
        "acc",
        "accuracy",
    ):
        value = task_metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _extract_task_accuracies(results: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    task_results = results.get("results", {})
    if not isinstance(task_results, dict):
        return out
    for task_name, metrics in task_results.items():
        if not isinstance(metrics, dict):
            continue
        acc = _pick_accuracy_metric(metrics)
        if acc is not None:
            out[str(task_name)] = acc
    return out


def main() -> None:
    cli = _parse_args()
    rank, world_size, initialized_here = _init_distributed_if_needed()

    if not os.path.isfile(cli.config_path):
        raise FileNotFoundError(f"Config not found: {cli.config_path}")
    if not os.path.isfile(cli.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {cli.checkpoint_path}")
    if not cli.tasks:
        raise ValueError("At least one task must be provided via --tasks.")

    # Avoid interactive "Do you trust remote code?" prompts from datasets when
    # running under torchrun/non-interactive eval jobs.
    if cli.trust_remote_code:
        try:
            from datasets import config as datasets_config

            datasets_config.HF_DATASETS_TRUST_REMOTE_CODE = True
            os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
            if rank == 0:
                print("[lm-eval] trust_remote_code enabled for datasets loading.")
        except Exception:
            if rank == 0:
                print(
                    "[lm-eval] Warning: failed to set datasets trust_remote_code; "
                    "dataset loading may prompt or fail for script-based tasks."
                )

    default_cfg = OmegaConf.load("defaults.yaml")
    user_cfg = OmegaConf.load(cli.config_path)
    cfg = OmegaConf.merge(default_cfg, user_cfg)

    if cfg.trainer.train_probe:
        raise ValueError(
            "Probe models are not causal language models; lm-eval.py does not support train_probe=true."
        )
    if cfg.use_bst:
        raise ValueError(
            "BST wrapper is currently unsupported in lm-eval.py due non-standard forward API."
        )

    tokenizer = _load_tokenizer(cfg, cli.tokenizer_name_or_path)
    if getattr(tokenizer, "eos_token_id", None) is None:
        raise ValueError("Tokenizer must provide eos_token_id for lm-eval scoring.")
    # In training this is often set by the datamodule. For standalone eval,
    # make sure vocab_size is initialized before checkpoint load.
    if int(getattr(cfg.model, "vocab_size", 0)) <= 0:
        cfg.model.vocab_size = int(getattr(tokenizer, "vocab_size", len(tokenizer)))
    # Compilation is useful for long training runs, but unnecessary overhead in eval.
    cfg.trainer.compile = False

    device = torch.device(
        cli.device if (cli.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    if device.type == "cuda" and world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    if world_size > 1:
        fabric = L.Fabric(accelerator="cuda" if device.type == "cuda" else "cpu")
    else:
        fabric = L.Fabric(
            accelerator="cuda" if device.type == "cuda" else "cpu", devices=1
        )
    fabric.launch()

    # Minimal tokenizer object expected by initialize_model.
    init_tok = SimpleNamespace(
        eos_token_id=int(tokenizer.eos_token_id),
        convert_tokens_to_ids=getattr(
            tokenizer, "convert_tokens_to_ids", lambda x: None
        ),
    )
    model = initialize_model(
        fabric=fabric,
        config=cfg,
        tokenizer=init_tok,
        initialize_optimizer=False,
        checkpoint_path=cli.checkpoint_path,
    )
    model.eval()
    if cli.max_length > int(cfg.model.block_size):
        print(
            f"max_length={cli.max_length} exceeds model block_size={cfg.model.block_size}; "
            f"using {cfg.model.block_size}."
        )
        cli.max_length = int(cfg.model.block_size)

    lm = NextLatLMWrapper(
        model_wrapper=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=cli.batch_size,
        max_length=cli.max_length,
        max_gen_toks=cli.max_gen_toks,
        temperature=cli.temperature,
        top_k=cli.top_k,
        progress_log_interval_sec=cli.progress_log_interval_sec,
        progress_log_every_n_requests=cli.progress_log_every_n_requests,
    )
    lm._rank = rank
    lm._world_size = world_size
    if world_size > 1:
        lm.accelerator = Accelerator()

    eval_kwargs: dict[str, Any] = {}
    if cli.limit is not None:
        eval_kwargs["limit"] = cli.limit
    if cli.num_fewshot is not None:
        eval_kwargs["num_fewshot"] = cli.num_fewshot

    results: Optional[dict[str, Any]] = None
    for task_idx, task_name in enumerate(cli.tasks, start=1):
        task_start = time.perf_counter()
        if rank == 0:
            print(
                f"[rank {rank}] Starting task {task_idx}/{len(cli.tasks)}: "
                f"{task_name}"
            )
        task_result = simple_evaluate(model=lm, tasks=[task_name], **eval_kwargs)
        if task_result is None:
            if rank == 0:
                raise RuntimeError(
                    f"lm-eval returned no results for task '{task_name}' on rank 0."
                )
            # In distributed lm-eval, non-zero ranks may return None while rank 0
            # owns the aggregated metrics/results payload.
            if rank == 1:
                print(
                    f"[rank {rank}] Task {task_name} returned no local payload; "
                    "continuing (expected for non-zero ranks)."
                )
            continue

        if results is None:
            results = task_result
        else:
            # Merge per-task outputs while preserving lm-eval's top-level schema.
            for key in ("results", "versions", "n-shot", "higher_is_better", "groups"):
                if key in task_result and isinstance(task_result[key], dict):
                    base = results.get(key)
                    if not isinstance(base, dict):
                        base = {}
                    base.update(task_result[key])
                    results[key] = base
            for key in ("configs",):
                if key in task_result and isinstance(task_result[key], dict):
                    base = results.get(key)
                    if not isinstance(base, dict):
                        base = {}
                    base.update(task_result[key])
                    results[key] = base
            # Keep run metadata from first call; refresh date to latest.
            if "date" in task_result:
                results["date"] = task_result["date"]

        if rank == 0:
            elapsed = time.perf_counter() - task_start
            print(
                f"[rank {rank}] Finished task {task_name} in {elapsed:.1f}s "
                f"({elapsed/60.0:.1f} min)"
            )

    if rank == 0 and results is None:
        raise RuntimeError("No lm-eval results were produced.")

    if rank == 0:
        assert results is not None
        compact = results.get("results", results)
        print(json.dumps(compact, indent=2))
        task_accuracies = _extract_task_accuracies(results)
        if task_accuracies:
            avg_accuracy = sum(task_accuracies.values()) / len(task_accuracies)
            print(
                f"[rank {rank}] Average accuracy over {len(task_accuracies)} "
                f"task(s): {avg_accuracy:.6f}"
            )
            summary = results.get("summary")
            if not isinstance(summary, dict):
                summary = {}
            summary["average_accuracy"] = avg_accuracy
            results["summary"] = summary
        else:
            print(
                f"[rank {rank}] No accuracy metrics found "
                "(acc_norm/acc). Skipping average_accuracy."
            )

    if cli.output_path and rank == 0:
        assert results is not None
        out = Path(cli.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"Saved results to {out}")

    if dist.is_initialized():
        dist.barrier()
        if initialized_here:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
