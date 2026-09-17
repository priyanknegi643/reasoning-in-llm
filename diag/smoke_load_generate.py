#!/usr/bin/env python3
"""
Step 2c (runtime load) + Step 3 (one example, one answer).

The repository-native loader is the SINGLE loading authority: core_train.initialize_model
-> ModelBase.load_checkpoint -> fabric.load, with strict checking left intact. Key
accounting is captured at that load by recording what load_state_dict returns; the
checkpoint is never opened a second time.

An incorrect generated path still passes. This establishes functioning inference, not
accuracy.

Usage (from anywhere -- paths are resolved from this file, then cwd is moved to the
NextLat root so the repo's relative dataset paths resolve):

    python Reasoning/diag/smoke_load_generate.py \
        --checkpoint ckpt_iter_5000_0.6055.pt

    # fresh-process repeatability check
    python Reasoning/diag/smoke_load_generate.py \
        --checkpoint ckpt_iter_5000_0.6055.pt --out diag/step3_repeat.json
"""

import argparse
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone

DIAG_DIR = os.path.dirname(os.path.abspath(__file__))
REASONING = os.path.dirname(DIAG_DIR)
NEXTLAT = os.path.join(REASONING, "nextlat", "NextLat")
DEFAULT_RUN = "NextLat-proj_factor0.5_lambda_kl1.0_mtp_horizon8_seed1234_lambda_mse1.0"

CKPT_ITER_RE = re.compile(r"ckpt_iter_(\d+)(?:_([0-9.]+))?\.pt$")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint filename inside the run directory, or an absolute path.",
    )
    p.add_argument("--run-dir", default=DEFAULT_RUN, help="Run directory name.")
    p.add_argument(
        "--config",
        default=None,
        help="materialized_config.yaml (default: the one beside the checkpoint).",
    )
    p.add_argument(
        "--example-index",
        type=int,
        default=None,
        help="Line index into the test file. Default: first entry of "
        "diag/diagnostic_indices.json.",
    )
    p.add_argument("--seed", type=int, default=1234, help="Seed for THIS execution.")
    p.add_argument("--out", default=os.path.join(DIAG_DIR, "step3_example.json"))
    p.add_argument(
        "--precision",
        default="32-true",
        help="Fabric precision for this run. Recorded in the output.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ---- resolve every path to an absolute path BEFORE changing directory ----
    run_dir = os.path.join(NEXTLAT, "output", "stargraph", args.run_dir)
    ckpt_path = (
        args.checkpoint
        if os.path.isabs(args.checkpoint)
        else os.path.join(run_dir, args.checkpoint)
    )
    ckpt_path = os.path.abspath(ckpt_path)
    cfg_path = os.path.abspath(
        args.config or os.path.join(os.path.dirname(ckpt_path), "materialized_config.yaml")
    )
    out_path = os.path.abspath(args.out)
    indices_path = os.path.join(DIAG_DIR, "diagnostic_indices.json")

    for label, path in (("checkpoint", ckpt_path), ("config", cfg_path)):
        if not os.path.isfile(path):
            sys.exit(f"{label} not found: {path}")

    # The repo's dataset paths are relative to the NextLat root.
    os.chdir(NEXTLAT)
    sys.path.insert(0, NEXTLAT)

    import torch
    import lightning as L
    from omegaconf import OmegaConf

    from core_train import initialize_model
    from data.stargraph import StarGraphDataModule

    record = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": ckpt_path,
        "config_path": cfg_path,
        "cwd_for_run": NEXTLAT,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "lightning": L.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "requested_precision": args.precision,
        },
        "seed_note": (
            "This seed controls THIS execution only. The RNG state in force at any "
            "historical training-time evaluation is unknown and not recoverable."
        ),
        "execution_seed": args.seed,
    }

    # ---- key accounting, captured at the native load ----
    load_calls = []
    _orig_lsd = torch.nn.Module.load_state_dict

    def _recording_load_state_dict(self, state_dict, strict=True, *a, **kw):
        result = _orig_lsd(self, state_dict, strict=strict, *a, **kw)  # strict preserved
        load_calls.append(
            {
                "module_class": type(self).__name__,
                "strict": bool(strict),
                "num_keys_in_state_dict": len(state_dict),
                "missing_keys": list(getattr(result, "missing_keys", []) or []),
                "unexpected_keys": list(getattr(result, "unexpected_keys", []) or []),
            }
        )
        return result

    torch.nn.Module.load_state_dict = _recording_load_state_dict

    try:
        # ---- stage 3 of the configuration provenance record ----
        config = OmegaConf.merge(
            OmegaConf.load(os.path.join(NEXTLAT, "defaults.yaml")),
            OmegaConf.load(cfg_path),
        )
        record["config_provenance_stage3"] = {
            "stage": "final smoke-test configuration",
            "struct_mode": OmegaConf.is_struct(config),
            "cli_overrides": [],
            "observed": {
                "temperature": repr(getattr(config.trainer, "temperature", 1.0)),
                "top_k": repr(getattr(config.trainer, "top_k", None)),
                "compile": repr(getattr(config.trainer, "compile", "<absent>")),
                "sampling_mode": repr(
                    getattr(config.trainer, "sampling_mode", "<absent>")
                ),
            },
        }

        fabric = L.Fabric(devices=1, precision=args.precision)
        fabric.launch()
        assert fabric.world_size == 1, f"world_size must be 1, got {fabric.world_size}"
        fabric.seed_everything(args.seed)

        datamodule = StarGraphDataModule(fabric, config)
        datamodule.update_config(config)
        tokenizer = datamodule.get_tokenizer()
        record["data"] = {
            "vocab_size": datamodule.vocab_size,
            "graph_description_len": datamodule.graph_description_len,
            "total_len": datamodule.total_len,
            "graph_target_len": datamodule.val_dataset.graph_target_len,
            "eos_token_id": tokenizer.eos_token_id,
            "test_data_path": config.data.stargraph_test_data_path,
        }

        # ---- the native load. initialize_model does: empty init -> optional
        # compile (config.trainer.compile) -> setup_fabric -> model.load_checkpoint
        model = initialize_model(
            fabric,
            config,
            tokenizer,
            initialize_optimizer=False,
            checkpoint_path=ckpt_path,
        )
    finally:
        torch.nn.Module.load_state_dict = _orig_lsd

    record["load_accounting"] = {
        "calls": load_calls,
        "total_missing_keys": sum(len(c["missing_keys"]) for c in load_calls),
        "total_unexpected_keys": sum(len(c["unexpected_keys"]) for c in load_calls),
        "note": (
            "Recorded from the return value of load_state_dict during the native load. "
            "strict was passed through unchanged; no second load was performed."
        ),
    }

    # ---- restored counter vs filename ----
    m = CKPT_ITER_RE.search(os.path.basename(ckpt_path))
    filename_iter = int(m.group(1)) if m else None
    restored = getattr(model, "training_steps", None)
    record["training_steps"] = {
        "restored_from_checkpoint": restored,
        "filename_iteration": filename_iter,
        "match": (restored == filename_iter),
    }

    # ---- device placement and dtype of actual parameters ----
    params = list(model.model.named_parameters())
    record["placement"] = {
        "fabric_device": str(fabric.device),
        "param_count": len(params),
        "devices": sorted({str(p.device) for _, p in params}),
        "dtypes": sorted({str(p.dtype) for _, p in params}),
        "sample": [
            {"name": n, "shape": list(p.shape), "device": str(p.device), "dtype": str(p.dtype)}
            for n, p in params[:3]
        ],
    }
    record["checkpoint_internals_resolved"] = {
        "state_dict_key_count": (
            load_calls[0]["num_keys_in_state_dict"] if load_calls else None
        ),
        "note": (
            "Supersedes the pending_runtime_confirmation entries in inventory.json for "
            "this checkpoint."
        ),
    }

    if filename_iter is not None and restored != filename_iter:
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2, default=str)
        sys.exit(
            f"STOP: restored training_steps={restored} != filename iteration="
            f"{filename_iter}. Investigate the save/increment semantics "
            f"(core_train.py:774-776, :925-950) before proceeding. Wrote {out_path}"
        )

    # ---------------------------------------------------------------- Step 3 ----
    if args.example_index is not None:
        example_index = args.example_index
        index_source = "--example-index"
    elif os.path.isfile(indices_path):
        with open(indices_path) as f:
            example_index = json.load(f)["indices"][0]
        index_source = "diag/diagnostic_indices.json[0]"
    else:
        example_index = 0
        index_source = "fallback: line 0"

    with open(config.data.stargraph_test_data_path) as f:
        lines = f.readlines()
    raw_line = lines[example_index].strip()

    tokens, seq_len = tokenizer.tokenize(raw_line)
    tokens = list(map(int, tokens))

    target_len = datamodule.val_dataset.graph_target_len
    tokens_including_eos = target_len + 1
    prefix_ids = tokens[:-tokens_including_eos]
    target_ids = tokens[-tokens_including_eos:-1]

    def decode(ids):
        return ",".join(tokenizer.decoder[int(i)] for i in ids)

    # ---- input boundary assertions, before generating ----
    eq_id = tokenizer.encoder["="]
    checks = {
        "seq_len_equals_total_len": (seq_len == datamodule.total_len),
        "last_token_is_eos": (tokens[-1] == tokenizer.eos_token_id),
        "prefix_len": len(prefix_ids),
        "prefix_ends_with_equals": (prefix_ids[-1] == eq_id),
        "target_len_equals_graph_target_len": (len(target_ids) == target_len),
    }
    record["example"] = {
        "index": example_index,
        "index_source": index_source,
        "indexing_convention": "0-based line index into the test file as written on disk",
        "raw_line": raw_line,
        "encoded_ids": tokens,
        "seq_len": seq_len,
        "prefix_ids": prefix_ids,
        "prefix_decoded": decode(prefix_ids),
        "target_ids": target_ids,
        "target_decoded": decode(target_ids),
        "boundary_checks": checks,
    }
    failed = [k for k, v in checks.items() if v is False]
    if failed:
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2, default=str)
        sys.exit(f"STOP: input boundary checks failed: {failed}. Wrote {out_path}")

    # ---- generate. eval mode AND no grad; generate() is @torch.inference_mode but
    # do not rely on that alone.
    model.eval()
    temperature = getattr(config.trainer, "temperature", 1.0)
    top_k = getattr(config.trainer, "top_k", None)

    prefix = torch.tensor([prefix_ids], dtype=torch.long, device=fabric.device)
    with torch.no_grad():
        out = model.generate(
            prefix, max_new_tokens=target_len, temperature=temperature, top_k=top_k
        )

    gen_ids = [int(i) for i in out[0, -target_len:].tolist()]
    full_ids = [int(i) for i in out[0].tolist()]
    special = {
        i: tokenizer.decoder[i]
        for i in set(gen_ids)
        if i >= config.data.stargraph_max_nodes
    }

    record["generation"] = {
        "decoding": {
            "method": "model.generate (multinomial sampling; no argmax branch, no EOS stop)",
            "temperature": temperature,
            "top_k": top_k,
            "max_new_tokens": target_len,
            "source_of_settings": "config_provenance_stage3.observed",
        },
        "generated_ids": gen_ids,
        "generated_decoded": decode(gen_ids),
        "full_output_ids": full_ids,
        "special_tokens_in_output": special or None,
        "exact_match_vs_target": (gen_ids == target_ids),
        "per_position_match": [int(a == b) for a, b in zip(gen_ids, target_ids)],
        "note": (
            "An incorrect path still passes Step 3. This records functioning inference, "
            "not accuracy. generate() recomputes the transformer every step, so this "
            "exercises no recursive latent-only rollout and says nothing about "
            "speculative_propose or speculative verification."
        ),
    }

    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    # ---- console summary ----
    print("\n" + "=" * 72)
    print(f"checkpoint      {os.path.basename(ckpt_path)}")
    print(
        f"load accounting missing={record['load_accounting']['total_missing_keys']} "
        f"unexpected={record['load_accounting']['total_unexpected_keys']} "
        f"(state_dict keys={record['checkpoint_internals_resolved']['state_dict_key_count']})"
    )
    print(
        f"training_steps  restored={restored} filename={filename_iter} "
        f"match={record['training_steps']['match']}"
    )
    print(
        f"placement       device={record['placement']['devices']} "
        f"dtype={record['placement']['dtypes']}"
    )
    print(f"example         index={example_index} ({index_source})")
    print(f"  prefix  {record['example']['prefix_decoded']}")
    print(f"  target  {record['example']['target_decoded']}")
    print(f"  output  {record['generation']['generated_decoded']}")
    print(
        f"  exact match: {record['generation']['exact_match_vs_target']}   "
        f"per-position: {record['generation']['per_position_match']}"
    )
    print(f"\nwrote {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
