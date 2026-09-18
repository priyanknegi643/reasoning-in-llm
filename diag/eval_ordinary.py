#!/usr/bin/env python3
"""
Step 4: Ordinary-generation performance (baseline).

Measures whole-answer exact match + per-position token accuracy on the 200-example
diagnostic subset (diag/diagnostic_indices.json), using the repository-native
model.generate() path only -- full Transformer forward each step, KV cache, no
dynamics_model, no speculative_propose. This is the reference baseline every
downstream latent-rollout / speculative-decoding result gets compared against.

Follows the same loading authority as diag/smoke_load_generate.py:
core_train.initialize_model -> ModelBase.load_checkpoint -> fabric.load, strict
checking left intact, checkpoint opened exactly once.

Usage:
    python Reasoning/diag/eval_ordinary.py \
        --checkpoint ckpt_iter_20001_0.7224.pt \
        --run-dir NextLat-proj_factor0.5_lambda_kl1.0_mtp_horizon8_seed1234_lambda_mse1.0 \
        --indices diag/diagnostic_indices.json \
        --out diag/step4_ordinary_eval.json
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

# This script lives in /workspace/Research/diag/
# The NextLat repo is at /workspace/Research/nextlat/NextLat
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # /workspace/Research
NEXTLAT = os.path.join(PROJECT_ROOT, "nextlat", "NextLat")
DIAG_DIR = os.path.join(PROJECT_ROOT, "diag")
DEFAULT_RUN = "NextLat-proj_factor0.5_lambda_kl1.0_mtp_horizon8_seed1234_lambda_mse1.0"

CKPT_ITER_RE = re.compile(r"ckpt_iter_(\d+)(?:_([0-9.]+))?\.pt$")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True,
                    help="Checkpoint filename inside the run directory, or an absolute path.")
    p.add_argument("--run-dir", default=DEFAULT_RUN)
    p.add_argument("--config", default=None,
                    help="materialized_config.yaml (default: the one beside the checkpoint).")
    p.add_argument("--indices", default=os.path.join(DIAG_DIR, "diagnostic_indices.json"),
                    help="JSON file with the fixed diagnostic index list.")
    p.add_argument("--seed", type=int, default=1234, help="Seed for THIS execution only.")
    p.add_argument("--temperature", type=float, default=None,
                    help="Override; default reads config_provenance-style getattr fallback (1.0).")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--device", default=None, help="cuda / cpu; default: cuda if available.")
    p.add_argument("--out", default=os.path.join(DIAG_DIR, "step4_ordinary_eval.json"))
    return p.parse_args()


def resolve_paths(args):
    # First try the path as-is (relative to CWD or absolute)
    if os.path.isfile(args.checkpoint):
        ckpt_path = os.path.abspath(args.checkpoint)
        run_dir_path = os.path.dirname(ckpt_path)
    elif os.path.isabs(args.checkpoint):
        # Absolute path but doesn't exist
        sys.exit(f"Checkpoint not found: {args.checkpoint}")
    else:
        # Relative path that doesn't exist as-is, try relative to NEXTLAT
        run_dir_path = os.path.join(NEXTLAT, "output", "stargraph", args.run_dir)
        ckpt_path = os.path.join(run_dir_path, args.checkpoint)
        if not os.path.isfile(ckpt_path):
            sys.exit(f"Checkpoint not found: {ckpt_path}")

    cfg_path = args.config or os.path.join(run_dir_path, "materialized_config.yaml")
    if not os.path.isfile(cfg_path):
        sys.exit(f"Config not found: {cfg_path}")

    return ckpt_path, cfg_path


def main():
    args = parse_args()
    ckpt_path, cfg_path = resolve_paths(args)

    # Resolve output and indices paths to absolute BEFORE chdir
    out_path = os.path.abspath(args.out)
    indices_path = os.path.abspath(args.indices)

    if not os.path.isfile(indices_path):
        sys.exit(f"Diagnostic indices file not found: {indices_path}")
    with open(indices_path) as f:
        indices_doc = json.load(f)
    diagnostic_indices = indices_doc if isinstance(indices_doc, list) else indices_doc["indices"]

    os.chdir(NEXTLAT)  # relative dataset paths in the config resolve from here
    sys.path.insert(0, NEXTLAT)

    import torch
    import lightning as L
    from omegaconf import OmegaConf
    from core_train import initialize_model
    from data.stargraph import Tokenizer

    config = OmegaConf.load(cfg_path)

    torch.manual_seed(args.seed)
    accelerator = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    fabric = L.Fabric(accelerator=accelerator)
    fabric.launch()

    tokenizer = Tokenizer(config.data.stargraph_max_nodes)

    model = initialize_model(
        fabric, config, tokenizer,
        initialize_optimizer=False,
        checkpoint_path=ckpt_path,
    )
    model.eval()
    core = model._unwrap_module() if hasattr(model, "_unwrap_module") else model

    test_path = config.data.stargraph_test_data_path
    with open(test_path) as f:
        test_lines = f.readlines()

    graph_target_len = config.data.get("graph_target_len", None)
    # Fall back to what step3_example.json already established for this task if the
    # config field isn't present under this exact name.
    if graph_target_len is None:
        graph_target_len = 10

    temperature = args.temperature if args.temperature is not None else 1.0
    top_k = args.top_k  # None matches step3_example.json's observed config

    per_example = []
    per_position_correct = [0] * graph_target_len
    exact_match_count = 0
    error_position_histogram = {}

    m = CKPT_ITER_RE.search(os.path.basename(ckpt_path))
    ckpt_iter = int(m.group(1)) if m else None

    with torch.inference_mode():
        for idx in diagnostic_indices:
            line = test_lines[idx].strip()
            seq, _ = tokenizer.tokenize(line)
            seq_t = torch.tensor(seq, dtype=torch.long, device=fabric.device).unsqueeze(0)

            tokens_including_eos = graph_target_len + 1
            prefix = seq_t[:, :-tokens_including_eos]
            target = seq_t[:, -tokens_including_eos:-1]

            generated = model.generate(
                prefix,
                max_new_tokens=graph_target_len,
                temperature=temperature,
                top_k=top_k,
            )
            gen_tail = generated[:, -graph_target_len:]

            match_per_pos = (gen_tail == target).squeeze(0).tolist()
            is_exact = all(match_per_pos)
            if is_exact:
                exact_match_count += 1
            else:
                first_wrong = next(i for i, ok in enumerate(match_per_pos) if not ok)
                error_position_histogram[first_wrong] = error_position_histogram.get(first_wrong, 0) + 1

            for i, ok in enumerate(match_per_pos):
                if ok:
                    per_position_correct[i] += 1

            per_example.append({
                "index": idx,
                "exact_match": is_exact,
                "per_position_match": match_per_pos,
            })

    n = len(diagnostic_indices)
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": os.path.basename(ckpt_path),
        "checkpoint_iter": ckpt_iter,
        "config_path": cfg_path,
        "num_examples": n,
        "exact_match_rate": exact_match_count / n if n else None,
        "per_position_accuracy": [c / n if n else None for c in per_position_correct],
        "error_position_histogram": error_position_histogram,
        "generation_settings": {"temperature": temperature, "top_k": top_k},
        "execution_seed": args.seed,
        "per_example": per_example,
        "note": (
            "Uses model.generate() only -- full Transformer forward per step, KV cache. "
            "dynamics_model / speculative_propose are not exercised here; this is the "
            "reference baseline for Steps 6-9."
        ),
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"exact_match_rate={result['exact_match_rate']:.4f}  n={n}  -> {out_path}")


if __name__ == "__main__":
    main()
