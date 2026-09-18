#!/usr/bin/env python3
"""
Step 5: Transition/target alignment verification.

Manually replicates the recursive teacher-forced loss for k=1,2 on a single batch,
against the actual loaded model, and checks it against what the training code is
claimed to do (model_nextlat.py: _nextlat_compute_losses lines 374-409,
_nextlat_loss_function lines 298-335, NextLatDynamicsModel.forward lines 78-92).

This script does NOT trust those line numbers -- it re-derives k=1,2 from
model.forward(..., return_hidden_states=True) and the model's own dynamics_model /
lm_head calls, so a mismatch between what the code comment claims and what the model
actually does will show up here rather than being assumed away.

No gradients. Single batch (B=4 by default) from val_dataloader.

Usage:
    python Reasoning/diag/check_alignment.py \
        --checkpoint ckpt_iter_20001_0.7224.pt \
        --run-dir NextLat-proj_factor0.5_lambda_kl1.0_mtp_horizon8_seed1234_lambda_mse1.0 \
        --out diag/step5_alignment_check.json
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

# This script lives in /home/shivaanshgusain/Research/
# The NextLat repo is at /home/shivaanshgusain/Research/Reasoning/nextlat/NextLat
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REASONING = os.path.join(SCRIPT_DIR)
NEXTLAT = os.path.join(REASONING, "nextlat", "NextLat")
DIAG_DIR = os.path.join(REASONING, "diag")
DEFAULT_RUN = "NextLat-proj_factor0.5_lambda_kl1.0_mtp_horizon8_seed1234_lambda_mse1.0"

CKPT_ITER_RE = re.compile(r"ckpt_iter_(\d+)(?:_([0-9.]+))?\.pt$")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--run-dir", default=DEFAULT_RUN)
    p.add_argument("--config", default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eos-id", type=int, default=None,
                    help="Override; default reads from stargraph_max_nodes+4 via Tokenizer.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=os.path.join(DIAG_DIR, "step5_alignment_check.json"))
    return p.parse_args()


def resolve_paths(args):
    if os.path.isabs(args.checkpoint) and os.path.isfile(args.checkpoint):
        ckpt_path = args.checkpoint
        run_dir_path = os.path.dirname(ckpt_path)
    else:
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

    os.chdir(NEXTLAT)
    sys.path.insert(0, NEXTLAT)

    import torch
    import torch.nn.functional as F
    import lightning as L
    from omegaconf import OmegaConf
    from core_train import initialize_model
    from data.stargraph import Tokenizer, StarGraphDataModule

    config = OmegaConf.load(cfg_path)

    torch.manual_seed(args.seed)
    accelerator = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    fabric = L.Fabric(accelerator=accelerator)
    fabric.launch()

    tokenizer = Tokenizer(config.data.stargraph_max_nodes)
    eos_id = args.eos_id if args.eos_id is not None else tokenizer.eos_token_id

    model = initialize_model(
        fabric, config, tokenizer,
        initialize_optimizer=False,
        checkpoint_path=ckpt_path,
    )
    model.eval()
    core = model._unwrap_module() if hasattr(model, "_unwrap_module") else model

    if not hasattr(core, "dynamics_model"):
        sys.exit(
            "Loaded model has no `dynamics_model` attribute -- this checkpoint/config "
            "combination is not NextLat, or the attribute name has changed upstream. "
            "Aborting rather than silently checking the wrong thing."
        )

    datamodule = StarGraphDataModule(fabric, config)
    val_loader = datamodule.val_dataloader()
    batch = next(iter(val_loader))
    if batch.shape[0] > args.batch_size:
        batch = batch[: args.batch_size]
    batch = batch.to(fabric.device)
    B, T = batch.shape

    with torch.inference_mode():
        # Ground truth: full-attention forward over the real sequence.
        out = core(batch, return_hidden_states=True)
        # forward() may return (logits, hidden_states) or an object; handle both.
        if isinstance(out, tuple):
            _, hidden_states = out
        else:
            hidden_states = out.hidden_states
        token_embeds = core.token_embedding(batch)

        h = hidden_states            # (B, T, D)
        x_emb = token_embeds         # (B, T, D)

        # k=1: pred_{t+1} = dynamics(h_t, E(x_{t+1})); target = h_{t+1}, detached.
        h_t = h[:, :-1, :]
        x_next = x_emb[:, 1:, :]
        pred_1 = core.dynamics_model(h_t, x_next)
        target_1 = h[:, 1:, :].detach()
        mse_1_per_pos = F.smooth_l1_loss(pred_1, target_1, reduction="none").mean(dim=-1)  # (B, T-1)

        # k=2: pred_{t+2} = dynamics(pred_{t+1}, E(x_{t+2})); target = h_{t+2}, detached.
        # This is where an off-by-one is most likely to hide: pred_1 must be shifted
        # to line up with x_{t+2}, not reused at its own original t-index.
        pred_1_shifted = pred_1[:, :-1, :]     # drop last, so index i corresponds to t+1 at position t
        x_next2 = x_emb[:, 2:, :]
        pred_2 = core.dynamics_model(pred_1_shifted, x_next2)
        target_2 = h[:, 2:, :].detach()
        mse_2_per_pos = F.smooth_l1_loss(pred_2, target_2, reduction="none").mean(dim=-1)  # (B, T-2)

        # EOS-crossing mask: positions where the INPUT token (the one being predicted
        # from, i.e. x_t at that position) is EOS should be excluded from the loss,
        # per the claimed masking in model_nextlat.py:394. Recompute independently here.
        input_tokens_k1 = batch[:, :-1]   # aligned with mse_1_per_pos
        eos_mask_k1 = (input_tokens_k1 == eos_id)
        input_tokens_k2 = batch[:, :-2]
        eos_mask_k2 = (input_tokens_k2 == eos_id)

        mse_1_masked = mse_1_per_pos.clone()
        mse_1_masked[eos_mask_k1] = float("nan")
        mse_2_masked = mse_2_per_pos.clone()
        mse_2_masked[eos_mask_k2] = float("nan")

        k1_mse_mean = mse_1_per_pos[~eos_mask_k1].mean().item() if (~eos_mask_k1).any() else None
        k2_mse_mean = mse_2_per_pos[~eos_mask_k2].mean().item() if (~eos_mask_k2).any() else None

        # Residual-connection check: is dynamics_model literally h + delta, or does it
        # already include the residual internally? Compare against a zero-token-input
        # control isn't meaningful here since the token conditions the delta, so instead
        # verify by inspecting whether dynamics_model output norm tracks h's norm (a
        # proxy, not a proof) -- flagged as inconclusive-by-design if ambiguous.
        pred_norm_ratio = (pred_1.norm(dim=-1) / (h_t.norm(dim=-1) + 1e-8)).mean().item()

    m = CKPT_ITER_RE.search(os.path.basename(ckpt_path))
    ckpt_iter = int(m.group(1)) if m else None

    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": os.path.basename(ckpt_path),
        "checkpoint_iter": ckpt_iter,
        "config_path": cfg_path,
        "batch_shape": [B, T],
        "eos_token_id_used": eos_id,
        "k1_mse": k1_mse_mean,
        "k2_mse": k2_mse_mean,
        "k1_mse_first_few_positions": mse_1_per_pos[0, :5].tolist(),
        "k2_mse_first_few_positions": mse_2_per_pos[0, :5].tolist(),
        "shapes": {
            "pred_1": list(pred_1.shape),
            "target_1": list(target_1.shape),
            "pred_2": list(pred_2.shape),
            "target_2": list(target_2.shape),
        },
        "eos_masked_positions_k1": int(eos_mask_k1.sum().item()),
        "eos_masked_positions_k2": int(eos_mask_k2.sum().item()),
        "pred_over_h_norm_ratio": pred_norm_ratio,
        "concat_order_assumed": "dynamics_model(h_t, next_token_embedding) -- NOT independently "
                                  "verified against source; re-check against the actual "
                                  "NextLatDynamicsModel.forward signature if this script's k1_mse "
                                  "diverges sharply from the training log's mse_loss at a "
                                  "comparable step.",
        "note": (
            "k1_mse here is computed on a fresh val batch at inference time and is NOT "
            "expected to exactly match the training log's mse_loss at step 20000 (different "
            "batch, no dropout randomness during eval, etc.) -- compare ORDER OF MAGNITUDE "
            "and trend, not exact equality. A large discrepancy (>1 order of magnitude) is "
            "the actual alignment-bug signal to look for."
        ),
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"k1_mse={k1_mse_mean}  k2_mse={k2_mse_mean}  -> {args.out}")


if __name__ == "__main__":
    main()
