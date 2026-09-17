#!/usr/bin/env python
"""
Comprehensive checkpoint analysis for NextLat on PathStar (StarGraph).
Runs all 6 issues identified in the forensic reconstruction.
"""

import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
import lightning as L
import sys
import os

# Add project root to path
sys.path.insert(0, '/workspace/Research/nextlat')

from NextLat.models.model_nextlat import NextLat, NextLatConfig
from NextLat.data.stargraph import StarGraphDataModule, Tokenizer
from NextLat.utils.speculative_sampling import speculative_decode_v2, normalize_logits, sample_from_probs


def load_checkpoint_and_config():
    """Load checkpoint and materialized config."""
    ckpt_path = "nextlat/NextLat/output/stargraph/NextLat-proj_factor0.5_lambda_kl1.0_mtp_horizon8_seed1234_lambda_mse1.0/ckpt_iter_20001_0.7224.pt"
    config_path = "nextlat/NextLat/output/stargraph/NextLat-proj_factor0.5_lambda_kl1.0_mtp_horizon8_seed1234_lambda_mse1.0/materialized_config.yaml"

    config = OmegaConf.load(config_path)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    return config, ckpt


def setup_model_and_data(config, ckpt):
    """Initialize model, fabric, and dataloader."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fabric = L.Fabric(accelerator=device, devices=1)
    fabric.launch()

    # Update config with vocab size from data module
    datamodule = StarGraphDataModule(fabric, config)
    datamodule.update_config(config)
    config.model.vocab_size = datamodule.vocab_size

    # Initialize model
    model = NextLat(NextLatConfig(**config.model))
    model.setup_fabric(fabric)

    # Load checkpoint
    model.model.load_state_dict(ckpt['model'])
    model.eval()

    val_loader = datamodule.val_dataloader()

    return model, datamodule, val_loader, fabric


@torch.inference_mode()
def measure_horizon_error(model, batch, config, max_k=20):
    """Issue 2: Measure error accumulation beyond training horizon."""
    core = model._unwrap_module()
    _, hidden_states = core(batch, return_hidden_states=True)
    token_embeds = core.token_embedding(batch)

    results = []
    start_pos = config.model.context_length  # 56

    # Start from last prefix token
    pred_h = hidden_states[:, start_pos-1:start_pos, :]

    for k in range(1, max_k+1):
        if start_pos + k >= hidden_states.shape[1]:
            break

        # Teacher-forced token embedding
        next_tok_emb = token_embeds[:, start_pos + k : start_pos + k + 1, :]
        true_h = hidden_states[:, start_pos + k : start_pos + k + 1, :]

        # Predict next latent
        pred_h = core.dynamics_model(pred_h.squeeze(1), next_tok_emb.squeeze(1)).unsqueeze(1)

        # Metrics
        mse = F.smooth_l1_loss(pred_h, true_h).item()
        cos_sim = F.cosine_similarity(pred_h, true_h, dim=-1).mean().item()

        # Token prediction from predicted latent
        logits = core.lm_head(pred_h)
        pred_tok = logits.argmax(-1)
        true_tok = batch[:, start_pos + k : start_pos + k + 1]
        acc = (pred_tok == true_tok).float().mean().item()

        results.append({'k': k, 'mse': mse, 'cos_sim': cos_sim, 'token_acc': acc})

    return results


@torch.inference_mode()
def measure_rollout_mismatch(model, batch, config):
    """Issue 1: Compare teacher-forced vs free-running rollout."""
    core = model._unwrap_module()
    _, hidden_states = core(batch, return_hidden_states=True)
    token_embeds = core.token_embedding(batch)

    start_pos = config.model.context_length
    horizon = config.model.mtp_horizon

    # Teacher-forced (what training does)
    pred_h_tf = hidden_states[:, start_pos-1:start_pos, :]
    tf_results = []

    for k in range(1, horizon+1):
        if start_pos + k >= hidden_states.shape[1]:
            break
        next_tok_emb = token_embeds[:, start_pos + k : start_pos + k + 1, :]
        true_h = hidden_states[:, start_pos + k : start_pos + k + 1, :]
        pred_h_tf = core.dynamics_model(pred_h_tf.squeeze(1), next_tok_emb.squeeze(1)).unsqueeze(1)

        mse = F.smooth_l1_loss(pred_h_tf, true_h).item()
        cos_sim = F.cosine_similarity(pred_h_tf, true_h, dim=-1).mean().item()
        logits = core.lm_head(pred_h_tf)
        pred_tok = logits.argmax(-1)
        true_tok = batch[:, start_pos + k : start_pos + k + 1]
        acc = (pred_tok == true_tok).float().mean().item()
        tf_results.append({'k': k, 'mse': mse, 'cos_sim': cos_sim, 'token_acc': acc})

    # Free-running (what speculative decoding does)
    pred_h_fr = hidden_states[:, start_pos-1:start_pos, :]
    fr_results = []

    for k in range(1, horizon+1):
        if start_pos + k >= hidden_states.shape[1]:
            break
        # Use own predictions for next token embedding
        logits = core.lm_head(pred_h_fr)
        pred_tok = logits.argmax(-1, keepdim=True)
        next_tok_emb = core.token_embedding(pred_tok).squeeze(1)
        pred_h_fr = core.dynamics_model(pred_h_fr.squeeze(1), next_tok_emb).unsqueeze(1)

        true_h = hidden_states[:, start_pos + k : start_pos + k + 1, :]
        true_tok = batch[:, start_pos + k : start_pos + k + 1]

        mse = F.smooth_l1_loss(pred_h_fr, true_h).item()
        cos_sim = F.cosine_similarity(pred_h_fr, true_h, dim=-1).mean().item()
        acc = (pred_tok == true_tok).float().mean().item()
        fr_results.append({'k': k, 'mse': mse, 'cos_sim': cos_sim, 'token_acc': acc})

    return {'teacher_forced': tf_results, 'free_running': fr_results}


@torch.inference_mode()
def measure_collapse(model, val_loader, config, max_batches=10):
    """Issue 3: Measure representation collapse/degeneration."""
    core = model._unwrap_module()

    all_h_true = []
    all_h_pred = []
    all_targets = []

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break

        _, hidden_states = core(batch, return_hidden_states=True)
        token_embeds = core.token_embedding(batch)

        # Teacher-forced rollout for horizon=8
        h_true = hidden_states[:, :-1].reshape(-1, hidden_states.shape[-1])
        h_pred = hidden_states[:, :-1].clone()
        next_emb = token_embeds[:, 1:]

        for k in range(config.model.mtp_horizon):
            h_pred = core.dynamics_model(h_pred, next_emb)
            next_emb = next_emb[:, 1:]
            if next_emb.shape[1] == 0:
                break

        h_pred = h_pred.reshape(-1, h_pred.shape[-1])

        all_h_true.append(h_true)
        all_h_pred.append(h_pred)
        all_targets.append(batch[:, 1:].reshape(-1))

    H_true = torch.cat(all_h_true, dim=0)
    H_pred = torch.cat(all_h_pred, dim=0)
    Y = torch.cat(all_targets, dim=0)

    # 1. Numerical rank
    rank_true = torch.linalg.matrix_rank(H_true, atol=1e-3, rtol=1e-3).item()
    rank_pred = torch.linalg.matrix_rank(H_pred, atol=1e-3, rtol=1e-3).item()

    # 2. Effective rank
    def effective_rank(X):
        U, S, V = torch.svd(X)
        p = S / S.sum()
        p = torch.clamp(p, min=1e-12)
        return torch.exp(-(p * p.log()).sum()).item()

    eff_rank_true = effective_rank(H_true)
    eff_rank_pred = effective_rank(H_pred)

    # 3. Pairwise cosine similarity (sample for efficiency)
    n_samples = min(1000, H_pred.shape[0])
    idx = torch.randperm(H_pred.shape[0])[:n_samples]
    H_sample = H_pred[idx]
    cos_sims = F.cosine_similarity(H_sample.unsqueeze(1), H_sample.unsqueeze(0), dim=-1)
    mean_cos = cos_sims.mean().item()
    # Exclude diagonal (self-similarity = 1)
    mask = ~torch.eye(n_samples, dtype=bool)
    max_cos = cos_sims[mask].max().item()

    # 4. Linear probe: predict next token from latent
    from sklearn.linear_model import LogisticRegression

    X_true = H_true.cpu().numpy()
    X_pred = H_pred.cpu().numpy()
    y = Y.cpu().numpy()

    probe_true = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1).fit(X_true, y)
    probe_pred = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1).fit(X_pred, y)

    probe_acc_true = probe_true.score(X_true, y)
    probe_acc_pred = probe_pred.score(X_pred, y)

    return {
        'rank_true': rank_true, 'rank_pred': rank_pred,
        'eff_rank_true': eff_rank_true, 'eff_rank_pred': eff_rank_pred,
        'mean_cos_sim': mean_cos, 'max_cos_sim': max_cos,
        'probe_acc_true': probe_acc_true, 'probe_acc_pred': probe_acc_pred,
        'n_samples': H_true.shape[0]
    }


@torch.inference_mode()
def measure_memory_sufficiency(model, prefix, target, config):
    """Issue 4: Compare standard AR vs transition-only generation."""
    core = model._unwrap_module()

    # Mode 1: Standard AR generation (full attention)
    standard_out = core.generate(prefix, max_new_tokens=target.shape[1], temperature=1.0, top_k=None)
    standard_acc = (standard_out[:, -target.shape[1]:] == target).float().mean().item()
    standard_exact = (standard_out[:, -target.shape[1]:] == target).all(dim=1).float().mean().item()

    # Mode 2: Pure transition rollout (no Transformer after prefix)
    _, hidden_states = core(prefix, return_hidden_states=True)
    h = hidden_states[:, -1, :]
    transition_tokens = []
    for _ in range(target.shape[1]):
        logits = core.lm_head(h)
        tok = logits.argmax(-1, keepdim=True)
        transition_tokens.append(tok)
        h = core.dynamics_model(h, core.token_embedding(tok).squeeze(1))
    transition_out = torch.cat([prefix, torch.cat(transition_tokens, dim=1)], dim=1)
    transition_acc = (transition_out[:, -target.shape[1]:] == target).float().mean().item()
    transition_exact = (transition_out[:, -target.shape[1]:] == target).all(dim=1).float().mean().item()

    # Mode 3: Speculative decoding (uses transition for draft)
    draft_tokens, q_probs = core.speculative_propose(prefix, target.shape[1], temp=1.0, top_k=None, top_p=None)
    # Verify with target model (same model here)
    full = torch.cat([prefix, draft_tokens], dim=1)
    logits = core(full)
    p_probs_steps = []
    for i in range(draft_tokens.shape[1]):
        p_probs_steps.append(normalize_logits(logits[:, prefix.shape[1]-1+i, :], temperature=1.0))
    p_probs_steps = torch.stack(p_probs_steps, dim=1)

    # Compute acceptance
    accepted = 0
    total = 0
    for i in range(draft_tokens.shape[1]):
        tok = draft_tokens[:, i:i+1]
        q_prob = q_probs[i].gather(1, tok).squeeze().clamp_min(1e-12)
        p_prob = p_probs_steps[:, i, :].gather(1, tok).squeeze()
        accept_prob = torch.minimum(torch.ones_like(q_prob), p_prob / q_prob)
        accepted += (torch.rand_like(q_prob) < accept_prob).float().sum().item()
        total += q_prob.shape[0]

    acceptance_rate = accepted / total if total > 0 else 0

    return {
        'standard_ar_token_acc': standard_acc,
        'standard_ar_exact_acc': standard_exact,
        'transition_only_token_acc': transition_acc,
        'transition_only_exact_acc': transition_exact,
        'speculative_acceptance_rate': acceptance_rate,
    }


@torch.inference_mode()
def probe_backbone_quality(model, val_loader, config, max_batches=20):
    """Issue 5: Linear probe on hidden states to predict future tokens."""
    core = model._unwrap_module()

    all_h = []
    all_targets_by_k = {k: [] for k in [1, 2, 3, 5, 8, 10]}

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        _, h = core(batch, return_hidden_states=True)
        # h shape: (B, T, d)
        all_h.append(h[:, :-1].reshape(-1, h.shape[-1]))

        for k in [1, 2, 3, 5, 8, 10]:
            if k < batch.shape[1]:
                all_targets_by_k[k].append(batch[:, k:].reshape(-1))

    H = torch.cat(all_h, dim=0).cpu().numpy()

    from sklearn.linear_model import LogisticRegression

    results = {}
    for k in [1, 2, 3, 5, 8, 10]:
        if k not in all_targets_by_k or len(all_targets_by_k[k]) == 0:
            continue
        Y = torch.cat(all_targets_by_k[k], dim=0).cpu().numpy()
        # Truncate to match
        min_len = min(H.shape[0], Y.shape[0])
        probe = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1).fit(H[:min_len], Y[:min_len])
        acc = probe.score(H[:min_len], Y[:min_len])
        results[f'probe_k{k}_acc'] = acc

    return results


@torch.inference_mode()
def benchmark_speculative(model, val_loader, config, num_prompts=20, gamma=4):
    """Issue 6: Speculative decoding benchmark."""
    core = model._unwrap_module()

    # Collect prompts
    prompts = []
    for batch in val_loader:
        for i in range(min(batch.shape[0], num_prompts - len(prompts))):
            prompts.append(batch[i:i+1, :config.model.context_length])
        if len(prompts) >= num_prompts:
            break

    def propose_fn(seq, steps):
        return core.speculative_propose(seq, steps, temp=1.0, top_k=None, top_p=None)

    def target_probs_fn(seq, draft_tokens):
        full = torch.cat([seq, draft_tokens], dim=1)
        logits = core(full)
        prefix_len = seq.size(1)
        steps = draft_tokens.size(1)
        step_logits = logits[:, prefix_len-1:prefix_len-1+steps, :]
        next_logits = logits[:, prefix_len-1+steps, :]
        probs_steps = []
        for t in range(steps):
            probs_steps.append(normalize_logits(step_logits[:, t, :], temperature=1.0))
        p_probs_steps = torch.stack(probs_steps, dim=1)
        p_next_probs = normalize_logits(next_logits, temperature=1.0)
        return p_probs_steps, p_next_probs

    all_stats = []
    for prompt in prompts:
        output, stats = speculative_decode_v2(
            prompt, max_new_tokens=10, gamma=gamma,
            propose_fn=propose_fn, target_probs_fn=target_probs_fn
        )
        all_stats.append(stats)

    # Aggregate
    agg = {}
    for key in ['acceptance_rate', 'avg_accepted_tokens_per_step', 'proposed_tokens', 'accepted_tokens', 'rejected_tokens']:
        agg[key] = np.mean([s.get(key, 0) for s in all_stats])

    # Per-position acceptance
    for i in range(1, gamma):
        pos = i + 1
        key = f'acceptance_rate_pos_{pos}'
        agg[key] = np.mean([s.get(key, 0) for s in all_stats])

    agg['theoretical_max_speedup'] = gamma
    agg['actual_speedup'] = 1 + agg['avg_accepted_tokens_per_step']

    return agg


def main():
    print("=" * 60)
    print("LOADING CHECKPOINT AND CONFIG")
    print("=" * 60)
    config, ckpt = load_checkpoint_and_config()
    model, datamodule, val_loader, fabric = setup_model_and_data(config, ckpt)

    # Get a test batch
    batch = next(iter(val_loader))
    prefix = batch[:, :config.model.context_length]
    target = batch[:, config.model.context_length:config.model.context_length + 10]

    print(f"Batch shape: {batch.shape}")
    print(f"Prefix shape: {prefix.shape}, Target shape: {target.shape}")
    print(f"Context length: {config.model.context_length}")
    print(f"MTP horizon: {config.model.mtp_horizon}")
    print(f"Target tokens: {target.shape[1]}")

    # Issue 1: Rollout Mismatch
    print("\n" + "=" * 60)
    print("ISSUE 1: ROLLOUT MISMATCH (Teacher-forced vs Free-running)")
    print("=" * 60)
    results_1 = measure_rollout_mismatch(model, batch, config)
    print("Teacher-forced:")
    for r in results_1['teacher_forced']:
        print(f"  k={r['k']}: MSE={r['mse']:.6f}, cos={r['cos_sim']:.4f}, token_acc={r['token_acc']:.4f}")
    print("Free-running:")
    for r in results_1['free_running']:
        print(f"  k={r['k']}: MSE={r['mse']:.6f}, cos={r['cos_sim']:.4f}, token_acc={r['token_acc']:.4f}")

    # Issue 2: Horizon Error Accumulation
    print("\n" + "=" * 60)
    print("ISSUE 2: LONG-HORIZON ERROR ACCUMULATION (k=1..20)")
    print("=" * 60)
    results_2 = measure_horizon_error(model, batch, config, max_k=20)
    for r in results_2:
        print(f"  k={r['k']}: MSE={r['mse']:.6f}, cos={r['cos_sim']:.4f}, token_acc={r['token_acc']:.4f}")

    # Issue 3: Representation Collapse
    print("\n" + "=" * 60)
    print("ISSUE 3: REPRESENTATION COLLAPSE / DEGENERATION")
    print("=" * 60)
    results_3 = measure_collapse(model, val_loader, config)
    print(f"  Numerical rank - True: {results_3['rank_true']}, Pred: {results_3['rank_pred']}")
    print(f"  Effective rank - True: {results_3['eff_rank_true']:.2f}, Pred: {results_3['eff_rank_pred']:.2f}")
    print(f"  Mean cos sim (pred): {results_3['mean_cos_sim']:.4f}")
    print(f"  Max cos sim (pred): {results_3['max_cos_sim']:.4f}")
    print(f"  Probe acc - True: {results_3['probe_acc_true']:.4f}, Pred: {results_3['probe_acc_pred']:.4f}")
    print(f"  N samples: {results_3['n_samples']}")

    # Issue 4: Memory Sufficiency
    print("\n" + "=" * 60)
    print("ISSUE 4: PERSISTENT MEMORY / STATE SUFFICIENCY")
    print("=" * 60)
    results_4 = measure_memory_sufficiency(model, prefix, target, config)
    print(f"  Standard AR - Token acc: {results_4['standard_ar_token_acc']:.4f}, Exact: {results_4['standard_ar_exact_acc']:.4f}")
    print(f"  Transition only - Token acc: {results_4['transition_only_token_acc']:.4f}, Exact: {results_4['transition_only_exact_acc']:.4f}")
    print(f"  Speculative acceptance: {results_4['speculative_acceptance_rate']:.4f}")

    # Issue 5: Backbone Quality
    print("\n" + "=" * 60)
    print("ISSUE 5: BACKBONE QUALITY (LINEAR PROBES)")
    print("=" * 60)
    results_5 = probe_backbone_quality(model, val_loader, config)
    for k, v in results_5.items():
        print(f"  {k}: {v:.4f}")

    # Issue 6: Speculative Decoding
    print("\n" + "=" * 60)
    print("ISSUE 6: SPECULATIVE DECODING USEFULNESS")
    print("=" * 60)
    results_6 = benchmark_speculative(model, val_loader, config)
    for k, v in results_6.items():
        print(f"  {k}: {v:.4f}")

    # Save results
    import json
    all_results = {
        'issue1_rollout_mismatch': results_1,
        'issue2_horizon_error': results_2,
        'issue3_collapse': results_3,
        'issue4_memory_sufficiency': results_4,
        'issue5_backbone_probe': results_5,
        'issue6_speculative': results_6,
    }

    with open('checkpoint_analysis_results.json', 'w') as f:
        # Convert tensors to scalars
        def convert(obj):
            if isinstance(obj, (torch.Tensor, np.generic)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        json.dump(convert(all_results), f, indent=2)

    print("\n" + "=" * 60)
    print("RESULTS SAVED TO checkpoint_analysis_results.json")
    print("=" * 60)


if __name__ == "__main__":
    main()