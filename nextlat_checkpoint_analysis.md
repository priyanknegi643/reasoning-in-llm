# NextLat Checkpoint Analysis: Deep-Dive on 6 Issues

**Date:** 2026-09-14  
**Checkpoints Available:** Only seed 1234 has saved checkpoints (`ckpt_iter_5000_0.6055.pt`, `ckpt_iter_20001_0.7224.pt`) — other 5 seeds ran with `save_last_checkpoint: false`, `save_best_checkpoint: false`.

---

## Issue 1: Generated-Token Rollout Mismatch

### The Problem
**Training (teacher-forced multi-step):**
```
h_t → dyn(h_t, E(x_{t+1})) → ĥ_{t+1}
ĥ_{t+1} → dyn(ĥ_{t+1}, E(x_{t+2})) → ĥ_{t+2}
...
```
- Latent predictions are **recursive** (ĥ feeds into next step)
- But token embeddings are **ground-truth** (E(x_{t+k}) from data)

**Free-running (speculative decoding / inference):**
```
h_t → dyn(h_t, E(x̂_{t+1})) → ĥ_{t+1}
ĥ_{t+1} → dyn(ĥ_{t+1}, E(x̂_{t+2})) → ĥ_{t+2}
...
```
- Both latent AND tokens are model-generated
- Distribution shift: model sees its own errors compounded

### What We Can Measure With Existing Checkpoint

```python
# File: inspect_rollout_mismatch.py
import torch
from nextlat.NextLat.models.model_nextlat import NextLat, NextLatConfig
from nextlat.NextLat.data.stargraph import StarGraphDataModule, Tokenizer
import lightning as L
from omegaconf import OmegaConf

# Load checkpoint
ckpt_path = "nextlat/NextLat/output/stargraph/NextLat-proj_factor0.5_lambda_kl1.0_mtp_horizon8_seed1234_lambda_mse1.0/ckpt_iter_20001_0.7224.pt"
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# Reconstruct config (from materialized_config.yaml)
config = OmegaConf.load("nextlat/NextLat/output/stargraph/NextLat-proj_factor0.5_lambda_kl1.0_mtp_horizon8_seed1234_lambda_mse1.0/materialized_config.yaml")

# Initialize model
fabric = L.Fabric(accelerator='cpu')
tokenizer = Tokenizer(config.data.stargraph_max_nodes)
# ... setup model, load weights

# Test on validation data
from nextlat.NextLat.data.stargraph import StarGraphDataModule
datamodule = StarGraphDataModule(fabric, config)
datamodule.update_config(config)
val_loader = datamodule.val_dataloader()

# For a batch, compare:
# 1. Teacher-forced rollout (what training does)
# 2. Free-running rollout (what speculative decoding does)
# 3. Measure divergence in latent space and prediction accuracy
```

**Key Metrics to Compute:**
| Metric | Teacher-Forced | Free-Running | Interpretation |
|--------|----------------|--------------|----------------|
| MSE(ĥ_{t+k}, h_{t+k}) | Training loss | Eval metric | How much error accumulates |
| KL(teacher_logits \| student_logits) | Training loss | Eval metric | Distributional drift |
| Token prediction accuracy @ k | N/A | Eval metric | Downstream task degradation |
| Latent cosine similarity | 1.0 (by def) | Eval metric | Representation drift |

---

## Issue 2: Long-Horizon Latent Error Accumulation

### The Problem
Training horizon = 8 (mtp_horizon). But PathStar requires 10 target tokens. What happens at k=9,10,11...?

### Measurement Protocol

```python
# File: measure_horizon_error.py
@torch.inference_mode()
def measure_horizon_error(model, batch, max_k=20):
    """Roll out transition with teacher-forced tokens up to max_k"""
    core = model._unwrap_module()
    _, hidden_states = core(batch, return_hidden_states=True)
    token_embeds = core.token_embedding(batch)
    
    results = []
    h_true = hidden_states  # (B, T, d)
    
    # Start from position context_length (after prefix)
    start_pos = config.model.context_length
    
    pred_h = h_true[:, start_pos-1:start_pos, :]  # h at context boundary
    
    for k in range(1, max_k+1):
        if start_pos + k >= h_true.shape[1]:
            break
            
        # Teacher-forced token
        next_tok_emb = token_embeds[:, start_pos + k : start_pos + k + 1, :]
        true_h = h_true[:, start_pos + k : start_pos + k + 1, :]
        
        # Predict
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
```

**What to Plot:**
- MSE vs k (should grow if error accumulates)
- Cosine similarity vs k (should decay)
- Token accuracy vs k (should drop)

**Expected Pattern from Training Losses (metrics.csv):**
```
loss_token_1: ~0.002  (k=1)
loss_token_2: ~0.003
loss_token_3: ~0.008
loss_token_4: ~0.013
loss_token_5: ~0.020
loss_token_6: ~0.030
loss_token_7: ~0.047
loss_token_8: ~0.088
```
Already clear growth — extrapolation to k=10 would be ~0.15-0.20.

---

## Issue 3: Representation Collapse / Degeneration

### The Problem
Does the transition cause latent representations to:
- Lose effective rank (concentrate in subspace)?
- Lose task-relevant information (even if rank stays high)?
- Converge to a fixed point (all ĥ become similar)?

### Measurement Protocol

```python
# File: measure_collapse.py
@torch.inference_mode()
def measure_collapse(model, dataloader, max_steps=100):
    core = model._unwrap_module()
    
    all_h_true = []
    all_h_pred = []
    
    for batch in dataloader:
        _, hidden_states = core(batch, return_hidden_states=True)
        token_embeds = core.token_embedding(batch)
        
        # Teacher-forced rollout for horizon=8
        h_true = hidden_states[:, :-1]  # (B, T-1, d)
        h_pred = hidden_states[:, :-1].clone()
        next_emb = token_embeds[:, 1:]
        
        for k in range(8):
            h_pred = core.dynamics_model(h_pred, next_emb)
            next_emb = next_emb[:, 1:]  # shift
            if next_emb.shape[1] == 0:
                break
        
        all_h_true.append(h_true.reshape(-1, h_true.shape[-1]))
        all_h_pred.append(h_pred.reshape(-1, h_pred.shape[-1]))
    
    H_true = torch.cat(all_h_true, dim=0)   # (N, d)
    H_pred = torch.cat(all_h_pred, dim=0)   # (N, d)
    
    # 1. Numerical rank
    rank_true = torch.linalg.matrix_rank(H_true, atol=1e-3, rtol=1e-3).item()
    rank_pred = torch.linalg.matrix_rank(H_pred, atol=1e-3, rtol=1e-3).item()
    
    # 2. Effective rank (entropy of singular values)
    def effective_rank(X):
        U, S, V = torch.svd(X)
        p = S / S.sum()
        p = torch.clamp(p, min=1e-12)
        return torch.exp(-(p * p.log()).sum()).item()
    
    eff_rank_true = effective_rank(H_true)
    eff_rank_pred = effective_rank(H_pred)
    
    # 3. Pairwise cosine similarity (collapse indicator)
    # Sample subset for efficiency
    idx = torch.randperm(H_pred.shape[0])[:2000]
    H_sample = H_pred[idx]
    cos_sims = F.cosine_similarity(H_sample.unsqueeze(1), H_sample.unsqueeze(0), dim=-1)
    mean_cos = cos_sims.mean().item()
    max_cos = cos_sims[cos_sims < 1.0].max().item()
    
    # 4. Task-relevant info: probe accuracy
    # Train linear probe on h_pred vs h_true to predict next token
    from sklearn.linear_model import LogisticRegression
    probe_true = LogisticRegression(max_iter=1000).fit(
        H_true.numpy(), batch[:, 1:].flatten().numpy()
    )
    probe_pred = LogisticRegression(max_iter=1000).fit(
        H_pred.numpy(), batch[:, 1:].flatten().numpy()
    )
    
    return {
        'rank_true': rank_true, 'rank_pred': rank_pred,
        'eff_rank_true': eff_rank_true, 'eff_rank_pred': eff_rank_pred,
        'mean_cos_sim': mean_cos, 'max_cos_sim': max_cos,
        'probe_acc_true': probe_true.score(...),
        'probe_acc_pred': probe_pred.score(...)
    }
```

**Interpretation Guide:**
| Observation | Meaning |
|-------------|---------|
| rank_pred << rank_true | **Collapse** — transition compresses representations |
| eff_rank_pred << eff_rank_true | **Concentration** — mass in fewer directions |
| mean_cos_sim > 0.9 | **Point collapse** — all latents becoming similar |
| probe_acc_pred ≈ probe_acc_true | **Benign compression** — task info preserved |
| probe_acc_pred << probe_acc_true | **Functional degradation** — task info lost |

---

## Issue 4: Persistent-Memory / State Sufficiency

### The Problem
Does a single hidden vector `h_t` (updated via transition) suffice to replace full KV-cache attention to earlier context?

### Key Distinction
| Regime | What the Model Sees |
|--------|---------------------|
| **Standard AR generation** | Full KV cache (all previous tokens via attention) |
| **Speculative decoding (draft)** | Only `h_t` + transition (no KV cache for draft steps) |
| **NextLat training** | Full Transformer on prefix, then transition on suffix |

### Measurement Protocol

```python
# File: measure_memory_sufficiency.py
@torch.inference_mode()
def compare_generation_modes(model, prefix, target_len=10):
    core = model._unwrap_module()
    
    # Mode 1: Standard AR (full attention)
    standard_out = core.generate(prefix, max_new_tokens=target_len)
    
    # Mode 2: Speculative (uses transition for draft)
    draft_tokens, _ = core.speculative_propose(prefix, target_len, temp=1.0, top_k=None, top_p=None)
    # Then verify with target model (which is same model here)
    
    # Mode 3: Pure transition rollout (no Transformer after prefix)
    _, hidden_states = core(prefix, return_hidden_states=True)
    h = hidden_states[:, -1, :]
    transition_tokens = []
    for _ in range(target_len):
        logits = core.lm_head(h)
        tok = logits.argmax(-1, keepdim=True)
        transition_tokens.append(tok)
        h = core.dynamics_model(h, core.token_embedding(tok).squeeze(1))
    transition_out = torch.cat([prefix, torch.cat(transition_tokens, dim=1)], dim=1)
    
    # Compare all three to ground truth
    return {
        'standard_vs_target': accuracy(standard_out, target),
        'transition_vs_target': accuracy(transition_out, target),
        'speculative_acceptance_rate': compute_acceptance(...)
    }
```

**Critical Experiment:** On PathStar, the prefix is the graph description (56 tokens). The target is 10 path tokens.
- Standard AR: Transformer attends to full 56-token prefix at every step
- Transition-only: Only `h_56` carries all graph info — can it predict 10-step path?

---

## Issue 5: Benefit to Ordinary Autoregressive Generation

### The Problem
The transition is **never used** during standard `generate()`:
```python
# model_nextlat.py:706-740
def generate(self, idx, max_new_tokens, ...):
    for _ in range(max_new_tokens):
        logits = self.model(idx_cond)  # Standard Transformer forward
        logits = logits[:, -1, :] / temperature
        # ... sample next token
        idx = torch.cat((idx, idx_next), dim=1)
```
No call to `dynamics_model` anywhere.

### Does Auxiliary Training Help the Backbone?

**Controlled Experiment Design:**
1. Train NextLat (with MSE+KL) → evaluate standard `generate()`
2. Train GPT (same architecture, no NextLat losses) → evaluate standard `generate()`
3. Compare PathStar accuracy

**With Existing Checkpoints:**
- NextLat checkpoint: `ckpt_iter_20001_0.7224.pt` (seed 1234)
- Need: GPT baseline checkpoint with same config (not available — would need to train)

**Proxy Analysis (from metrics.csv):**
- NextLat NTP loss at step 20001: ~0.0002 (val/next_token_loss)
- Compare to GPT baseline from paper/logs if available

```python
# File: probe_backbone_quality.py
@torch.inference_mode()
def probe_hidden_states(model, dataloader):
    """Train linear probes on hidden states to predict future tokens"""
    core = model._unwrap_module()
    
    # Collect hidden states at each position
    all_h = []
    all_targets = []
    for batch in dataloader:
        _, h = core(batch, return_hidden_states=True)
        all_h.append(h[:, :-1].reshape(-1, h.shape[-1]))
        all_targets.append(batch[:, 1:].reshape(-1))
    
    H = torch.cat(all_h).numpy()
    Y = torch.cat(all_targets).numpy()
    
    # Probe: predict token at offset k
    for k in [1, 2, 3, 5, 8, 10]:
        # Shift targets
        # Train logistic regression
        # Report accuracy
        pass
```

If NextLat hidden states are more linearly decodable for future tokens than GPT, the auxiliary losses improved the backbone representation.

---

## Issue 6: Speculative Decoding Usefulness

### The Problem
Speculative decoding uses the transition to draft tokens:
```python
# model_nextlat.py:678-703
def speculative_propose(self, seq, steps_to_propose, ...):
    _, hidden_states = core(seq, return_hidden_states=True)
    state = hidden_states[:, -1, :]
    
    for _ in range(steps_to_propose):
        logits = core.lm_head(state)
        q_probs = normalize_logits(logits, ...)
        tok = sample_from_probs(q_probs)
        drafted.append(tok)
        next_token_emb = core.token_embedding(tok).squeeze(1)
        state = core.dynamics_model(state, next_token_emb)  # FREE-RUNNING
```

**Questions:**
1. **Draft quality**: What's the acceptance rate?
2. **Verification cost**: Does verification reject many tokens?
3. **Speedup**: Net tokens/sec including verification?

### Measurement Protocol

```python
# File: measure_speculative.py
from nextlat.NextLat.utils.speculative_sampling import speculative_decode_v2

@torch.inference_mode()
def benchmark_speculative(model, prompts, gamma=4, max_new_tokens=50):
    core = model._unwrap_module()
    
    def propose_fn(seq, steps):
        return core.speculative_propose(seq, steps, temp=1.0, top_k=None, top_p=None)
    
    def target_probs_fn(seq, draft_tokens):
        full = torch.cat([seq, draft_tokens], dim=1)
        logits = core(full)
        prefix_len = seq.size(1)
        steps = draft_tokens.size(1)
        step_logits = logits[:, prefix_len-1:prefix_len-1+steps, :]
        next_logits = logits[:, prefix_len-1+steps, :]
        # normalize and return
        return p_probs_steps, p_next_probs
    
    results = []
    for prompt in prompts:
        output, stats = speculative_decode_v2(
            prompt, max_new_tokens, gamma, propose_fn, target_probs_fn
        )
        results.append(stats)
    
    # Aggregate
    avg_acceptance = np.mean([r['acceptance_rate'] for r in results])
    avg_speedup = np.mean([r['avg_accepted_tokens_per_step'] for r in results])
    
    return {
        'acceptance_rate': avg_acceptance,
        'avg_accepted_per_step': avg_speedup,
        'theoretical_max_speedup': gamma,
        'actual_speedup': 1 + avg_speedup  # 1 target step + accepted drafts
    }
```

**Expected from Paper (FineWeb):**
- NextLat acceptance_rate ~ 0.6-0.7 at gamma=4
- Speedup ~ 1.6-1.8x

**On PathStar (short sequences, deterministic):**
- May be higher (less entropy)
- But sequence length only 68, so limited benefit

---

## Unified Checkpoint Inspection Script

```python
# File: run_full_analysis.py
"""
Run all 6 analyses on the available checkpoint.
"""
import torch
from omegaconf import OmegaConf
import lightning as L
from nextlat.NextLat.models.model_nextlat import NextLat, NextLatConfig
from nextlat.NextLat.data.stargraph import StarGraphDataModule, Tokenizer

# Load checkpoint and config
ckpt_path = "nextlat/NextLat/output/stargraph/NextLat-proj_factor0.5_lambda_kl1.0_mtp_horizon8_seed1234_lambda_mse1.0/ckpt_iter_20001_0.7224.pt"
config_path = "nextlat/NextLat/output/stargraph/NextLat-proj_factor0.5_lambda_kl1.0_mtp_horizon8_seed1234_lambda_mse1.0/materialized_config.yaml"

config = OmegaConf.load(config_path)
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# Setup
fabric = L.Fabric(accelerator='cuda' if torch.cuda.is_available() else 'cpu')
fabric.launch()
tokenizer = Tokenizer(config.data.stargraph_max_nodes)

datamodule = StarGraphDataModule(fabric, config)
datamodule.update_config(config)

# Initialize model
model = NextLat(NextLatConfig(**config.model))
model.setup_fabric(fabric)
model.load_checkpoint(ckpt_path)
model.eval()

val_loader = datamodule.val_dataloader()

# Run analyses
print("=" * 60)
print("ISSUE 1: Rollout Mismatch")
print("=" * 60)
results_1 = measure_rollout_mismatch(model, val_loader)

print("=" * 60)
print("ISSUE 2: Horizon Error Accumulation")
print("=" * 60)
results_2 = measure_horizon_error(model, next(iter(val_loader)))

print("=" * 60)
print("ISSUE 3: Representation Collapse")
print("=" * 60)
results_3 = measure_collapse(model, val_loader)

print("=" * 60)
print("ISSUE 4: Memory Sufficiency")
print("=" * 60)
batch = next(iter(val_loader))
prefix = batch[:, :config.model.context_length]
results_4 = compare_generation_modes(model, prefix)

print("=" * 60)
print("ISSUE 5: Backbone Quality (Probe)")
print("=" * 60)
results_5 = probe_hidden_states(model, val_loader)

print("=" * 60)
print("ISSUE 6: Speculative Decoding")
print("=" * 60)
prompts = [batch[i:i+1, :config.model.context_length] for i in range(min(10, batch.shape[0]))]
results_6 = benchmark_speculative(model, prompts)

# Save all results
import json
with open('checkpoint_analysis_results.json', 'w') as f:
    json.dump({
        'rollout_mismatch': results_1,
        'horizon_error': results_2,
        'collapse': results_3,
        'memory_sufficiency': results_4,
        'backbone_probe': results_5,
        'speculative': results_6
    }, f, indent=2)
```

---

## What You Need to Run This

1. **Install dependencies** (already in your venv):
   ```bash
   pip install scikit-learn  # for probes
   ```

2. **Run the unified script:**
   ```bash
   cd /workspace/Research
   python run_full_analysis.py
   ```

3. **Key outputs to look for:**
   - `horizon_error`: MSE/cosine/accuracy curves up to k=20
   - `collapse`: rank, effective rank, cosine similarity, probe accuracy
   - `memory_sufficiency`: transition-only vs standard AR accuracy on PathStar
   - `speculative`: acceptance rate, speedup

---

## What This Will Tell You

| Issue | What the Data Will Show |
|-------|-------------------------|
| 1. Rollout mismatch | Quantified gap between teacher-forced and free-running performance |
| 2. Horizon error | Whether k=9,10 (PathStar target) are already in error saturation regime |
| 3. Collapse | Whether transition *functionally* degrades representations (not just rank) |
| 4. Memory | Whether `h_t` alone suffices for PathStar (vs full attention) |
| 5. Backbone | Whether NextLat losses improved the Transformer's hidden states |
| 6. Speculative | Whether the transition provides practical speedup on this task |

---

## Critical Caveat

**Only seed 1234 has checkpoints.** The other 5 seeds (1235-1238) have metrics but no weights. For statistical significance, you'd need to:
1. Re-train seeds 1235-1238 with `save_last_checkpoint: true`
2. Or accept single-seed analysis (common in DL but weak)

The metrics.csv for all 5 seeds show **high variance in final accuracy** (0.29 to 0.97), suggesting seed matters enormously. Single-seed conclusions are risky.