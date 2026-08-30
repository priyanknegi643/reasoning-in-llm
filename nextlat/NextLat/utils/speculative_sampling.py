"""
Speculative decoding implementation adapted from
https://github.com/feifeibear/LLMSpeculativeSampling.

Algorithm: Leviathan et al. (2023),
"Fast inference from transformers via speculative decoding" (ICML 2023).

Note: this implementation does not include KV-cache optimization.
"""

import torch
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional, Tuple


def sample_from_probs(probs: torch.Tensor) -> torch.Tensor:
    return torch.multinomial(probs, num_samples=1)


def normalize_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    temperature = max(float(temperature), 1e-6)
    logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(int(top_k), logits.size(-1))
        v, _ = torch.topk(logits, k)
        logits = logits.masked_fill(logits < v[:, [-1]], -float("Inf"))

    if top_p is not None and 0.0 < float(top_p) < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > float(top_p)
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(sorted_mask, -float("Inf"))
        logits = torch.full_like(logits, -float("Inf"))
        logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)

    return F.softmax(logits, dim=-1)


@torch.inference_mode()
def speculative_decode_v2(
    idx: torch.Tensor,
    max_new_tokens: int,
    gamma: int,
    propose_fn: Callable[[torch.Tensor, int], Tuple[torch.Tensor, List[torch.Tensor]]],
    target_probs_fn: Callable[
        [torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
    ],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Speculative decoding loop.

    Args:
        idx: input prefix ids [B, T].
        max_new_tokens: number of new tokens to generate.
        gamma: drafted tokens per speculative round.
        propose_fn: returns (draft_tokens [1, S], q_probs_steps list len=S each [1, V]).
        target_probs_fn: returns (p_probs_steps [1, S, V], p_next_probs [1, V]).
    """
    assert idx.dim() == 2, "idx must be rank-2 [batch, seq]"
    assert gamma >= 1, "gamma must be >= 1"

    device = idx.device
    generated = []
    total_accepted = 0
    total_proposed = 0
    total_proposed_excl_pos1_all_drafted = 0
    total_target_samples = 0
    total_resamples = 0
    # Per-position acceptance stats for drafted tokens in each speculative round.
    # Position is 1-indexed up to gamma.
    proposed_by_pos = [0 for _ in range(gamma)]
    accepted_by_pos = [0 for _ in range(gamma)]

    for b in range(idx.size(0)):
        seq = idx[b : b + 1]
        target_len = seq.size(1) + max_new_tokens

        while seq.size(1) < target_len:
            steps_to_propose = min(gamma, target_len - seq.size(1))
            draft_tokens, q_probs_steps = propose_fn(seq, steps_to_propose)
            steps = draft_tokens.size(1)
            total_proposed += steps
            total_proposed_excl_pos1_all_drafted += max(steps - 1, 0)

            p_probs_steps, p_next_probs = target_probs_fn(seq, draft_tokens)

            accepted_prefix = seq
            rejected_at = -1
            for i in range(steps):
                proposed_by_pos[i] += 1
                tok_idx = draft_tokens[:, i : i + 1]
                q_prob_tok = (
                    q_probs_steps[i].gather(dim=1, index=tok_idx).squeeze(1).squeeze(0)
                ).clamp_min(1e-12)
                p_prob_tok = (
                    p_probs_steps[:, i, :]
                    .gather(dim=1, index=tok_idx)
                    .squeeze(1)
                    .squeeze(0)
                )
                accept_prob = torch.minimum(
                    torch.ones((), device=device), p_prob_tok / q_prob_tok
                )
                if torch.rand((), device=device) < accept_prob:
                    accepted_prefix = torch.cat(
                        (accepted_prefix, draft_tokens[:, i : i + 1]), dim=1
                    )
                    total_accepted += 1
                    accepted_by_pos[i] += 1
                else:
                    rejected_at = i
                    break

            if rejected_at >= 0:
                p_dist = p_probs_steps[:, rejected_at, :]
                q_dist = q_probs_steps[rejected_at]
                correction = torch.clamp(p_dist - q_dist, min=0.0)
                correction_sum = correction.sum(dim=-1, keepdim=True)
                use_target = correction_sum <= 0
                correction = correction / correction_sum.clamp_min(1e-12)
                corrected = torch.where(use_target, p_dist, correction)
                next_tok = sample_from_probs(corrected)
                total_resamples += 1
            else:
                next_tok = sample_from_probs(p_next_probs)
                total_target_samples += 1

            seq = torch.cat((accepted_prefix, next_tok), dim=1)

        generated.append(seq)

    output = torch.cat(generated, dim=0)
    stats = {
        "accepted_tokens": float(total_accepted),
        "proposed_tokens": float(total_proposed),
        "rejected_tokens": float(total_proposed - total_accepted),
        "target_samples": float(total_target_samples),
        "resamples": float(total_resamples),
    }
    # Overall acceptance intentionally excludes position 1 because that position is the
    # same as next token prediction; almost always accepted for these draft-target pairs.
    # acceptance_rate: accepted pos-2+ / all drafted pos-2+ tokens.
    accepted_excl_pos1 = float(sum(accepted_by_pos[1:])) if gamma > 1 else 0.0
    stats["accepted_tokens_pos2plus"] = accepted_excl_pos1
    stats["proposed_tokens_pos2plus_all_drafted"] = float(
        total_proposed_excl_pos1_all_drafted
    )
    stats["acceptance_rate"] = (
        accepted_excl_pos1 / float(total_proposed_excl_pos1_all_drafted)
        if total_proposed_excl_pos1_all_drafted > 0
        else 0.0
    )
    # Average accepted drafted tokens beyond position 1 per speculative step.
    # One speculative step ends either with target sampling (no rejection) or
    # resampling (after a rejection), so total steps = target_samples + resamples.
    total_spec_steps = float(total_target_samples + total_resamples)
    stats["avg_accepted_tokens_per_step"] = (
        accepted_excl_pos1 / total_spec_steps if total_spec_steps > 0.0 else 0.0
    )
    for i in range(1, gamma):
        pos = i + 1
        proposed_pos = float(proposed_by_pos[i])
        accepted_pos = float(accepted_by_pos[i])
        stats[f"proposed_tokens_pos_{pos}"] = proposed_pos
        stats[f"accepted_tokens_pos_{pos}"] = accepted_pos
        stats[f"acceptance_rate_pos_{pos}"] = (
            accepted_pos / proposed_pos if proposed_pos > 0.0 else 0.0
        )
    return output, stats
