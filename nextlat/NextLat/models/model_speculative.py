from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

import torch

from utils.speculative_sampling import normalize_logits, sample_from_probs


class SpeculativeModel(ABC):
    """Common interface for models with fast speculative proposal heads."""

    def _unwrap_module(self) -> torch.nn.Module:
        return self.model.module if hasattr(self.model, "module") else self.model

    def speculative_forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_ids)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise RuntimeError(
                "Expected logits tensor with shape [B, T, V] for speculative decoding."
            )
        return logits

    def _propose_autoregressive(
        self,
        seq: torch.Tensor,
        steps_to_propose: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        cur = seq
        drafted: List[torch.Tensor] = []
        q_probs_steps: List[torch.Tensor] = []
        for _ in range(steps_to_propose):
            logits = self.speculative_forward_logits(cur)
            q_probs = normalize_logits(
                logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
            )
            tok = sample_from_probs(q_probs)
            drafted.append(tok)
            q_probs_steps.append(q_probs)
            cur = torch.cat((cur, tok), dim=1)
        return torch.cat(drafted, dim=1), q_probs_steps

    @abstractmethod
    def speculative_propose(
        self,
        seq: torch.Tensor,
        steps_to_propose: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Return drafted tokens and per-step q-probabilities."""

    def build_speculative_propose_fn(
        self,
        *,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> Callable[[torch.Tensor, int], Tuple[torch.Tensor, List[torch.Tensor]]]:
        return lambda seq, s: self.speculative_propose(
            seq,
            s,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    def build_speculative_target_probs_fn(
        self,
        *,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        def target_probs_fn(
            seq: torch.Tensor,
            draft_tokens: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            full = torch.cat((seq, draft_tokens), dim=1)
            logits = self.speculative_forward_logits(full)
            prefix_len = seq.size(1)
            steps = draft_tokens.size(1)
            step_logits = logits[:, prefix_len - 1 : prefix_len - 1 + steps, :]
            next_logits = logits[:, prefix_len - 1 + steps, :]
            probs_steps = []
            for t in range(step_logits.size(1)):
                probs_steps.append(
                    normalize_logits(
                        step_logits[:, t, :],
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )
                )
            p_probs_steps = torch.stack(probs_steps, dim=1)
            p_next_probs = normalize_logits(
                next_logits, temperature=temperature, top_k=top_k, top_p=top_p
            )
            return p_probs_steps, p_next_probs

        return target_probs_fn
