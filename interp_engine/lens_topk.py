"""The top-k of a lens read-out, as every backend ranks it.

Torch only, so the vLLM worker imports it without the rest of the engine.
"""

from __future__ import annotations

import torch


def lens_topk(
    logits: torch.Tensor,
    *,
    top_n: int,
    mask: torch.Tensor | None = None,
    rows_per_group: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(top_idx, top_probs)``, each ``[n_rows, k]``, from ``[n_rows, vocab]`` logits.

    ``logits`` is laid out as contiguous groups of ``rows_per_group`` (one group per position, the
    group's last row being the final layer). ``mask``, when given, is a 1-D bool vocab mask used for
    RANKING only: ``log_z`` is taken before it is applied, so probabilities stay normalised over the
    whole vocab, and each group's final row keeps its true top-1 even where that token is non-word.

    float32 throughout: ``log_z`` sums the whole vocab and ``top_logits - log_z`` cancels two large
    nearly equal values, which in bf16 reports probabilities of 1.0 and top-k rows summing above 1.
    """
    if top_n <= 0:
        raise ValueError(f"top_n must be > 0, got {top_n}")
    if rows_per_group <= 0:
        raise ValueError(f"rows_per_group must be > 0, got {rows_per_group}")
    logits_f = logits.float()
    log_z = logits_f.logsumexp(dim=-1, keepdim=True)
    ranked = logits_f
    if mask is not None:
        # Align to the logits vocab: tokenizer.vocab_size can under-count a padded embedding
        # table (Llama-3: 128000 vs 128256). Extra slots are never word-like.
        vocab = int(ranked.shape[-1])
        if mask.dim() != 1:
            raise ValueError(f"word_mask must be 1-D, got shape {tuple(mask.shape)}")
        mask = mask.to(device=ranked.device, dtype=torch.bool)
        if mask.shape[0] < vocab:
            mask = torch.nn.functional.pad(mask, (0, vocab - mask.shape[0]), value=False)
        elif mask.shape[0] > vocab:
            mask = mask[:vocab]
        n_rows = int(ranked.shape[0])
        # Tensor ops rather than a per-group loop: each int()/float() readback is a device sync.
        finals = torch.arange(rows_per_group - 1, n_rows, rows_per_group, device=ranked.device)
        keep_idx = ranked[finals].argmax(dim=-1, keepdim=True)
        keep_val = ranked[finals].gather(-1, keep_idx)
        # `.float()` returns `logits` itself when it is already float32; do not write into it.
        if ranked is logits:
            ranked = ranked.clone()
        ranked.masked_fill_(~mask.unsqueeze(0), torch.finfo(ranked.dtype).min)
        ranked[finals.unsqueeze(-1), keep_idx] = keep_val
    k = min(top_n, int(ranked.shape[-1]))
    top_idx = ranked.topk(k, dim=-1).indices
    top_probs = (ranked.gather(-1, top_idx) - log_z).exp()
    return top_idx, top_probs
