"""Rebuilding the routing decision where no module boundary carries it.

The MoE routing points are read off the router module wherever one runs, and that is still the rule --
see :mod:`interp_engine.points`. This module is the exception it makes room for: transformers' MXFP4
path for gpt-oss replaces the sparse block's ``forward`` and routes inline, handing the logits to a
Triton kernel that returns only the combined output. The router module is in the tree, correct, and
never called, so ``expert_weights`` and ``expert_indices`` have no boundary to be read from -- and the
alternative to rebuilding them is not reading them, it is not having them at all.

**When a recompute is allowed.** Two conditions, both required:

1. *It costs nothing measurable.* The derivation runs on a tensor the pass already captured -- here a
   ``[tokens, experts]`` logit matrix, a top-k and a softmax over ~32 values per token. No second
   forward, no extra module kept alive, no memory beyond the k-wide result. A recompute that needed
   its own forward pass, or that held the whole attention matrix to get there, would not qualify:
   that is the difference between this and ``attn_probs``, which is a recompute *because* the tensor
   cannot exist otherwise and is priced accordingly (:mod:`interp_engine.attn_scores`).
2. *Its correctness is verified against the read path, on a real checkpoint.* Not argued from the
   modeling code -- asserted, by a test that reads the decision off the eager router and compares.
   The conventions in use are mutually incompatible and all yield k weights summing to 1
   (:data:`interp_engine.facts.ROUTING_CONVENTIONS` lists them), so a wrong guess is plausible and
   silent, and "I read the source" is exactly the evidence that failed to distinguish them before.

What is *not* allowed is recomputing where a read is possible. A family whose router runs has its
decision read even though the derivation would agree, because the read cannot drift when the family
changes its convention and a hardcoded derivation can.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

#: The points this module can rebuild. ``router_logits`` is not among them: it is *read*, off the
#: block's own output tuple, even on the fused path (:meth:`interp_engine.arch.ArchSpec.inline_routing_logits`).
DERIVED_POINTS: tuple[str, ...] = ("expert_weights", "expert_indices")

#: The source it rebuilds them from, which the capture path adds to its request when needed.
SOURCE_POINT = "router_logits"

Derivation = Callable[[torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]]


def _topk_then_softmax(logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """gpt-oss: select on the raw logits, then softmax only the survivors.

    Line for line what ``GptOssTopKRouter.forward`` does, and what the MXFP4 path's kernel does with
    ``sm_first=False`` -- its own torch reference (``routing_torch``) is a ``topk`` followed by a
    ``softmax`` over the k selected values. Both orderings appear in the wild and the other one
    (softmax over all experts, then select, then renormalize) also returns k weights summing to 1, so
    which is right here is settled by the test, not by this comment.
    """
    values, indices = torch.topk(logits, top_k, dim=-1)
    return torch.softmax(values, dim=-1, dtype=values.dtype), indices


DERIVATIONS: dict[str, Derivation] = {"topk_then_softmax": _topk_then_softmax}


def derive(convention: str, logits: torch.Tensor, top_k: int) -> dict[str, torch.Tensor]:
    """``{"expert_weights": ..., "expert_indices": ...}`` from the logits the model routed on.

    Both at once, from one top-k, because they are two halves of a single decision: returning them
    from separate calls would let a caller pair weights with indices from different sorts.
    """
    derivation = DERIVATIONS.get(convention)
    if derivation is None:
        raise ValueError(f"No routing derivation registered under {convention!r}; have {sorted(DERIVATIONS)}")
    if top_k <= 0:
        raise ValueError(
            f"Cannot rebuild the routing decision without the top-k: this checkpoint's config reports "
            f"{top_k} experts per token. Read them off the router instead (load without the fused path)."
        )
    if logits.shape[-1] < top_k:
        raise ValueError(
            f"Router logits are {logits.shape[-1]} wide but the config routes to {top_k} experts, so this "
            "is not the whole logit vector -- refusing to select a top-k from part of it."
        )
    weights, indices = derivation(logits, top_k)
    return {"expert_weights": weights, "expert_indices": indices}
