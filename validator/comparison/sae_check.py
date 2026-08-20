"""Optional SAE spot-check: encode a captured activation and summarize the feature acts, so we
can assert that engines which agree on the residual also agree on SAE features.

Defensive: any failure (sae_lens missing, SAE unavailable, API drift) returns ``None`` and the
caller records the SAE check as skipped rather than failing the whole capture.
"""

from __future__ import annotations

import numpy as np


def _load_sae(release: str, sae_id: str, device: str, loader: str | None = None):
    from sae_lens import SAE

    kwargs: dict = {"release": release, "sae_id": sae_id, "device": device}
    if loader == "dictionary_learning":
        # Non-registry dictionary_learning SAEs (e.g. adamkarvonen/*) need an explicit converter.
        from sae_lens.loading.pretrained_sae_loaders import dictionary_learning_sae_huggingface_loader_1

        kwargs["converter"] = dictionary_learning_sae_huggingface_loader_1
    loaded = SAE.from_pretrained(**kwargs)
    # sae_lens has returned either the SAE or a (sae, cfg, sparsity) tuple across versions.
    return loaded[0] if isinstance(loaded, tuple) else loaded


def encode_summary(
    activation: np.ndarray, release: str, sae_id: str, device: str = "cpu", loader: str | None = None
) -> dict | None:
    """Return a small, comparable summary of the SAE features at the LAST token.

    ``activation`` is ``[seq, d_model]`` (the captured point the SAE was trained on).
    """
    try:
        import torch

        sae = _load_sae(release, sae_id, device, loader)
        acts = torch.tensor(np.asarray(activation, dtype=np.float32), device=device)
        with torch.no_grad():
            feats = sae.encode(acts)  # [seq, d_sae]
        last = feats[-1].float().cpu()
        nonzero = last.abs() > 1e-6
        top_val, top_idx = last.max(dim=-1)
        return {
            "release": release,
            "sae_id": sae_id,
            "last_token_l0": int(nonzero.sum().item()),
            "last_token_top_index": int(top_idx.item()),
            "last_token_top_value": float(top_val.item()),
            "acts_sum": float(last.sum().item()),
        }
    except Exception as exc:  # noqa: BLE001
        return {"release": release, "sae_id": sae_id, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
