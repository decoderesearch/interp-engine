"""What the lens endpoint loads once: the fitted Jacobian lens, and the word mask.

A Jacobian lens file is a ``torch.save`` of ``{"J": {layer: [d_model, d_model]}, ...}``, the format
the J-Lens fitter writes, or a J++ Lens file, which holds the same maps under
``parameters.jacobians``. It is read from a local path or from ``hf://<owner>/<repo>/<path>``.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any

import torch

HF_PREFIX = "hf://"


def _local_path(source: str) -> Path:
    if not source.startswith(HF_PREFIX):
        return Path(source)
    parts = source[len(HF_PREFIX) :].split("/")
    if len(parts) < 3 or not all(parts):
        raise ValueError(f"{source!r}: an hf:// source is hf://<owner>/<repo>/<path in the repo>")
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download("/".join(parts[:2]), "/".join(parts[2:])))


def jacobian_mapping(checkpoint: Any) -> dict | None:
    """The ``{layer: J_bar}`` mapping of a lens file: ``J`` (Neuronpedia), or
    ``parameters.jacobians`` (a J++ Lens file from safety-research/jpp_lens). None for neither."""
    if not isinstance(checkpoint, dict):
        return None
    if isinstance(checkpoint.get("J"), dict):
        return checkpoint["J"]
    parameters = checkpoint.get("parameters")
    if isinstance(parameters, dict) and isinstance(parameters.get("jacobians"), dict):
        return parameters["jacobians"]
    return None


def load_jacobians(
    source: str, *, n_layers: int, d_model: int, dtype: torch.dtype, device: torch.device | str | None = None
) -> dict[int, torch.Tensor]:
    """One ``[d_model, d_model]`` J_bar per layer, at ``dtype`` on ``device``. Refuses a lens for another model."""
    checkpoint = torch.load(_local_path(source), map_location="cpu", weights_only=True)
    mapping = jacobian_mapping(checkpoint)
    if mapping is None:
        raise ValueError(f"{source!r} is not a Jacobian lens file: it has no 'J' mapping")
    out: dict[int, torch.Tensor] = {}
    for key, matrix in mapping.items():
        layer = int(key)
        if not 0 <= layer < n_layers:
            raise ValueError(f"{source!r} has a J_bar for layer {layer}; this model has {n_layers} layers")
        if tuple(matrix.shape) != (d_model, d_model):
            raise ValueError(
                f"{source!r} layer {layer} J_bar is {tuple(matrix.shape)}; this model needs {(d_model, d_model)}"
            )
        out[layer] = matrix.to(device=device, dtype=dtype)
    if not out:
        raise ValueError(f"{source!r} holds no J_bar")
    return out


def is_word_like(token: str) -> bool:
    """Not blank, not a special token, and only letters and digits, with ' - ’ allowed inside."""
    stripped = token.strip()
    if not stripped or "<|" in stripped or (stripped.startswith("<") and stripped.endswith(">")):
        return False
    last = len(stripped) - 1
    return all(
        unicodedata.category(ch)[0] in ("L", "N") or (0 < i < last and ch in ("'", "-", "\u2019"))
        for i, ch in enumerate(stripped)
    )


def word_mask(tokenizer: Any, vocab_size: int) -> torch.Tensor:
    """``[vocab_size]`` bool, True where the id decodes to a word-like token. One decode per id."""
    flags = torch.zeros(vocab_size, dtype=torch.bool)
    for token_id in range(vocab_size):
        try:
            text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        except Exception:
            continue
        flags[token_id] = is_word_like(text)
    return flags
