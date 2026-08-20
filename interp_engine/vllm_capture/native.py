"""vLLM's own residual extraction, as an alternative to hooking the tree.

Unrelated to the hook machinery in the rest of the package: this asks the engine to write
residuals out itself at construction time. Kept because it needs no ``enforce_eager``, and
limited because it only reaches whole-layer residuals -- intra-block taps (``mlp_in``, ``z``,
``value``) and attention recompute still go through the hooks.
"""

from __future__ import annotations

from typing import Any

import torch

# =============================================================================
# Native residual capture (vLLM >= 0.18 extract_hidden_states) -- PREFERRED path
# for the residual stream. No hooks / no monkeypatching: vLLM writes per-layer
# resid to safetensors via the ExampleHiddenStatesConnector.
#
# Validated on vLLM 0.25.1 (scripts/vllm_native_extract_check.py): aux layer_id L
# == resid ENTERING layer L == resid_post[L-1]; layer_ids 1..N-1 match HF exactly
# (cos ~1.0). The FINAL layer (id == num_hidden_layers) is broken in 0.25.1 (PR
# #36063 not landed), so we extract 1..N-1 (i.e. resid_post[0..N-2]) and handle
# the final layer via the model's real output logits (caller's concern).
#
# The forward-hook capture below stays for intra-block taps (mlp_in/z/value) and
# attention recompute that native extraction does not provide.
# =============================================================================

DEFAULT_HS_STORAGE_PATH = "/dev/shm/ie_hidden_states"


def extract_hidden_states_engine_kwargs(
    num_hidden_layers: int,
    *,
    shared_storage_path: str = DEFAULT_HS_STORAGE_PATH,
    include_final: bool = False,
) -> tuple[dict[str, Any], list[int]]:
    """Build the vLLM construction kwargs that turn on native residual extraction.

    Splat the returned dict into ``LLM(...)`` / ``AsyncEngineArgs`` and keep the
    returned ``layer_ids`` (their order matches the safetensors' layer axis).

    ``include_final`` requests ``num_hidden_layers`` too (broken on 0.25.1 -- leave
    False until the vLLM bump; see plan ``bump-vllm-final-layer``).
    """
    last = num_hidden_layers + 1 if include_final else num_hidden_layers
    layer_ids = list(range(1, last))  # 1..N-1 (== resid_post[0..N-2]), optionally +N
    kwargs = {
        "speculative_config": {
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": layer_ids}},
        },
        "kv_transfer_config": {
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {"shared_storage_path": shared_storage_path},
        },
        # extract_hidden_states requires chunked prefill off.
        "enable_chunked_prefill": False,
    }
    return kwargs, layer_ids


def read_resid_post_from_output(
    request_output: object,
    layer_ids: list[int],
    *,
    cleanup: bool = True,
) -> dict[int, torch.Tensor]:
    """Read native-extracted residuals for one request -> ``{resid_post_index: [tokens, hidden]}``.

    ``request_output`` is a vLLM ``RequestOutput`` whose ``kv_transfer_params`` holds
    the safetensors path. ``layer_ids`` are those returned by
    :func:`extract_hidden_states_engine_kwargs`; each maps to ``resid_post[id-1]``.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (  # pyright: ignore[reportMissingImports]
        cleanup_hidden_states,
        load_hidden_states,
    )

    params = getattr(request_output, "kv_transfer_params", None) or {}
    path = params.get("hidden_states_path")
    if path is None:
        raise RuntimeError(
            "RequestOutput has no kv_transfer_params['hidden_states_path']; was the "
            "engine constructed with extract_hidden_states_engine_kwargs()?"
        )
    data = load_hidden_states(path)
    hs = data["hidden_states"]  # [num_tokens, len(layer_ids), hidden]
    out = {int(lid) - 1: hs[:, i, :] for i, lid in enumerate(layer_ids)}
    if cleanup:
        cleanup_hidden_states(path)
    return out
