"""Fit ``max_num_seqs`` to the Mamba state cache, so a hybrid trunk starts without a hand-set value.

With full decode CUDA graphs, vLLM needs one Mamba cache block per sequence, up to
``max_num_seqs`` (128 by default). It knows the block count only after it measures memory,
inside the engine-core child, and refuses to start when there are fewer blocks than sequences.
Static attention buffers take memory from that cache, so a config that starts without them can
fail with them. The parent sees only "Engine core initialization failed".

The worker half wraps vLLM's check: when it refuses, the worker writes the block count to the
file named by :data:`SEQ_FIT_ENV`, then raises as before. The parent half reads that file after
the failed start and starts again with ``max_num_seqs`` set to the count. A caller who sets
``max_num_seqs`` gets no retry: an explicit value wins, and fails the same way it did before.
"""

from __future__ import annotations

import inspect
import os
from typing import Any

# Path of the file the worker writes the Mamba block count to. The parent sets it before it
# starts the engine, and the engine-core and worker processes inherit it.
SEQ_FIT_ENV = "INTERP_ENGINE_SEQ_FIT_FILE"


def mamba_block_limit(kv_cache_config: Any, max_num_reqs: int | None) -> int | None:
    """Return the Mamba block count when it is below ``max_num_reqs``, else ``None``."""
    if kv_cache_config is None or max_num_reqs is None:
        return None
    if not getattr(kv_cache_config, "has_mamba_layers", False):
        return None
    blocks = int(getattr(kv_cache_config, "num_blocks", 0))
    if blocks <= 0 or max_num_reqs <= blocks:
        return None
    return blocks


def record_block_limit(blocks: int) -> None:
    """Write ``blocks`` to the file named by :data:`SEQ_FIT_ENV`. Do nothing when it is unset."""
    path = os.environ.get(SEQ_FIT_ENV)
    if not path:
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(blocks))


def read_block_limit(path: str) -> int | None:
    """Return the block count a worker wrote to ``path``, or ``None`` when none did."""
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
    except OSError:
        return None
    return int(text) if text.isdigit() and int(text) > 0 else None


def patch_mamba_seq_check() -> None:
    """Wrap ``CompilationConfig.resolve_cudagraph_mode_and_sizes`` to record the block count.

    The wrap changes no result. It only writes the count before vLLM's own error propagates.
    """
    try:
        from vllm.config.compilation import CompilationConfig  # pyright: ignore[reportMissingImports]
    except Exception:  # noqa: BLE001
        return
    orig = CompilationConfig.resolve_cudagraph_mode_and_sizes
    if getattr(orig, "_ie_seq_fit", False):
        return
    signature = inspect.signature(orig)

    def resolve_cudagraph_mode_and_sizes(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return orig(self, *args, **kwargs)
        except ValueError:
            try:
                bound = dict(signature.bind(self, *args, **kwargs).arguments)
            except TypeError:
                bound = {}
            blocks = mamba_block_limit(bound.get("kv_cache_config"), bound.get("max_num_reqs"))
            if blocks is not None:
                record_block_limit(blocks)
            raise

    resolve_cudagraph_mode_and_sizes._ie_seq_fit = True  # pyright: ignore[reportFunctionMemberAccess]
    CompilationConfig.resolve_cudagraph_mode_and_sizes = resolve_cudagraph_mode_and_sizes
