"""How much memory a checkpoint needs, answered from its config alone — before anything is loaded.

Two decisions in the sweep have to be made *before* the weights are on the GPU, and getting either
wrong costs whole rows rather than one cell:

  - :func:`run_engine._native_dtype` — the sweep loads each checkpoint's native dtype on purpose, and
    a float32-native 27B (Gemma-2) is ~109 GB of weights, so on an 80 GB card the reference engine
    OOM'd and took every other engine's cell for that model with it.
  - :mod:`comparison.engines.tlens_engine` — legacy ``HookedTransformer.from_pretrained_no_processing``
    loads the HF model and *then* converts it, so its host-RAM peak is close to twice the weights.
    Past the container's limit the OOM killer sends SIGKILL and the capture leaves no record at all.

The parameter count is estimated from config dims rather than measured: reaching for the weights to
decide whether the weights will fit defeats the purpose. It is within a few percent on dense and MoE
transformers alike, which is far inside the margin of a fits/doesn't-fit call.
"""

from __future__ import annotations

import subprocess
from typing import Any

_DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2, "float8_e4m3fn": 1}


def estimated_param_count(config: Any) -> int:
    """Parameters implied by an HF config's dims, or 0 if the config doesn't carry enough to say.

    Counts attention (q/k/v/o at the per-layer head dims, so GQA is not overcounted), the MLP as a
    gated triple, routed + shared experts on the layers that are sparse, and embeddings once when
    tied and twice when not.
    """
    from interp_engine import facts

    f = facts.resolve_facts(config)
    if not f.n_layers or not f.d_model:
        return 0
    cfg = facts.text_config(config)
    inter = int(getattr(cfg, "intermediate_size", 0) or 0) or 4 * f.d_model
    moe_inter = int(getattr(cfg, "moe_intermediate_size", 0) or 0) or inter

    q_dim, kv_dim = f.n_heads * f.head_dim, f.n_kv_heads * f.head_dim
    attn = f.d_model * (q_dim + 2 * kv_dim) + q_dim * f.d_model
    # Gated triple (gate/up/down). Whether a family gates its MLP is not derivable from the config
    # alone, so an ungated MLP (GPT-2, GPT-NeoX) is overcounted by a third of its layer — and
    # overcounting is the safe direction, since it errs toward the dtype that fits.
    dense_mlp = 3 * f.d_model * inter
    sparse_mlp = (f.n_experts + f.n_shared_experts) * 3 * f.d_model * moe_inter if f.n_experts else dense_mlp

    n_sparse = len(f.moe_layers)
    # Gemma-4's sparse layers keep their dense MLP and add the experts beside it, so the dense branch
    # is paid on every layer there rather than only on the ones with no experts.
    n_dense = f.n_layers if f.dense_mlp_beside_experts else f.n_layers - n_sparse
    embeddings = f.vocab_size * f.d_model * (1 if f.tied_embeddings else 2)
    return f.n_layers * attn + n_sparse * sparse_mlp + n_dense * dense_mlp + embeddings


def weight_bytes(config: Any, dtype: str) -> int:
    """Bytes the weights occupy at ``dtype`` (0 when the parameter count is unknown)."""
    return _DTYPE_BYTES.get(dtype, 4) * estimated_param_count(config)


def gpu_memory_bytes() -> int:
    """Total VRAM of GPU 0 via ``nvidia-smi``, or 0 if it can't be read.

    Asked out-of-process deliberately: this runs before any engine is imported, and initializing a
    CUDA context in the parent just to ask torch would change how vLLM and SGLang start up.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return int(float(out.stdout.strip().splitlines()[0])) * 1024**2
    except Exception:  # noqa: BLE001 - no GPU, no nvidia-smi, unparseable output: all "unknown"
        return 0


def host_memory_bytes() -> int:
    """Memory this process may actually use: the cgroup limit when one is set, else ``MemTotal``.

    Inside a container ``MemTotal`` is the *host's* RAM and is not the number that gets us killed —
    the sweep runs under a limit well below the machine's memory. An unlimited cgroup reads as
    ``max`` (v2) or a huge sentinel (v1), and both fall through to ``MemTotal`` via the ``min``.
    """
    limits: list[int] = []
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path) as f:
                value = f.read().strip()
        except OSError:
            continue
        if value.isdigit():
            limits.append(int(value))
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    limits.append(int(line.split()[1]) * 1024)
                    break
    except OSError:
        pass
    return min(limits) if limits else 0


def memory_budget_bytes(device: str, fraction: float = 0.85) -> int:
    """Weight budget on ``device``: a fraction of what exists, leaving room for activations, kernel
    workspace and (on GPU) the CUDA context.

    Returns 0 when the size cannot be determined, which every caller must read as "don't
    second-guess the checkpoint" rather than as "no room".
    """
    total = gpu_memory_bytes() if device.startswith("cuda") else host_memory_bytes()
    return int(total * fraction) if total else 0


def gib(n: int) -> str:
    return f"{n / 1024**3:.0f} GiB"
