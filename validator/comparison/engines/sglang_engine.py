"""Capture the residual stream from SGLang (resid_post at each captured layer).

SGLang runs the model in a scheduler subprocess with no public hook/RPC surface, so — unlike vLLM
(which exposes ``worker_extension_cls`` + ``collective_rpc``) — we can't attach hooks from the main
process. Instead we inject a gated ``sitecustomize.py`` (via ``PYTHONPATH``) that the scheduler
child runs at startup; it monkeypatches ``ModelRunner.load_model`` to register PyTorch forward
hooks inside the worker (see ``sglang_inject/sglang_hooks.py`` — the reusable substrate the
Tier-2 serving-time steering will share). The hooks dump ``resid_post`` per layer to ``.npy`` files
which we read back here.

Fused-engine caveats as vLLM: SGLang serves half precision (float32 raises ``KeyError: torch.float32``
in its scheduler), so we run bf16 → the loose "fused" tolerance tier (compare by cosine/relative
error). Fused kernels can't expose attention probabilities, so this is residual-only.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import tempfile

import numpy as np

from comparison.spec import SaeSpec, dump_key

_INJECT_DIR = os.path.join(os.path.dirname(__file__), "sglang_inject")


def _d_model(hf_id: str) -> int:
    """The checkpoint's hidden size, for the capture-time width check in the scheduler child (0 if it
    can't be read). Config-only, so it costs nothing and works before the engine is up."""
    try:
        from interp_engine import facts
        from transformers import AutoConfig

        return int(facts.resolve_facts(AutoConfig.from_pretrained(hf_id, trust_remote_code=True)).d_model)
    except Exception:  # noqa: BLE001 - no width check rather than no capture
        return 0


def capture(
    hf_id: str,
    input_ids: list[int],
    layers: list[int],
    points: list[str],
    saes: tuple[SaeSpec, ...] = (),  # noqa: ARG001 - SAE spot-check stays on eager engines
    device: str = "cuda",  # noqa: ARG001 - SGLang auto-detects device
    dtype: str = "bfloat16",
) -> tuple[dict[str, np.ndarray], list[dict]]:
    import sglang as sgl

    capturable = [p for p in points if p in ("resid_post", "resid_mid", "attn_out", "mlp_out")]
    if not capturable:
        return {}, []

    # SGLang serves half precision only (float32 raises KeyError: torch.float32 in its scheduler),
    # so a float32-native checkpoint (e.g. gpt2) is captured in bf16 — a documented cross-dtype cell
    # in the loose fused tier. bf16-native models (gemma/qwen) match the other engines' dtype.
    sg_dtype = "bfloat16" if dtype == "float32" else dtype

    # SGLang hard-aborts a *multimodal* load on torch==2.9.1 + cuDNN<9.15 (a known nn.Conv3d
    # perf/memory bug: github.com/pytorch/pytorch/issues/168167 — see
    # server_args.check_torch_2_9_1_cudnn_compatibility). That Conv3d lives in the VISION tower;
    # our capture is text-only (resid_post/mlp_out/attn_out on the decoder, input_ids only), so it
    # never runs and the guard is a false alarm that blocks the multimodal-wrapper checkpoints
    # (qwen3.5/3.6). Bypass it. (The "correct" alternative is to upgrade the sglang venv to
    # nvidia-cudnn-cu13>=9.15, but that's unnecessary for text-only capture.) `setdefault` so an
    # explicit override from the environment still wins.
    os.environ.setdefault("SGLANG_DISABLE_CUDNN_CHECK", "1")

    cap_dir = tempfile.mkdtemp(prefix="ie_sglang_cap_")
    # These env vars are inherited by the spawned scheduler child; PYTHONPATH makes it import our
    # gated sitecustomize at startup, which installs the capture hooks inside the worker.
    os.environ["PYTHONPATH"] = _INJECT_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["IE_SGLANG_CAPTURE_DIR"] = cap_dir
    os.environ["IE_SGLANG_CAPTURE_LAYERS"] = ",".join(str(x) for x in layers)
    os.environ["IE_SGLANG_CAPTURE_POINTS"] = ",".join(capturable)
    os.environ["IE_SGLANG_CAPTURE_D_MODEL"] = str(_d_model(hf_id))
    print(
        f"[sglang/{hf_id}] dtype={sg_dtype}, capturing {capturable} @ layers {layers} -> {cap_dir}",
        file=sys.stderr,
        flush=True,
    )

    # SGLang sizes its KV pool from whatever memory is left after the weights, and with no arguments
    # that leftover can be too small to admit a single request: gemma-4-31b (62 GB of weights) got a
    # 710-token sliding-window pool against a prefill admission floor of sliding_window + page_size =
    # 1024, and the scheduler aborted the load. `max_total_tokens` cannot fix it — SGLang only ever
    # applies that as a cap — so the levers are the static fraction (a bigger budget) and
    # `swa_full_tokens_ratio` (a bigger share of it for the windowed pool). Also pin `context_length`
    # to the prompt, as the vLLM adapter does with `max_model_len`, so nothing is reserved for a
    # 128k-token context we never use.
    engine = sgl.Engine(
        model_path=hf_id,
        dtype=sg_dtype,  # SGLang serves half precision only
        disable_cuda_graph=True,  # so forward hooks fire (CUDA graphs would bypass them)
        context_length=max(len(input_ids) + 8, 32),
        mem_fraction_static=float(os.environ.get("IE_SGLANG_MEM_FRACTION", "0.93")),
        trust_remote_code=True,
    )
    try:
        engine.generate(
            input_ids=list(input_ids),
            sampling_params={"max_new_tokens": 1, "temperature": 0.0},
        )
        arrays: dict[str, np.ndarray] = {}
        for point in capturable:
            for layer in layers:
                path = os.path.join(cap_dir, f"{point}.{layer}.npy")
                if os.path.exists(path):
                    arrays[dump_key(point, layer)] = np.load(path)
        return arrays, []
    finally:
        with contextlib.suppress(Exception):
            engine.shutdown()
        shutil.rmtree(cap_dir, ignore_errors=True)
        for var in (
            "IE_SGLANG_CAPTURE_DIR",
            "IE_SGLANG_CAPTURE_LAYERS",
            "IE_SGLANG_CAPTURE_POINTS",
            "IE_SGLANG_CAPTURE_D_MODEL",
        ):
            os.environ.pop(var, None)
