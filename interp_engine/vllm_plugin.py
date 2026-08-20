"""Attach interp-engine capture and steering to a vLLM engine you built yourself.

:class:`VLLMModel` is the batteries-included option. This module is for the case where you
already have a ``vllm.LLM`` or ``AsyncLLM`` -- your own sampling parameters, your own serving
setup -- and want to read activations out of it without handing engine construction over.

    from vllm import LLM, SamplingParams
    from interp_engine import Address, WORKER_EXTENSION_CLS, capture_engine_kwargs, decode_capture_payload

    llm = LLM(model="Qwen/Qwen3-0.6B", worker_extension_cls=WORKER_EXTENSION_CLS,
              **capture_engine_kwargs())

    llm.collective_rpc("install_capture", args=(["resid_post.7"],))
    llm.generate(["The capital of France is"], SamplingParams(max_tokens=1))
    captured = decode_capture_payload(llm.collective_rpc("collect_capture")[0])
    resid = captured[Address("resid_post", 7)]   # [n_prompt_tokens, d_model], CPU

Points cross the wire as canonical address strings (``"resid_post.7"``, and
``"resid_post.7.stream-2"`` on a hyper-connection trunk); ``decode_capture_payload`` parses them
back into :class:`~interp_engine.address.Address` keys. The grammar has one definition, in
:mod:`interp_engine.address`, which both processes import.

Why an extension class rather than passing the functions directly: ``collective_rpc`` accepts
a callable, but on vLLM v1 the engine core runs in another process and the call is
msgpack-encoded, which refuses function objects unless ``VLLM_ALLOW_INSECURE_SERIALIZATION=1``
is set. Registering these as worker methods means they are invoked BY NAME, so nothing is
pickled and that flag is unnecessary.

vLLM refuses an extension whose attribute names collide with the worker's own, so the methods
here are named without the ``worker_`` prefix the module-level functions carry, and there are
no other public attributes on the class.

Two capture styles are exposed, and mixing them on one engine will confuse both:

- ``install_capture`` / ``collect_capture``: global hooks, one request in flight. Simplest,
  and what the example above uses.
- ``register_capture`` / ``collect_request``: per-request hooks keyed by vLLM request id,
  which is what makes concurrent capture possible. Requires passing a ``request_id`` to the
  engine yourself so the ids line up. :class:`VLLMModel` uses this style.
"""

from __future__ import annotations

from typing import Any

from interp_engine.vllm_capture import (
    DEFAULT_HS_STORAGE_PATH,
    extract_hidden_states_engine_kwargs,
    worker_capture_attn,
    worker_clear_static_delta,
    worker_clear_steering,
    worker_collect_attn,
    worker_collect_attn_request,
    worker_collect_capture,
    worker_collect_request,
    worker_collect_static,
    worker_demux_debug,
    worker_drain_request,
    worker_drain_static,
    worker_install_capture,
    worker_install_lens_intervention,
    worker_install_steering,
    worker_lens_capture_readout,
    worker_lens_readout,
    worker_lens_transport,
    worker_lm_head_rows,
    worker_register_attn,
    worker_register_capture,
    worker_register_lens,
    worker_register_static_capture,
    worker_register_static_write,
    worker_register_steering,
    worker_resolvable_points,
    worker_set_lens_jacobians,
    worker_set_static_delta,
    worker_unembed,
    worker_unregister_static_write,
    worker_unregister_steering,
)
from interp_engine.vllm_capture.static import patch_worker_for_static

# Worker process imports this module for worker_extension_cls. Patching load_model here
# (not as an extension method named load_model) is what lets static wraps run before
# CUDA-graph capture. vLLM refuses an extension attribute that collides with Worker.
patch_worker_for_static()

# Pass as ``worker_extension_cls=`` when constructing your engine. vLLM resolves this
# qualified name inside each worker process, so it must stay importable under this path.
WORKER_EXTENSION_CLS = "interp_engine.vllm_plugin.InterpWorkerExtension"


def capture_engine_kwargs(*, enforce_eager: bool = True, enable_prefix_caching: bool = False) -> dict[str, Any]:
    """Engine kwargs that capture needs to be correct. Merge into your own constructor call.

    Both defaults are requirements rather than preferences:

    - ``enforce_eager=True`` because CUDA graphs skip Python forward hooks entirely, so
      capture returns nothing without it.
    - ``enable_prefix_caching=False`` because capture can only see tokens the worker actually
      forwards. On a prefix-cache hit vLLM serves the cached positions from the KV cache and
      schedules only the uncached suffix, so those positions never reach the hooks and the
      capture comes back SHORTER than the prompt, by an amount that depends on unrelated
      recent traffic. Only full blocks are cacheable, so this needs two prompts sharing a
      16-token prefix -- invisible in a quick script, routine on a real server.

    This is for callers who build their own engine and drive the worker functions by hand, which
    is why prefix caching goes off wholesale here. :class:`~interp_engine.vllm_backend.VLLMModel`
    does better and leaves it ON, because it controls every request it issues and can hand the
    intervening ones a ``cache_salt`` that isolates them (see ``VLLMModel._prompt``). Without that
    per-request control there is nothing to salt, so the engine-wide switch is the only safe answer.
    """
    return {"enforce_eager": enforce_eager, "enable_prefix_caching": enable_prefix_caching}


def native_extraction_engine_kwargs(
    n_layers: int, *, storage_path: str = DEFAULT_HS_STORAGE_PATH
) -> tuple[dict[str, Any], list[int]]:
    """Engine kwargs for vLLM's *native* ``resid_post`` extraction, plus the layer ids it writes.

    An alternative to hook capture, for residuals only: it rides a speculative draft + KV
    connector instead of forward hooks, writing safetensors under ``storage_path``. Hook
    capture serves every point including ``resid_post``, so prefer it -- this is here because
    the native path is the independent implementation ``VLLMModel`` is validated against, and
    its extra speculative forwards would otherwise pollute decode-time accumulate capture.

    ``kv_transfer_config`` comes back as a plain dict; wrap it in
    ``vllm.config.KVTransferConfig`` before handing it to the engine.
    """
    return extract_hidden_states_engine_kwargs(n_layers, shared_storage_path=storage_path)


class InterpWorkerExtension:
    """Capture, steering, and lens read-out as vLLM worker methods.

    Mixed into every worker by ``worker_extension_cls``, so ``self`` here IS the vLLM worker
    and the model is reachable at ``self.model_runner.model``. Each method delegates to the
    matching module-level function in :mod:`interp_engine.vllm_capture`, which is where the
    behavior and the payload formats are documented.

    Results come back from ``collective_rpc`` as a list with one entry per rank. Capture
    payloads must be decoded with ``interp_engine.decode_capture_payload`` (tensors are
    msgpack-friendly tuples on the wire). Under tensor parallelism read rank 0 -- but note
    that head-sharded points (``z``, ``value``) are only that rank's slice.
    """

    # --- global capture: one request in flight -------------------------------
    def resolvable_points(self, points: list[str]) -> dict[str, str]:
        """Which of ``points`` this checkpoint carries -> ``{address: "" | why not}``.

        Ask before ``install_capture``, which is all-or-nothing: a point the architecture does not
        have (QK-norm on gpt2, a router on a dense block) raises and costs the whole capture.
        """
        return worker_resolvable_points(self, points)

    def install_capture(self, points: list[str], accumulate: bool = False) -> None:
        return worker_install_capture(self, points, accumulate)

    def collect_capture(self) -> dict[str, tuple]:
        return worker_collect_capture(self)

    # --- global steering -----------------------------------------------------
    def install_steering(self, specs: list[dict]) -> None:
        return worker_install_steering(self, specs)

    def clear_steering(self) -> None:
        return worker_clear_steering(self)

    def install_lens_intervention(
        self, specs: list[dict], steer_generated: bool, skip_positions: list[int], prompt_len: int
    ) -> None:
        return worker_install_lens_intervention(self, specs, steer_generated, skip_positions, prompt_len)

    # --- attention (off-kernel recompute inputs) -----------------------------
    def capture_attn(self, layers: list[int]) -> None:
        return worker_capture_attn(self, layers)

    def collect_attn(self) -> dict[str, tuple]:
        return worker_collect_attn(self)

    # --- lens read-out -------------------------------------------------------
    def unembed(self, payload: tuple) -> tuple:
        return worker_unembed(self, payload)

    def lens_readout(
        self,
        residual_payload: tuple,
        top_n: int,
        softcap: float | None,
        word_mask_payload: tuple | None,
        rows_per_group: int,
    ) -> dict[str, tuple]:
        return worker_lens_readout(self, residual_payload, top_n, softcap, word_mask_payload, rows_per_group)

    def set_lens_jacobians(self, payloads: dict[str, tuple] | None) -> dict[str, int]:
        return worker_set_lens_jacobians(self, payloads)

    def lens_capture_readout(
        self, req_id: str, spec: dict, word_mask_payload: tuple | None, final: bool
    ) -> dict[str, Any]:
        return worker_lens_capture_readout(self, req_id, spec, word_mask_payload, final)

    def lens_transport(self, payload: tuple, layers: list[int]) -> dict[str, Any]:
        return worker_lens_transport(self, payload, layers)

    def lm_head_rows(self, token_ids: list[int]) -> dict:
        return worker_lm_head_rows(self, token_ids)

    # --- per-request demux: concurrent capture + steering --------------------
    def register_capture(self, req_id: str, points: list[str]) -> None:
        return worker_register_capture(self, req_id, points)

    def collect_request(self, req_id: str) -> dict[str, tuple]:
        return worker_collect_request(self, req_id)

    def drain_request(self, req_id: str) -> dict[str, tuple]:
        return worker_drain_request(self, req_id)

    def register_steering(
        self, req_id: str, specs: list[dict], skip_positions: list[int] | None = None, prompt_len: int = 0
    ) -> None:
        return worker_register_steering(self, req_id, specs, skip_positions, prompt_len)

    def register_lens(
        self, req_id: str, specs: list[dict], steer_generated: bool, skip_positions: list[int], prompt_len: int
    ) -> None:
        return worker_register_lens(self, req_id, specs, steer_generated, skip_positions, prompt_len)

    def unregister_steering(self, req_id: str) -> None:
        return worker_unregister_steering(self, req_id)

    def register_attn(self, req_id: str, layers: list[int]) -> None:
        return worker_register_attn(self, req_id, layers)

    def collect_attn_request(self, req_id: str) -> dict[str, tuple]:
        return worker_collect_attn_request(self, req_id)

    def demux_debug(self) -> dict:
        """Per-request hook bookkeeping, for diagnosing a capture that came back short."""
        return worker_demux_debug(self)

    # --- static taps: static copy_ / add_ installed in Worker.load_model ----
    def set_static_delta(self, specs: list[dict], lens_scope: dict | None = None) -> None:
        return worker_set_static_delta(self, specs, lens_scope)

    def clear_static_delta(self) -> None:
        return worker_clear_static_delta(self)

    def register_static_write(
        self,
        req_id: str,
        specs: list[dict],
        skip_positions: list[int] | None = None,
        prompt_len: int = 0,
        lens_scope: dict | None = None,
    ) -> None:
        return worker_register_static_write(self, req_id, specs, skip_positions, prompt_len, lens_scope)

    def unregister_static_write(self, req_id: str) -> None:
        return worker_unregister_static_write(self, req_id)

    def register_static_capture(self, req_id: str, points: list[str]) -> None:
        return worker_register_static_capture(self, req_id, points)

    def collect_static(self, req_id: str) -> dict[str, tuple]:
        return worker_collect_static(self, req_id)

    def drain_static(self, req_id: str) -> dict[str, tuple]:
        return worker_drain_static(self, req_id)
