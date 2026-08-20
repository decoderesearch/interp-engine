"""interp-engine-owned activation capture for the vLLM backend.

Installs PyTorch forward hooks on the *real* model modules inside a vLLM worker
(reached via ``collective_rpc``) and returns their I/O for the in-flight forward.

Two paths coexist, plus a third for CUDA-graph replay:

- **Single-request** (``worker_install_capture`` / ``worker_install_steering`` /
  ``worker_capture_attn`` etc., in :mod:`~interp_engine.vllm_capture.capture`,
  :mod:`~interp_engine.vllm_capture.steering` and :mod:`~interp_engine.vllm_capture.attn`):
  global hooks, one forward at a time. Simple, and the shape the tier-1
  ``worker_extension_cls`` plugin exposes to users driving their own ``vllm.LLM``; nothing
  inside the engine drives it.
- **Per-request demux** (``worker_register_*`` / ``worker_collect_request``, in
  :mod:`~interp_engine.vllm_capture.requests`): persistent refcounted hooks that attribute
  each row of the batched forward to its request (via the ``execute_model`` patch), enabling
  N-way concurrent capture + steering. This is what :class:`VLLMModel` uses when
  ``enforce_eager=True``.
- **Static taps** (``worker_register_static_capture`` / ``worker_set_static_delta``, in
  :mod:`~interp_engine.vllm_capture.static`): static ``copy_`` / ``add_`` wraps installed in
  ``Worker.load_model`` **before** CUDA-graph capture. A non-empty static set forces vLLM's
  breakable path (graph replay, no torch.compile) so the wrap stays Python ``add_eager``.
  Chosen at engine build via ``static_points`` / ``static_writes``.

This is deliberately independent of the vendored steerllm capture path (it
advances the eventual removal of steerllm/chatspace): it only needs a way to run
a callable on each worker.

vLLM specifics (verified on vLLM 0.11.2, Qwen3; the layout is shared by the
Qwen2/Qwen3/Llama-family vLLM model impls):

- The model is ``worker.model_runner.model`` (e.g. ``Qwen3ForCausalLM``); decoder
  layers live at ``model.model.layers``. Multimodal wrappers
  (``Qwen3_5ForConditionalGeneration`` / Qwen3.6) nest the text LM at
  ``language_model.model.layers``.
- Decoder layers use a *fused* RMSNorm+residual and
  ``forward(positions, hidden_states, residual) -> (hidden, residual)``. The true
  residual stream is the SUM of the two returned tensors, so
  ``resid_post[L] = out[0] + out[1]`` and, at the input,
  ``resid_pre[L] = hidden_states (+ residual if not None)``.
- The *pre-MLP* norm is fused the same way -- ``norm(hidden, residual) -> (normed, hidden +
  residual)`` -- so ``resid_mid[L]`` is likewise the sum of its two ARGUMENTS, which is one add
  earlier than the tensor it hands on. Families that add before the call (vLLM's ``gpt2``, whose
  ``ln_2`` is a plain ``nn.LayerNorm``) pass a single argument that is already the residual.
- ``self_attn.forward(positions, hidden_states) -> attn_out`` (already post
  ``o_proj``); ``z`` (pre-``W_O``) is the input to ``self_attn.o_proj``.
- ``mlp.forward(x)`` so ``mlp_in`` is its input and ``mlp_out`` its output.
- ``value`` is the v-slice of the fused ``qkv_proj`` output.
- Activation tensors are FLATTENED ``[num_tokens, hidden]`` (no batch dim); for a
  single prefill they are the prompt tokens in order.

Requirements: hooked capture needs ``enforce_eager=True``, because Python hooks are
skipped under CUDA graphs. Static taps (``static_points=``) is the other way: bake
``copy_`` / ``add_`` into the recorded graph. Capture also needs every token to be forwarded, which a prefix-cache
hit prevents -- but that is handled per request rather than by an engine flag: ``VLLMModel._prompt``
gives each capture a unique ``cache_salt`` so it cannot hit the cache. A caller building their own
engine and driving these functions by hand gets the same guarantee from
:func:`~interp_engine.vllm_plugin.capture_engine_kwargs`, which turns prefix caching off outright.

These functions take the worker as their first argument, so they can be reached two ways.
Passing one to ``collective_rpc`` directly needs ``VLLM_ALLOW_INSECURE_SERIALIZATION=1``,
because vLLM v1 msgpack-encodes the call to an out-of-process engine core and refuses
function objects. The alternative needs no flag: :mod:`interp_engine.vllm_plugin` exposes
them as methods on a ``worker_extension_cls``, invoked by name. That is what
:class:`VLLMModel` and ``load_model`` use, so callers going through either need do nothing.

Supported points here are exactly ``points.vllm_hookable()`` -- the residual stream
(``resid_pre``/``resid_mid``/``resid_post``), both sublayer boundaries and their post-norm
contributions (``attn_in``, ``attn_out``, ``attn_out_post``, ``mlp_in``, ``mlp_out``,
``mlp_out_post``), the attention internals (``z``, ``value``, and the four QK-norm points), the
neuron basis (``mlp_act``) and ``router_logits``. That set is pinned to the table by
``tests/test_points_registry.py`` on both the single-request and the per-request path, so this
sentence cannot be the thing that goes stale.

Attention probabilities / DFA and the pre-softmax ``attn_scores`` are handled by the off-kernel
recompute in :mod:`~interp_engine.vllm_capture.attn` (``worker_capture_attn`` +
``recompute_attn_scores`` / ``attn_probs_from_scores``), which captures post-rope q/k/v at
``self_attn.attn`` and rebuilds the softmax rather than hooking a matrix the paged kernel never
forms. Static those taps with ``Address("attn", layer)``; ``capture_attention`` harvests the
static buffers instead of installing hooks.

Five of the seven hyper-connection points (``MHC_KERNEL_POINTS``) are served a third way, in
:mod:`~interp_engine.vllm_capture.mhc`: they are locals of a DeepSeek-V4 decoder layer's forward, so
the tap is on the mHC kernel functions the layer calls rather than on any module. Static installs
the same wraps before graph capture (``add_eager``) so ``"auto"`` on a hyper-connection trunk is
``resid_streams`` at every layer. They arrive through the same stores and the same keys as every
hooked point, so a caller sees no difference -- but the mechanism is NVIDIA-tree-specific where a
module hook would not be, which is why it is a set of its own rather than folded into the sides
above.

Layout
------

This was one 2300-line module until the Jacobian-lens read-out moved into the worker and made
it clear that jlens had ended up in three non-adjacent places in it. The modules are layered so
that imports run one way, leaves first::

    _payload  _tree            wire format; which module (or kernel) a point reads from
       |         |
    _hooks    _demux    mhc    hook factories; per-request state and row layout; the mHC kernel taps
       |         |       |
    capture  steering  lens  attn  native
                  \\     |     /
                   requests    per-request hooks + the worker_register_* surface

``_demux`` holds state only, with no dependency on steering or the lens, which is what lets
``lens`` reach the capture store while ``requests`` installs the lens's own write-hook -- the
two would otherwise import each other.

Everything below is re-exported here, so ``from interp_engine.vllm_capture import X`` keeps
working and this list stays readable as an index of what the worker can be asked to do.
"""

from __future__ import annotations

# Internals with callers outside this package -- tests reaching for a helper directly, and the
# drift guard in test_facts.py that asserts the name vocabularies ARE ``facts``'s objects. The
# redundant `as` is what marks each one an intentional re-export rather than an unused import.
#
# Note for anyone adding a test: `monkeypatch.setattr` on one of these patches the alias, not
# the name the defining module looks up, so it has no effect. Patch the submodule instead --
# `interp_engine.vllm_capture.lens`, say -- which is what test_logit_transform.py does.
from interp_engine.vllm_capture._demux import _Demux as _Demux
from interp_engine.vllm_capture._demux import _get_demux as _get_demux
from interp_engine.vllm_capture._hooks import _make_pre_hook as _make_pre_hook
from interp_engine.vllm_capture._payload import (
    ATTN_PAYLOAD_ROLES,
    attn_payload_key,
    decode_capture_payload,
    decode_tensor_payload,
    encode_tensor_payload,
    hook_site,
    select_stream,
)
from interp_engine.vllm_capture._tree import _ATTN_ATTRS as _ATTN_ATTRS
from interp_engine.vllm_capture._tree import _ATTN_OUT_PROJ_ATTRS as _ATTN_OUT_PROJ_ATTRS
from interp_engine.vllm_capture._tree import _ATTN_QKV_PROJ_ATTRS as _ATTN_QKV_PROJ_ATTRS
from interp_engine.vllm_capture._tree import _FINAL_NORM_ATTRS as _FINAL_NORM_ATTRS
from interp_engine.vllm_capture._tree import _GLOBAL_POINTS as _GLOBAL_POINTS
from interp_engine.vllm_capture._tree import _INPUT_POINTS as _INPUT_POINTS
from interp_engine.vllm_capture._tree import _KWARG_INPUT_POINTS as _KWARG_INPUT_POINTS
from interp_engine.vllm_capture._tree import _LAYER_LIST_ATTRS as _LAYER_LIST_ATTRS
from interp_engine.vllm_capture._tree import _OUTPUT_POINTS as _OUTPUT_POINTS
from interp_engine.vllm_capture._tree import _TRUNK_CONTAINER_ATTRS as _TRUNK_CONTAINER_ATTRS
from interp_engine.vllm_capture._tree import HOOK_CAPTURE_POINTS, MHC_KERNEL_POINTS, STEERABLE_POINTS
from interp_engine.vllm_capture._tree import _get_layers as _get_layers
from interp_engine.vllm_capture._tree import _resolve_module as _resolve_module
from interp_engine.vllm_capture.attn import (
    attn_probs_from_scores,
    causal_window_mask,
    recompute_attn_probs,
    recompute_attn_scores,
    worker_capture_attn,
    worker_collect_attn,
)
from interp_engine.vllm_capture.capture import (
    worker_addresses,
    worker_collect_capture,
    worker_install_capture,
    worker_resolvable_points,
)
from interp_engine.vllm_capture.lens import _local_lm_head_rows as _local_lm_head_rows
from interp_engine.vllm_capture.lens import _worker_final_norm as _worker_final_norm
from interp_engine.vllm_capture.lens import _worker_logits_processor as _worker_logits_processor
from interp_engine.vllm_capture.lens import _worker_unembed_weight as _worker_unembed_weight
from interp_engine.vllm_capture.lens import (
    merge_lm_head_row_payloads,
    worker_install_lens_intervention,
    worker_lens_capture_readout,
    worker_lens_readout,
    worker_lens_transport,
    worker_lm_head_rows,
    worker_set_lens_jacobians,
    worker_unembed,
)
from interp_engine.vllm_capture.mhc import mhc_taps as mhc_taps
from interp_engine.vllm_capture.mhc import stream_collapse
from interp_engine.vllm_capture.native import (
    DEFAULT_HS_STORAGE_PATH,
    extract_hidden_states_engine_kwargs,
    read_resid_post_from_output,
)
from interp_engine.vllm_capture.requests import _process_point as _process_point
from interp_engine.vllm_capture.requests import (
    _refuse_unreachable_resid_mid_steer as _refuse_unreachable_resid_mid_steer,
)
from interp_engine.vllm_capture.requests import (
    worker_collect_attn_request,
    worker_collect_request,
    worker_demux_debug,
    worker_drain_request,
    worker_register_attn,
    worker_register_capture,
    worker_register_lens,
    worker_register_steering,
    worker_unregister_steering,
)
from interp_engine.vllm_capture.static import (
    ATTN_STATIC_POINT,
    worker_clear_static_delta,
    worker_collect_static,
    worker_drain_static,
    worker_register_static_capture,
    worker_register_static_write,
    worker_set_static_delta,
    worker_unregister_static_write,
)
from interp_engine.vllm_capture.steering import worker_clear_steering, worker_install_steering

__all__ = [
    "ATTN_STATIC_POINT",
    "ATTN_PAYLOAD_ROLES",
    "DEFAULT_HS_STORAGE_PATH",
    "HOOK_CAPTURE_POINTS",
    "MHC_KERNEL_POINTS",
    "STEERABLE_POINTS",
    "attn_payload_key",
    "attn_probs_from_scores",
    "causal_window_mask",
    "decode_capture_payload",
    "decode_tensor_payload",
    "encode_tensor_payload",
    "extract_hidden_states_engine_kwargs",
    "hook_site",
    "merge_lm_head_row_payloads",
    "read_resid_post_from_output",
    "recompute_attn_probs",
    "recompute_attn_scores",
    "select_stream",
    "stream_collapse",
    "worker_addresses",
    "worker_capture_attn",
    "worker_clear_static_delta",
    "worker_clear_steering",
    "worker_collect_static",
    "worker_collect_attn",
    "worker_collect_attn_request",
    "worker_collect_capture",
    "worker_collect_request",
    "worker_demux_debug",
    "worker_drain_static",
    "worker_drain_request",
    "worker_install_capture",
    "worker_install_lens_intervention",
    "worker_install_steering",
    "worker_lens_capture_readout",
    "worker_lens_readout",
    "worker_lens_transport",
    "worker_lm_head_rows",
    "worker_register_attn",
    "worker_register_static_capture",
    "worker_register_static_write",
    "worker_register_capture",
    "worker_register_lens",
    "worker_register_steering",
    "worker_resolvable_points",
    "worker_set_static_delta",
    "worker_set_lens_jacobians",
    "worker_unembed",
    "worker_unregister_static_write",
    "worker_unregister_steering",
]
