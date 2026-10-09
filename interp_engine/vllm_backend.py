"""Engine-owned vLLM backend (replaces the vendored steerllm/chatspace wrapper).

interp-engine constructs and owns the vLLM engine directly, with native
residual extraction turned on (``extract_hidden_states`` + the hidden-states KV
connector). This is the foundation of removing steerllm: the engine -- not a
vendored wrapper -- owns vLLM construction, capture, (later) generation and
steering.

This first layer is the synchronous ``LLM`` capture path (residuals via native
extraction; intra-block taps + steering via the collective_rpc hooks in
``vllm_capture``). Async generation/streaming + steering write-hooks + server
wiring come next; see plan ``engine-owns-vllm``.

Requires vLLM (Linux/CUDA); imports are lazy so importing this module is safe on
macOS/CPU.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from typing import Any, TypeVar

import torch

from interp_engine import facts
from interp_engine._loop import refuse_foreign_loop
from interp_engine.address import Address, format_address, to_address
from interp_engine.api import DirectionSet, EngineDescription, LensSpec, LensStep
from interp_engine.autograd_support import GradSupport, vllm_grad_support
from interp_engine.describe import describe_model
from interp_engine.directions import check_directions, project_by_capture, to_wire
from interp_engine.dispatch import refuse
from interp_engine.notebook_stdout import ensure_stdout_descriptor
from interp_engine.points import Scope, d_model_wide, hyper_connection_names, point_spec, refusal_reasons
from interp_engine.points import steer_refusal_reason as points_steer_refusal
from interp_engine.protocol import (
    REFUSAL_ERRORS,
    EmbedsSample,
    Point,
    checked_prompt_embeds,
    checked_rows,
    layer_out_of_range,
)
from interp_engine.residual_basis import ResidualBasis, vllm_residual_basis
from interp_engine.sampling import RecommendedSampling, SamplingSettings, read_recommended_sampling, resolve_sampling
from interp_engine.steer_specs import SteerMethod
from interp_engine.vllm_capture import (
    _GLOBAL_POINTS,
    DEFAULT_HS_STORAGE_PATH,
    HOOK_CAPTURE_POINTS,
    STEERABLE_POINTS,
    attn_payload_key,
    attn_probs_from_scores,
    decode_capture_payload,
    decode_tensor_payload,
    encode_tensor_payload,
    extract_hidden_states_engine_kwargs,
    read_resid_post_from_output,
    recompute_attn_scores,
)
from interp_engine.vllm_capture.static import (
    DECODE_ONLY_GRAPHS,
    STATIC_ENV,
    STATIC_SKIP_ABSENT_ENV,
    STATIC_WRITE_OPS,
    apply_breakable_env,
    decode_only_graphs_reason,
    encode_static_env,
    estimate_weight_bytes,
    fit_max_num_batched_tokens,
    kv_cache_width,
    quantized_on_load_bytes,
    resid_stream_aliases,
    resolve_static_points,
    sm100_cudagraph_refusal_reason,
    static_read_width,
    static_unsupported_reason,
)
from interp_engine.vllm_plugin import WORKER_EXTENSION_CLS

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def _device_capability() -> tuple[int, int] | None:
    """This GPU's compute capability, or None when there is no CUDA device to ask.

    None rather than a default, because callers gate refusals on it: a guess would either refuse a
    machine nobody measured or wave through the one that is known wrong.
    """
    try:
        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(0)
        return int(major), int(minor)
    except Exception:  # noqa: BLE001 - no driver, or a torch without the call
        return None


# Residual-width sites the warmup sentinel can prove. ``mlp_act`` / ``z`` need a different
# vector length than ``d_model``, which this process does not know until a worker wrap exists.
_STATIC_SENTINEL_WRITE_POINTS = frozenset(
    {
        "resid_pre",
        "resid_post",
        "resid_mid",
        "resid_streams",
        "mlp_out",
        "attn_out",
        "mlp_out_post",
        "attn_out_post",
        "attn_stream_collapse",
        "mlp_stream_collapse",
    }
)
_STATIC_SENTINEL = 50.0


def _static_set_literal(declared: Sequence[Address], extra: Sequence[Address] = ()) -> str:
    """Python for a ``static_points=`` list that holds ``declared`` plus ``extra``.

    Spelled as a comprehension when the declared set is one point name at every layer from 0,
    which is the shape ``"auto"`` produces and so the shape almost every engine has. A literal
    list of 40 addresses is technically the same answer and useless to paste.
    """
    tail = "".join(f', Address("{a.name}", {a.layer})' for a in extra)
    every_layer = _one_name_at_every_layer(declared)
    if every_layer:
        name, n_layers = every_layer
        return f'[*(Address("{name}", i) for i in range({n_layers})){tail}]'
    listed = ", ".join(f'Address("{a.name}", {a.layer})' for a in declared)
    return f"[{listed}{tail}]"


def _one_name_at_every_layer(declared: Sequence[Address]) -> tuple[str, int] | None:
    """``(point, n_layers)`` when ``declared`` is one point name at layers ``0..n-1``, else None."""
    names = {a.name for a in declared}
    if len(names) != 1:
        return None
    layers = sorted(a.layer for a in declared if a.layer is not None)
    if not layers or layers != list(range(len(layers))):
        return None
    return next(iter(names)), len(layers)


def _describe_static_set(declared: Sequence[Address]) -> str:
    """The declared set in prose, collapsed where naming all of it would just be noise."""
    if not declared:
        return "nothing"
    every_layer = _one_name_at_every_layer(declared)
    if every_layer:
        name, n_layers = every_layer
        return f"{name} at every layer (0-{n_layers - 1})"
    return ", ".join(str(a) for a in declared)


def _static_miss_message(
    what: str,
    missing: Sequence[Address],
    declared: Sequence[Address],
    hf_model_id: str | None,
    *,
    kwarg: str = "static_points",
) -> str:
    """Why a point is not servable here, and the one way to fix it.

    Two different refusals, because the fix differs. A point this backend could have taken but
    was not asked for is a reload away, so the message carries the call to make. A point no
    static tap can serve at all would fail that reload too, so it is sent to the hooked backend
    instead -- with the engine's own reason rather than a paraphrase, since "not declared" and
    "cannot be declared" are the difference between editing one line and changing backend.
    """
    undeclarable = [(a, static_unsupported_reason(a.name)) for a in missing]
    blocked = [(a, reason) for a, reason in undeclarable if reason]
    if blocked:
        lines = [
            f"{what} asked for {', '.join(str(a) for a, _ in blocked)}, which no static tap can "
            f"serve on backend='vllm-static':"
        ]
        lines += [f"  {a}: {reason}" for a, reason in blocked]
        lines.append(
            "Reloading with a wider static_points= would refuse the same way. Use "
            "backend='vllm' (hooked, serves every point) for these."
        )
        return "\n".join(lines)
    return (
        f"{what} asked for {', '.join(str(a) for a in missing)}, which this engine did not "
        f"declare. It was built with backend='vllm-static', whose tap set is fixed when the "
        f"CUDA graphs are recorded, so no new tap can be installed on the running model.\n"
        f"Declared: {_describe_static_set(declared)}\n"
        f"Reload with them included:\n"
        f'    model = load_model("{hf_model_id or "<hf_model_id>"}", backend="vllm-static",\n'
        f"                       {kwarg}={_static_set_literal(declared, missing)})"
    )


def _sentinel_steering(site: Address, width: int) -> Any:
    from interp_engine.steer_specs import AddSpec, LayerSteeringSpec, SteeringSpec

    if site.layer is None:
        raise ValueError(f"the static self-test steers one layer, so its write site needs one; got {site}")
    vector = [_STATIC_SENTINEL if i % 2 == 0 else -_STATIC_SENTINEL for i in range(max(int(width), 1))]
    return SteeringSpec(
        layers={int(site.layer): LayerSteeringSpec(operations=[AddSpec(vector=vector, scale=1.0)])},
        point=str(site.name),
    )


def _assert_live_harvest(tensor: torch.Tensor, site: Address) -> None:
    """A traced-away ``copy_`` leaves a zero buffer of the right shape. That is a miss."""
    if not torch.isfinite(tensor).all():
        raise RuntimeError(
            f"static self-test: harvest at {site} is not finite. The copy_ tap ran but the "
            "values are NaN/Inf. Refuse to serve."
        )
    if not torch.any(tensor != 0):
        raise RuntimeError(
            f"static self-test: harvest at {site} is all zeros. The static copy_ did not run "
            "on CUDA-graph replay (a traced-away wrap looks like a zero buffer of the right "
            "shape). Refuse to serve. Check VLLM_USE_BREAKABLE_CUDAGRAPH=1, or reload with "
            'backend="vllm", whose hooks do not depend on a recorded graph.'
        )


def _static_steer_lock(model: Any) -> asyncio.Lock:
    lock = getattr(model, "_static_steer_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        model._static_steer_lock = lock
    return lock


class _StaticDeltaLease:
    """One steered request's hold on the process-global static write buffer."""

    def __init__(
        self,
        model: Any,
        specs: list[dict],
        position_mask: Any = None,
        lens_scope: dict | None = None,
    ) -> None:
        self._model = model
        self._specs = specs
        self._mask = position_mask
        self._lens_scope = lens_scope
        self._held = False

    async def start(self) -> None:
        await _static_steer_lock(self._model).acquire()
        self._held = True
        try:
            args: tuple = (self._specs,) if self._lens_scope is None else (self._specs, self._lens_scope)
            await self._model.engine.collective_rpc("set_static_delta", args=args)
        except BaseException:
            self._release_lock()
            raise

    async def finish(self) -> None:
        try:
            if self._held:
                await self._model.engine.collective_rpc("clear_static_delta")
        finally:
            self._release_lock()

    def _release_lock(self) -> None:
        if not self._held:
            return
        self._held = False
        _static_steer_lock(self._model).release()


def vllm_installed() -> bool:
    """Whether vLLM is importable, without paying for the import.

    ``find_spec`` rather than a real import: on a no-CUDA box we want to answer this and
    move on. The auto ladder only imports vLLM for real once CUDA is present (inside
    ``select._vllm_supports_arch``), and a spec-present-but-broken install surfaces as a
    clear error at construction instead of being silently downgraded to eager.
    """
    return importlib.util.find_spec("vllm") is not None


def require_vllm(what: str) -> None:
    """Refuse ``what`` before doing any work when vLLM is not installed.

    The extra is genuinely optional and cannot be made conditional at install time -- a wheel's
    dependencies are resolved from platform metadata, which cannot see whether the machine has a
    GPU -- so "the vLLM backend was asked for on an install that has no vLLM" is a normal outcome
    of ``pip install interp-engine`` rather than a broken environment. It deserves a sentence
    naming both ways out, which is what this is.

    It has to be said here, up front, because the alternative is where the absence surfaces
    otherwise: the vLLM imports are all lazy (deliberately -- this module must import on
    macOS/CPU), so the first thing to touch the engine raises ``ModuleNotFoundError: No module
    named 'vllm'`` from inside ``_ensure_engine``, several frames into an ``await`` on a
    background loop thread, with nothing saying that an extra was missing or that the eager
    backend would have served the same request.
    """
    if vllm_installed():
        return
    raise RuntimeError(
        f"{what} needs vLLM, but vLLM is not installed. vLLM is Linux/CUDA-only and heavy, so it "
        "is an optional extra rather than a base dependency: a plain `pip install interp-engine` "
        "deliberately leaves it out. Either install it -- `pip install 'interp-engine[vllm]'` on a "
        "CUDA box -- or load this model on the eager backend instead "
        "(`load_model(..., backend='eager')`), which serves every point in the registry, "
        "single-stream rather than batched. `interp_engine.vllm_installed()` is this same check, "
        "to branch on rather than catch."
    )


def read_attn_dims(hf_model_id: str, trust_remote_code: bool = True) -> dict[str, Any]:
    """Read attention dims + the config-driven softmax quirks from the HF config.

    ``sliding_window`` / ``layer_types`` are as load-bearing here as the softcap: the fused
    kernel bands the layers ``layer_types`` marks, and the off-kernel recompute has to
    reproduce that band or it attends across the whole prompt. (The third quirk, attention
    sinks, is a weight and arrives with the capture payload instead.)

    ``unsupported`` carries anything attention-relevant in the config that the recompute
    cannot honor, including fields nobody has classified yet -- see ``attn_config``. The
    attention endpoint refuses on a non-empty list rather than serving a plausible-looking
    pattern that is not the model's.
    """
    from transformers import AutoConfig

    from interp_engine.attn_config import unsupported_attn_config
    from interp_engine.facts import resolve_facts, text_config

    cfg = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=trust_remote_code)
    model_facts = resolve_facts(cfg)
    # `unsupported_attn_config` classifies fields on the text config itself, so it needs that
    # object rather than the resolved facts.
    return {
        "n_heads": model_facts.n_heads,
        "n_kv_heads": model_facts.n_kv_heads,
        "head_dim": model_facts.head_dim,
        # Gemma-4 widens the head on non-sliding layers, so `head_dim` above is only the sliding
        # value there and reshaping every layer by it mis-splits a third of them. Carried alongside
        # rather than resolved here because this dict crosses a process boundary as plain data.
        "global_head_dim": model_facts.global_head_dim,
        # The same fact as transformers >= 5.15 states it (`per_layer_config`), which is where
        # Gemma-4's widths moved to -- and the only place its per-layer kv-head count has ever been.
        # Empty on a config that describes one shape for the whole model.
        "per_layer_head_dim": model_facts.per_layer_head_dim,
        "per_layer_kv_heads": model_facts.per_layer_kv_heads,
        # The older spelling of that kv-head count (`num_global_key_value_heads`), and the flag
        # Gemma-4 gates it on. Carried so a transformers below 5.15 -- which states no per-layer
        # table -- still gets the wide layers' count right rather than the sliding one's.
        "global_kv_heads": model_facts.global_kv_heads,
        "k_eq_v": model_facts.k_eq_v,
        # From here on a layer reuses an earlier layer's keys/values and has no v_proj to hook, so
        # `value`/DFA is unavailable there (Gemma-4). None when every layer projects its own.
        "first_kv_shared_layer": model_facts.first_kv_shared_layer,
        # The declared value-head width (MiMo-V2, DeepSeek MLA project a value unlike their q/k).
        # Equal to `head_dim` when the family declares nothing, which is why
        # `value_head_dim_for_layer` compares the two rather than testing this for truth.
        "v_head_dim": model_facts.v_head_dim,
        # What an MLA trunk caches per token per layer in place of K and V; 0 elsewhere. Sizes the
        # static ladder's KV floor, where the head dims above would price 64 heads that never reach
        # the cache.
        "kv_latent_width": model_facts.kv_latent_width,
        # Gemma scales by `query_pre_attn_scalar` rather than head_dim, and the two are not
        # required to be equal. The model-wide value; ask `scaling_for_layer` per layer, since the
        # derivation is a function of a head width that Gemma-4 varies by layer.
        "scaling": model_facts.attn_scaling,
        # What the family *states* its multiplier is (None where it states none, and the derivation
        # applies). Carried separately because the two answer different questions and only this one
        # survives a per-layer head width: Gemma 4 states 1.0 in its modeling code with no config
        # field, and deriving there gives 1/16 on the narrow layers and 1/22.6 on the wide ones,
        # neither of which is what either engine computed.
        "stated_scaling": model_facts.stated_attn_scaling,
        "query_pre_attn_scalar": model_facts.query_pre_attn_scalar,
        "attn_logit_softcapping": model_facts.attn_logit_softcapping,
        "sliding_window": model_facts.sliding_window,
        "layer_types": model_facts.layer_types or (),
        "unsupported": tuple(unsupported_attn_config(text_config(cfg))),
    }


def read_residual_facts(hf_model_id: str, trust_remote_code: bool = True) -> dict[str, Any]:
    """Read the config fields the residual-basis verdict turns on.

    Its own function rather than three more keys on :func:`read_attn_dims`, whose dict is the
    attention payload that crosses to the workers -- these never leave the client process. The
    config read is the same one, and ``transformers`` caches it, so calling both costs one.
    """
    from transformers import AutoConfig

    from interp_engine.facts import resolve_facts

    cfg = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=trust_remote_code)
    model_facts = resolve_facts(cfg)
    architectures = getattr(cfg, "architectures", None) or ()
    return {
        "n_residual_streams": model_facts.n_residual_streams,
        "parallel_attn_mlp": model_facts.parallel_attn_mlp,
        "architecture": architectures[0] if architectures else "",
    }


def is_linear_attention_layer(dims: dict[str, Any], layer: int) -> bool:
    """Whether ``layer`` is linear attention, which has no softmax probs to capture.

    The vLLM-side twin of ``ArchSpec.is_linear_attention_layer`` — same ``layer_types``
    field and the same shared predicate, read from the config dict rather than a loaded eager
    model, so the attention endpoint can refuse the layer on whichever backend is serving.
    """
    return facts.is_linear_attention_layer(tuple(dims.get("layer_types") or ()), layer)


def sliding_window_for_layer(dims: dict[str, Any], layer: int) -> int | None:
    """The window ``layer`` is banded by, or None when it sees the whole prefix.

    Windowed models normally alternate (``layer_types``), so this is per layer rather than
    per model -- banding a ``full_attention`` layer is exactly as wrong as leaving a
    ``sliding_attention`` one unbanded. transformers >= 5 synthesizes ``layer_types`` for
    every family that has a window (including Gemma-2, whose checkpoint config predates the
    field), so the no-``layer_types`` fallback below is only for a model with one global
    window on every layer -- which is also what transformers defaults such a config to.
    """
    return facts.sliding_window_for_layer(dims.get("sliding_window"), tuple(dims.get("layer_types") or ()), layer)


def head_dim_for_layer(dims: dict[str, Any], layer: int) -> int:
    """``layer``'s head dim, which is not constant across layers on Gemma-4.

    The vLLM-side twin of ``ArchSpec.head_dim_for_layer``, through the same shared predicate. Prefer
    it to ``dims["head_dim"]`` anywhere q/k/v is reshaped per head.
    """
    return facts.head_dim_for_layer(
        int(dims["head_dim"]),
        dims.get("global_head_dim"),
        tuple(dims.get("layer_types") or ()),
        layer,
        tuple(dims.get("per_layer_head_dim") or ()),
    )


def kv_heads_for_layer(dims: dict[str, Any], layer: int) -> int:
    """How many key/value heads ``layer`` attends with, per the config.

    The vLLM-side twin of ``ArchSpec.kv_heads_for_layer``. Used as the *expected* count in
    :func:`_heads_in` rather than in place of it: the width in hand is still the authority, and a
    disagreement between the two is worth saying out loud.
    """
    return facts.kv_heads_for_layer(
        int(dims.get("n_kv_heads") or 0),
        layer,
        tuple(dims.get("per_layer_kv_heads") or ()),
        dims.get("global_kv_heads"),
        tuple(dims.get("layer_types") or ()),
        bool(dims.get("k_eq_v")),
    )


def scaling_for_layer(dims: dict[str, Any], layer: int) -> float:
    """The factor ``layer``'s scores are multiplied by, before any softcap.

    The vLLM-side twin of ``ModelFacts.attn_scaling_for_layer``, through the same shared function.
    Prefer it to ``dims["scaling"]``, which is the model-wide value and is derived from the
    model-wide head width -- the one Gemma-4 does not have.
    """
    return facts.attn_scaling_from(
        dims.get("stated_scaling"), dims.get("query_pre_attn_scalar"), head_dim_for_layer(dims, layer)
    )


def value_head_dim_for_layer(dims: dict[str, Any], layer: int) -> int:
    """``layer``'s *value* head width, for reshaping ``value``/``z``.

    The vLLM-side twin of ``ModelFacts.value_head_dim_for_layer``, and it repeats that method's one
    subtlety on purpose: only a declared ``v_head_dim`` that *differs* from ``head_dim`` counts as an
    override, because the field is filled with ``head_dim`` when a family declares nothing. A plain
    truthiness test therefore reads every model as overriding and pins all layers to the model-level
    width -- which on Gemma-4 divides cleanly into the wide layers and mis-splits them in silence.
    """
    declared = int(dims.get("v_head_dim") or 0)
    head_dim = int(dims["head_dim"])
    if declared and declared != head_dim:
        return declared
    return head_dim_for_layer(dims, layer)


def kv_shared_source_layer(dims: dict[str, Any], layer: int) -> int | None:
    """Which layer computed the keys/values ``layer`` attends over, or None if it does itself.

    Non-None means ``layer`` has no k/v projection of its own, so there is nothing to capture there
    and the recompute has to read the named layer instead.
    """
    return facts.kv_source_layer(tuple(dims.get("layer_types") or ()), dims.get("first_kv_shared_layer"), layer)


def _heads_in(tensor: torch.Tensor, head_dim: int, which: str, layer: int, expected: int | None = None) -> int:
    """How many heads a flat ``[seq, heads*head_dim]`` capture holds, by division.

    Counted from the tensor rather than read from the config because the config's ``num_key_value_heads``
    is one number for the whole model and Gemma-4's is not: its ``full_attention`` layers carry a
    different kv-head count *and* a different head width from its ``sliding_attention`` ones (16x256 vs
    4x512 on the 31B). The width in hand is the layer's own, so dividing it by the layer's own head dim
    is the only reading that cannot disagree with the tensor being reshaped.

    ``expected`` is the config's count, checked when the two should agree so that a wrong ``head_dim``
    is reported here rather than as a plausible reshape: an inexact division raises either way, but a
    q-head count that divides *and* disagrees with the config is the silent case.
    """
    width = int(tensor.shape[-1])
    if head_dim <= 0 or width % head_dim:
        raise ValueError(
            f"captured {which} at layer {layer} is {width} wide, which is not a whole number of "
            f"{head_dim}-wide heads. The head dim is resolved per layer (`head_dim_for_layer`), so "
            "either this layer's `layer_types` entry is wrong or the capture is a tensor-parallel shard."
        )
    heads = width // head_dim
    if expected is not None and heads != expected:
        raise ValueError(
            f"captured {which} at layer {layer} holds {heads} heads of {head_dim} ({width} wide), but "
            f"the config says {expected}. The recompute would run on a mis-split tensor."
        )
    return heads


def attn_capture_layers(dims: dict[str, Any], layers: Sequence[int]) -> list[int]:
    """Which layers a worker has to record q/k/v at in order to serve ``layers``.

    Itself, on every family but one. Gemma-4's top layers share an earlier layer's keys and values,
    and the recompute needs the layer that *computed* them -- so asking for a shared layer's scores
    means recording two layers' worth of q/k/v. Call this before ``capture_attn``; the extra layers
    cost one clone each and are dropped from the result by :func:`recompute_attn_from_payloads`,
    which returns only what was asked for.
    """
    wanted = {int(x) for x in layers}
    sources = {s for x in wanted if (s := kv_shared_source_layer(dims, x)) is not None}
    return sorted(wanted | sources)


def _kv_payload(p: dict, which: str, kv_layer: int, layer: int) -> Any:
    """The key or value payload ``layer`` attended over, which may be another layer's."""
    key = attn_payload_key(which, kv_layer)
    if key not in p:
        raise KeyError(
            f"Layer {layer} shares layer {kv_layer}'s keys and values, which were not captured. "
            "Pass `attn_capture_layers(dims, layers)` to `capture_attn` rather than the layers "
            "themselves: the recompute cannot read a shared layer's k/v at the layer that shares them."
        )
    return p[key]


def recompute_attn_from_payloads(payloads, layers, dims, tensor_parallel_size: int = 1) -> dict:
    """Shared client-side: decode worker q/k/v payloads -> {layer: {scores, probs, value}}.

    ``scores`` is the ``attn_scores`` point: the pre-softmax matrix ``probs`` is the softmax of,
    so both come out of one pass rather than one being rebuilt from the other's inputs. It is the
    tensor the fused kernel never materializes, which is why neither is a hook.

    Public because both worker lifecycles hand back the same payloads and neither owns this step:
    :class:`VLLMModel` reaches it through :meth:`VLLMModel.capture_attention`, while a caller
    driving ``vllm.LLM`` itself pairs it with the plugin's ``capture_attn`` / ``collect_attn`` and
    ``read_attn_dims``. Leaving it private meant the second of those had no way to finish the job.

    ``tensor_parallel_size`` is accepted for callers that pass it and no longer changes the
    arithmetic: the worker all-gathers q/k/v across ranks at collect
    (:mod:`interp_engine.vllm_capture._tp`), so rank 0's payload holds every head at any TP size.
    ``_heads_in`` still checks q's head count against ``dims`` and names a shard if one got through.
    """
    del tensor_parallel_size
    p = payloads[0] if isinstance(payloads, list | tuple) else payloads
    out: dict[int, dict[str, torch.Tensor]] = {}
    for layer in layers:
        # A KV-shared layer projects no keys or values of its own: vLLM splits them out of the packed
        # `qkv_proj` (whose k and v slots the checkpoint never loaded), applies neither `k_norm` nor
        # RoPE to them, and hands them to an attention op that ignores the tensors entirely and reads
        # the source layer's KV cache. So the k and v captured *at* such a layer are not what it
        # attended over -- they are unnormed, unrotated, quite possibly uninitialized memory, and
        # they divide by the head width just as cleanly as the real thing. Read the source layer's.
        kv_layer = kv_shared_source_layer(dims, int(layer))
        kv_layer = int(layer) if kv_layer is None else int(kv_layer)
        q = decode_tensor_payload(p[attn_payload_key("q", layer)])
        k = decode_tensor_payload(_kv_payload(p, "k", kv_layer, int(layer)))
        v = decode_tensor_payload(_kv_payload(p, "v", kv_layer, int(layer)))
        sink_payload = p.get(attn_payload_key("sinks", layer))
        # Every dim here is the *layer's*, not the model's. Gemma-4 widens the head on its
        # `full_attention` layers and changes the kv-head count with it, so one config-derived triple
        # describes neither kind of layer: the reshape below raises on the wide layers and mis-splits
        # nothing only because it raises. `head_dim_for_layer` already existed for exactly this and
        # was not being called.
        head_dim = head_dim_for_layer(dims, int(layer))
        n_heads = _heads_in(q, head_dim, "q", int(layer), expected=dims["n_heads"])
        # Checked against the config only where the config states a *per-layer* count. The model-wide
        # one is not a claim about this layer -- it disagrees with Gemma-4's wide layers by design, and
        # on an MLA or tensor-parallel capture it describes something other than the width in hand --
        # so passing it as `expected` would turn a working recompute into a raise.
        states_per_layer_kv = bool(dims.get("per_layer_kv_heads")) or bool(
            dims.get("k_eq_v") and dims.get("global_kv_heads")
        )
        stated_kv = kv_heads_for_layer(dims, int(layer)) if states_per_layer_kv else None
        n_kv_heads = _heads_in(k, head_dim, "k", int(layer), expected=stated_kv)
        scores = recompute_attn_scores(
            q,
            k,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            scaling=scaling_for_layer(dims, int(layer)),
            attn_logit_softcapping=dims["attn_logit_softcapping"],
            sliding_window=sliding_window_for_layer(dims, int(layer)),
        )
        probs = attn_probs_from_scores(
            scores, decode_tensor_payload(sink_payload) if sink_payload is not None else None
        )
        seq = v.shape[0]
        # The value head can be its own width (MiMo-V2, DeepSeek MLA), and on Gemma-4 it follows the
        # layer, so the count comes from this tensor rather than from the key's.
        v_head_dim = value_head_dim_for_layer(dims, int(layer))
        value = v.float().view(seq, _heads_in(v, v_head_dim, "v", int(layer)), v_head_dim)
        out[int(layer)] = {"scores": scores, "probs": probs, "value": value}
    return out


# Capture points whose width is the model's ``hidden_size`` on every architecture, so a
# narrower tensor means the payload is a shard rather than the whole vector. ``z`` and
# ``value`` are deliberately absent -- by declaration, in ``points.Width``: they are
# ``n_heads * head_dim`` wide, which only coincides with ``hidden_size`` on some families (Llama
# yes, Gemma 3 no), so there is no width they can be checked against. Tensor parallelism shards
# them, and the served-point gate is what keeps them off a multi-GPU pod.
_DMODEL_WIDE_POINTS = d_model_wide()


def _assert_full_width_captured(captured: dict[Address, torch.Tensor], hidden_size: int) -> None:
    """Fail loudly when a d_model-wide capture came back narrower than d_model.

    Under tensor parallelism each rank holds a slice of the head- and intermediate-sharded
    tensors, and every caller here reads rank 0's payload alone. The residual points are
    all-reduced before we see them and so are complete, but a point that turns out to be
    sharded would otherwise flow into an SAE encode and either raise a confusing matmul
    error deep in the SAE or, if the widths happen to line up, silently produce numbers
    for a quarter of the model.
    """
    if hidden_size <= 0:
        return
    narrow = {
        str(pt): int(t.shape[-1])
        for pt, t in captured.items()
        if pt.name in _DMODEL_WIDE_POINTS and int(t.shape[-1]) != hidden_size
    }
    if narrow:
        raise RuntimeError(
            f"vLLM capture returned {narrow} for points that are {hidden_size} wide on "
            "this model. The payload is a tensor-parallel shard, not the full vector; "
            "reading it would attribute a slice of the model to the whole."
        )


def _assert_full_prompt_captured(captured: dict[Address, torch.Tensor], n_prompt_tokens: int) -> None:
    """Fail loudly when a capture came back with fewer rows than the prompt has tokens.

    Callers index these tensors by token position, so a short capture is corruption, not
    a degraded result: it silently truncates responses and, when a caller indexes past the
    end on GPU, trips a device-side assert that poisons the CUDA context and takes the
    whole process down. Raising here keeps the blast radius at one request.

    The known cause is a KV-cache prefix hit, which stops the cached positions from ever
    being forwarded. Prefix caching is on engine-wide; what keeps a capture whole is the
    per-request ``cache_salt`` applied by ``VLLMModel._prompt``, so this firing means a
    capture reached the engine without one.
    """
    short = {str(pt): int(t.shape[0]) for pt, t in captured.items() if int(t.shape[0]) != n_prompt_tokens}
    if short:
        raise RuntimeError(
            f"vLLM capture returned {short} rows for a {n_prompt_tokens}-token prompt. "
            "Activations are indexed by token position, so this would corrupt the result. "
            "Most likely this request reached the prefix cache, which skips the forward for "
            "cached positions -- see VLLMModel._prompt, which exists to prevent exactly that."
        )


def _assert_points_captured(captured: Iterable[Address], requested: Sequence[str]) -> None:
    """Fail loudly when a requested point produced no tensor at all.

    The row- and width-checks above both filter on what came back, so they say nothing about a
    point that is simply absent, and an empty capture passes every one of them vacuously. That is
    not hypothetical: build the engine with ``enforce_eager=False`` and every capture returns ``{}``
    with no error, because CUDA graph replay does not run the Python forward that the hooks are
    attached to. Whatever the cause, a caller that asked for an activation and got nothing wants to
    hear about it here rather than downstream, where the absence reads as a ``KeyError`` on a point
    they know they requested.

    ``requested`` is the wire form from :func:`_validate_hook_points`, which is exactly how the
    worker keys its store, so this compares like with like. An empty request list checks nothing,
    which is right -- asking for no points and getting none is not a failure.

    Takes the addresses rather than the capture dict because the streaming path accumulates a set
    of points seen across drains and has no single dict to hand over.
    """
    present = {str(pt) for pt in captured}
    missing = sorted(set(requested) - present)
    if not missing:
        return
    if os.environ.get(STATIC_SKIP_ABSENT_ENV) == "1":
        # A static set built from a point spec rather than from this architecture, which asked the
        # install to drop what the checkpoint does not carry (see STATIC_SKIP_ABSENT_ENV). The install
        # logged each drop with its reason, so the absence here is already accounted for -- and the
        # caller is a matrix that scores a missing point as one blank row.
        logger.warning("vLLM capture: no tensor for %s; returning the %s points that came back", missing, len(present))
        return
    raise RuntimeError(
        f"vLLM capture returned nothing for {missing} (got {sorted(present)}). "
        "The hooks never fired for these points. The usual cause is a graph-replaying engine, "
        "where CUDA graph replay skips the Python forward the hooks live on: reload with "
        'backend="vllm" for hooked capture, or backend="vllm-static" with these points named in '
        "static_points=."
    )


def _decode_rank0(payloads: object) -> dict[Address, torch.Tensor]:
    """Decode the rank-0 capture payload from a ``collective_rpc`` result."""
    return decode_capture_payload(payloads[0] if isinstance(payloads, list | tuple) else payloads)  # type: ignore[index]


async def _settle(collect: Callable[[], Awaitable[_T]], release: Callable[[], Awaitable[None]]) -> _T:
    """A request's last worker calls: ``collect`` its rows, then ``release`` its registration.

    ``release`` runs even when ``collect`` raises, which a capture the worker dropped for lack of
    memory does on purpose. Both are shielded: a server cancels a request's task when its client
    goes, and under anyio every await in a cancelled scope is cancelled too, so an unshielded
    cleanup never reaches the worker and the request's rows stay there for the process lifetime.
    """

    async def run() -> _T:
        try:
            return await collect()
        finally:
            await release()

    return await asyncio.shield(run())


def _legacy_lens_steering(lens_intervention: dict | None, steering_spec: Any, method: str) -> Any:
    """Deprecated ``lens_intervention=`` (``{specs, steer_generated, skip_positions}``) as an ActiveSteering.

    Each wire spec (``op`` ``steer`` / ``ablate`` / ``swap`` with ``layer`` and ``delta``, plus
    ``strength`` / ``max_fraction`` or ``tgt``) becomes the matching op. Specs that share a point and
    stream share one :class:`~interp_engine.steer_specs.SteeringSpec`. None when there is nothing.
    """
    if not lens_intervention or not lens_intervention.get("specs"):
        return None
    from interp_engine.steer import ActiveSteering
    from interp_engine.steer_specs import AblateSpec, LayerSteeringSpec, NormScaledAddSpec, SteeringSpec, SwapSpec

    warnings.warn(
        f"VLLMModel.{method}(lens_intervention=...) is deprecated; open a steer() block with "
        "NormScaledAddSpec / AblateSpec / SwapSpec ops instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    if steering_spec is not None:
        raise ValueError(f"{method} takes steering_spec or lens_intervention, not both.")
    sites: dict[tuple[str, int | None], dict[int, LayerSteeringSpec]] = {}
    for wire in lens_intervention["specs"]:
        op = wire["op"]
        if op == "steer":
            new_op: Any = NormScaledAddSpec(
                vector=wire["delta"],
                strength=float(wire["strength"]),
                max_fraction=float(wire.get("max_fraction", 1.0)),
            )
        elif op == "ablate":
            new_op = AblateSpec(vector=wire["delta"])
        elif op == "swap":
            new_op = SwapSpec(vector=wire["delta"], target=wire["tgt"])
        else:
            raise ValueError(f"Unsupported lens intervention op {op!r}; one of ['ablate', 'steer', 'swap']")
        site = (str(wire.get("point") or "resid_post"), wire.get("stream"))
        sites.setdefault(site, {}).setdefault(int(wire["layer"]), LayerSteeringSpec()).operations.append(new_op)
    skip = [int(i) for i in (lens_intervention.get("skip_positions") or [])]
    return ActiveSteering(
        specs=tuple(
            SteeringSpec(layers=layers, point=point, stream=stream) for (point, stream), layers in sites.items()
        ),
        position_mask=skip or None,
        generated=bool(lens_intervention.get("steer_generated", False)),
    )


def _step_logprobs(per_position: object, index: int, n_logprobs: int) -> list[dict[str, float | int]] | None:
    """One generated position's top-n, in the shape :class:`~interp_engine.steer.GenStep` carries.

    vLLM gives ``completion.logprobs`` as one ``{token_id: Logprob}`` mapping per generated
    position, where ``Logprob`` has ``.logprob`` and ``.rank``. Eager's ``top_logprobs`` gives a
    list of ``{"token_id", "logprob"}`` in descending order, and that is the shape a caller reads,
    so this converts rather than exposing two.

    Sorted by logprob rather than trusted to arrive ordered: the mapping includes the *sampled*
    token even when it was outside the top n, so the insertion order is not the ranking. Trimmed
    to ``n_logprobs`` for the same reason -- asking for 5 must not sometimes yield 6.
    """
    if not n_logprobs or not per_position or index >= len(per_position):  # type: ignore[arg-type]
        return None
    entry = per_position[index]  # type: ignore[index]
    if entry is None:
        return None
    ranked = sorted(entry.items(), key=lambda kv: kv[1].logprob, reverse=True)
    return [{"token_id": int(tid), "logprob": float(lp.logprob)} for tid, lp in ranked[:n_logprobs]]


def _merge_captures(dst: dict[Address, torch.Tensor], new: dict[Address, torch.Tensor]) -> None:
    """Append ``new``'s rows to ``dst`` per point, in forward order.

    Draining a request more than once splits its rows across payloads; concatenating on
    arrival keeps the caller's view identical to a single collect at the end.
    """
    for key, tensor in new.items():
        prev = dst.get(key)
        dst[key] = tensor if prev is None else torch.cat([prev, tensor], dim=0)


def _validate_hook_points(
    points: Sequence[Address | str | tuple[str, int]],
    basis: ResidualBasis | None = None,
    served: frozenset[str] = HOOK_CAPTURE_POINTS,
) -> list[str]:
    """Check the requests against what worker hooks can serve, and return them in **wire** form.

    ``served`` is the worker's point set: the CUDA worker's by default.

    Wire form is the canonical address string, which is what the worker parses and what it keys its
    store with, so nothing between here and the store rebuilds the grammar by hand.

    A layer is required of every point except the trunk-level ones (``embeddings``, ``final_norm``),
    which have no layer to name: the worker reaches those by walking the model rather than by
    indexing its decoder layers. So the refusal is about the *resolver* rather than the wire, and it
    has to be keyed on which point was asked for rather than on the layer simply being absent.

    ``basis`` answers the stream coordinate -- and whether the trunk has streams at all, for the mHC
    points that exist only where it does -- here, on the client, where the model's architecture is
    known and the error can name it. Leaving it to the worker would surface the same refusal from
    inside a forward on another process, several frames deep, as a failed request. It is asked about
    **every** address rather than only the ones carrying a coordinate: an unqualified ``resid_post``
    on a hyper-connection trunk is exactly the request that must be refused, and it is the one no
    downstream check can catch, since the stack it returns is ``d_model`` in its last axis and
    full-length in its first.
    """
    addresses = [to_address(p) for p in points]
    if basis is not None:
        for address in addresses:
            basis.require_stream_coordinate(address.name, address.stream)
            if address.name in hyper_connection_names():
                basis.require_hyper_connections(address.name)
    missing_layer = sorted(str(a) for a in addresses if a.layer is None and a.name not in _GLOBAL_POINTS)
    if missing_layer:
        raise ValueError(
            f"vLLM worker-hook capture installs hooks on a decoder layer, so it needs a layer "
            f"index; got {missing_layer}. (The trunk-level points -- "
            f"{', '.join(sorted(_GLOBAL_POINTS))} -- are the exception and take no layer.)"
        )
    bad = sorted({a.name for a in addresses if a.name not in served})
    if bad:
        # Each refused point quotes its own reason from the point table, which is the difference
        # between "nobody has implemented this yet" and "no such tensor exists in a fused engine" --
        # and is the difference between filing a bug and switching backend.
        raise ValueError(
            f"vLLM worker-hook capture cannot serve points {bad}:\n{refusal_reasons(bad)}\nSupported: {sorted(served)}."
        )
    return [str(a) for a in addresses]


def _why_not_steerable(name: str) -> str:
    """The reason a point is outside :data:`STEERABLE_POINTS`, in the caller's terms.

    Three different kinds of "no", and which one applies decides what the caller does next: give up on
    the idea, wait for a wire change, or switch backend. The coefficient wording is
    :func:`~interp_engine.points.steer_refusal_reason`'s, shared with the worker's own refusal so the
    two cannot say different things about the same point.
    """
    coefficient = points_steer_refusal(name)
    if coefficient is not None:
        return coefficient
    if name in _GLOBAL_POINTS:
        return (
            f"{name} hangs off the trunk rather than a decoder layer, and a steering spec carries its "
            "layer as an int, so there is no way to name a layerless site on the wire. Nothing about "
            "the tensor forbids it -- capture reaches it by walking the trunk -- so this is a "
            "transport gap rather than a semantic one."
        )
    if name in HOOK_CAPTURE_POINTS:  # pragma: no cover - the two exclusions above are exhaustive today
        return f"it is captured but not written, and no reason is recorded:\n{refusal_reasons([name])}"
    return f"this backend does not capture it either, so there is nothing to write:\n{refusal_reasons([name])}"


def _validate_steer_points(specs: Sequence[dict], basis: ResidualBasis | None = None) -> None:
    """Check a flattened steer against what this backend and this model can be written at.

    The counterpart of :func:`_validate_hook_points` for writes, and it exists because the two
    questions genuinely differ: a point can be observable and not writable (the mHC coefficients are
    both), and a stream coordinate means something different on a write than on a read.

    That last difference is the subtle one. On a read, ``stream=k`` claims the *address* names one
    stream, and vLLM refuses it for the residual points because no hook there can reconstruct a
    single stream of a hyper-connection trunk. On a write it claims only that the delta lands in one
    row of a stack the worker already holds, which the mHC kernel wrapper does do -- so this asks the
    basis how many streams there are rather than :meth:`ResidualBasis.require_single_stream`, whose
    ``stream_addressable`` verdict is about the read.
    """
    for spec in specs:
        name = str(spec["point"])
        if name not in STEERABLE_POINTS:
            raise ValueError(
                f"vLLM cannot steer point {name!r}: {_why_not_steerable(name)}\nSteerable: {sorted(STEERABLE_POINTS)}."
            )
        if basis is None:
            continue
        if name in hyper_connection_names():
            basis.require_hyper_connections(name)
        stream = spec.get("stream")
        if stream is None:
            continue
        if basis.n_streams == 1:
            raise ValueError(
                f"stream={stream} was given for a steer of {name!r}, but this model carries a single "
                "residual stream, so there is no stream axis to write one row of. Drop the coordinate."
            )
        if not 0 <= int(stream) < basis.n_streams:
            raise ValueError(
                f"stream={stream} is out of range for a steer of {name!r}: this model carries "
                f"{basis.n_streams} residual streams (valid: 0..{basis.n_streams - 1})."
            )


def _build_extract_engine_kwargs(
    hf_model_id: str,
    *,
    dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int | None,
    enforce_eager: bool,
    trust_remote_code: bool,
    storage_path: str,
    enable_extraction: bool,
    enable_prompt_embeds: bool,
    tensor_parallel_size: int,
    extra_vllm_kwargs: dict[str, Any] | None,
    max_lora_rank: int | None = None,
) -> tuple[dict[str, Any], list[int], int, int, int]:
    """Shared construction kwargs for sync LLM / async AsyncEngineArgs.

    ``enable_extraction=True`` turns on native hidden-state extraction (a speculative
    draft + KV connector). That adds speculative forwards, so it is OFF by default now
    that worker-hook capture serves every point (including resid_post at all layers);
    the extra forwards otherwise pollute decode-time accumulate capture.

    Returns ``(kwargs, layer_ids, n_layers, hidden_size, vocab_size)``. ``vocab_size``
    is the HF config / embedding-table size (may exceed ``tokenizer.vocab_size`` when
    the table is padded, e.g. Llama-3 ``128256`` vs ``128000``).
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=trust_remote_code)
    text_cfg = getattr(cfg, "text_config", None) or cfg
    n_layers = int(text_cfg.num_hidden_layers)
    hidden_size = int(getattr(text_cfg, "hidden_size", 0))
    vocab_size = int(getattr(text_cfg, "vocab_size", 0) or 0)
    kwargs: dict[str, Any] = {
        "model": hf_model_id,
        "dtype": dtype,
        "enforce_eager": enforce_eager,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": trust_remote_code,
        "tensor_parallel_size": int(tensor_parallel_size),
        # Installs capture/steer/lens as worker METHODS, so this backend can drive them by
        # name over collective_rpc. Passing the functions themselves would require
        # VLLM_ALLOW_INSECURE_SERIALIZATION=1, because vLLM v1 msgpack-encodes the call to
        # an out-of-process engine core and refuses function objects.
        "worker_extension_cls": WORKER_EXTENSION_CLS,
    }
    layer_ids: list[int] = []
    if enable_extraction:
        from vllm.config import KVTransferConfig  # pyright: ignore[reportMissingImports]

        extract_kwargs, layer_ids = extract_hidden_states_engine_kwargs(n_layers, shared_storage_path=storage_path)
        kv_dict = extract_kwargs.pop("kv_transfer_config")
        kwargs["kv_transfer_config"] = KVTransferConfig(**kv_dict)
        kwargs.update(extract_kwargs)
    # Prefix caching is ON, and every request that reads or writes forward activations opts
    # itself out per-request via `cache_salt` (see `VLLMModel._prompt`). This was off
    # engine-wide until the two concerns were separated; the reason it had to be is real, and
    # is what `_prompt` now handles:
    #
    # Capture reads the worker's forward activations, so it can only see tokens that are
    # actually forwarded. On a prefix-cache hit vLLM serves the cached positions straight
    # from the KV cache and schedules only the uncached suffix, so those positions never
    # reach the hooks and no amount of accumulating across forwards can recover them
    # (accumulation covers chunked prefill, where every token is still forwarded once).
    # The result is a silently SHORT activation tensor whose length depends on unrelated
    # recent traffic -- which produced truncated /activation/* responses and a fatal CUDA
    # device-side assert once a caller indexed a token position past the short tensor.
    # Steering has the same dependence and a worse failure: a hit serves KV computed
    # WITHOUT the steering vector, so the output is quietly the unsteered one.
    #
    # Only full blocks (16 tokens) are cacheable, so either needs two >=16-token prompts
    # sharing a 16-token prefix on a long-lived server: invisible to the short-prompt
    # parity scripts, routine in production.
    #
    # Turning it back on is worth roughly 1.75x on time-to-first-token for a repeated long
    # prefix (48ms -> 27ms on a 2862-token shared prefix, gemma-3-1b), which is the shape of
    # chat traffic carrying a system prompt. An explicit `extra_vllm_kwargs` entry still wins,
    # since that is applied last, so a caller who wants the old behaviour can pass False.
    kwargs["enable_prefix_caching"] = True
    if enable_prompt_embeds:
        # Prefix caching stays on: vLLM hashes each block's embeds rows into the block key, so an
        # embeds prompt hits the cache only for identical rows. Tested on vLLM 0.28 (GDN hybrid).
        # Accept EmbedsPrompt ({"prompt_embeds": [T, d]}) inputs -- powers NLA concept
        # injection (activation vector spliced into the prompt embedding sequence).
        kwargs["enable_prompt_embeds"] = True
    if max_lora_rank is not None:
        # One adapter at a time (a LoRA read's). A request without a LoRARequest runs the base
        # weights, so capture and the lenses on this engine are unchanged.
        kwargs.update(enable_lora=True, max_lora_rank=int(max_lora_rank), max_loras=1)
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    # Some architectures serve attention through a KV layout that exists in one dtype only, and vLLM's
    # `auto` does not resolve to it -- the model class asserts instead, before any weight is read. The
    # fact is the architecture's, so the engine derives it rather than every caller remembering it, and
    # like every default here it is set before the merge below so an explicit request still wins.
    required_kv_dtype = facts.mandatory_kv_cache_dtype(getattr(cfg, "architectures", None))
    if required_kv_dtype is not None:
        kwargs["kv_cache_dtype"] = required_kv_dtype
    kwargs.update(extra_vllm_kwargs or {})
    return kwargs, layer_ids, n_layers, hidden_size, vocab_size


class VLLMModel:
    """Async engine-owned vLLM backend (vLLM 0.25 ``AsyncLLM``) with native extraction.

    This is the server-facing variant: non-blocking generation + capture. Must be
    constructed inside a running event loop (``AsyncLLM.from_engine_args`` starts a
    background engine loop). Steering write-hooks are added separately.

    This class is all three vLLM backends :func:`~interp_engine.load_model` offers, told apart
    by their tap set. Omitting ``static_points`` is ``backend="vllm"``: hooked, every point,
    chosen per request. Passing ``"auto"`` or a list of addresses is ``backend="vllm-static"``:
    CUDA-graph replay over exactly those taps, which also forces vLLM's breakable path (replay,
    no Dynamo). Passing ``[]`` is ``backend="vllm-generate"``: graphs with inductor and no taps,
    which serves generation only. Prefer ``load_model(backend=...)`` over constructing this
    directly, and prefer ``static_points`` over setting ``enforce_eager`` yourself.
    """

    #: The loop ``self.engine`` was built on, and so the only loop that may await it. See
    #: :meth:`_ensure_engine`. Declared on the class, not just set in ``__init__``, because the
    #: default has to mean something for an instance that never ran ``__init__`` and had its
    #: ``engine`` assigned from outside: nothing here built that engine, so nothing here knows
    #: which loop owns it, and the guard has no business refusing on a guess.
    _engine_loop: asyncio.AbstractEventLoop | None = None

    #: The points this front end's worker can tap: what a request is checked against before it
    #: leaves the process, and what the warmup probe asks the worker about.
    served_points: frozenset[str] = HOOK_CAPTURE_POINTS

    #: Whether the worker has ``collect_projected``. Without it, ``project`` captures, then projects here.
    _projects_on_worker: bool = True

    def __init__(
        self,
        hf_model_id: str,
        *,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        enforce_eager: bool = True,
        trust_remote_code: bool = True,
        storage_path: str = DEFAULT_HS_STORAGE_PATH,
        enable_extraction: bool = False,
        enable_prompt_embeds: bool = False,
        max_lora_rank: int | None = None,
        tensor_parallel_size: int = 1,
        extra_vllm_kwargs: dict[str, Any] | None = None,
        static_points: Sequence[Address | str | tuple[str, int]] | str | None = None,
        static_writes: Sequence[Address | str | tuple[str, int]] | None = None,
    ) -> None:
        import asyncio
        import os

        from transformers import AutoTokenizer

        # Before the tokenizer and config reads below, which hit the network: constructing this
        # class is a claim that vLLM will be there, and it is cheaper to answer it now than after
        # a checkpoint's config has been downloaded. `load_model` asks the same question earlier
        # (it can also offer the eager backend as a fallback rather than an alternative); this is
        # the check for everyone constructing the backend directly.
        require_vllm(f"VLLMModel({hf_model_id!r})")
        self.hf_model_id = hf_model_id
        self._recommended_sampling: RecommendedSampling | None = None
        self.enable_extraction = enable_extraction
        self.enable_prompt_embeds = enable_prompt_embeds
        self.max_lora_rank = max_lora_rank
        self._lora_ids: dict[str, int] = {}
        self.tensor_parallel_size = int(tensor_parallel_size)
        if enable_prompt_embeds:
            # prompt_embeds forces vLLM's legacy V1 model runner, which HANGS during
            # worker init under the default `fork` multiproc method on Blackwell
            # (sm_120 / RTX 5090). `spawn` initializes cleanly. Set before the engine
            # is constructed; requires the host process entrypoint to be import-safe
            # (guarded `__main__`), which the servers are.
            os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        (
            self._engine_kwargs,
            self._layer_ids,
            self.num_hidden_layers,
            self._hidden_size,
            self.vocab_size,
        ) = _build_extract_engine_kwargs(
            hf_model_id,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            trust_remote_code=trust_remote_code,
            storage_path=storage_path,
            enable_extraction=enable_extraction,
            enable_prompt_embeds=enable_prompt_embeds,
            tensor_parallel_size=tensor_parallel_size,
            extra_vllm_kwargs=extra_vllm_kwargs,
            max_lora_rank=max_lora_rank,
        )
        facts = read_residual_facts(hf_model_id, trust_remote_code)
        self._attn_dims = read_attn_dims(hf_model_id, trust_remote_code)
        reads, writes, graph = resolve_static_points(
            static_points,
            n_layers=self.num_hidden_layers,
            n_streams=int(facts["n_residual_streams"] or 1),
            static_writes=static_writes,
            enforce_eager=enforce_eager,
        )
        self._apply_static_state(reads, writes, graph, n_streams=int(facts["n_residual_streams"] or 1))
        # Sync tokenizer attribute (matches VLLMSteerModel.tokenizer; endpoints use it
        # synchronously for decode / apply_chat_template).
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=trust_remote_code)
        # EagerModel-compatible tokenization surface so the shared endpoints (tokenize,
        # activation/*, steer/completion) that call model.to_tokens/.to_str_tokens work
        # on this backend too.
        from interp_engine.chat_formatters import resolve_chat_formatter
        from interp_engine.tokenize import Tokenize

        self.default_prepend_bos = True
        self.tok = Tokenize(
            self.tokenizer,
            default_prepend_bos=True,
            device="cpu",
            # `AutoTokenizer` above deliberately bypasses vLLM's own tokenizer registry, so a
            # family whose chat format is code (DeepSeek-V4) would otherwise arrive here with no
            # way to render chat -- and since this backend hands vLLM token ids rather than
            # messages, vLLM's renderer never runs for our requests either.
            formatter=resolve_chat_formatter(
                [facts["architecture"]] if facts.get("architecture") else None,
                hf_model_id,
                trust_remote_code=trust_remote_code,
            ),
        )
        # The vLLM AsyncLLM is created lazily on first async use: AsyncLLM.from_engine_args
        # starts a background engine loop and must run inside a running event loop, whereas
        # the server constructs models in a (loop-less) thread pool.
        # Typed as Any: vLLM is an optional extra, and after ``_ensure_engine`` every call
        # site treats the engine as present (lazy init is the None case, not a typed state).
        self.engine: Any = None
        self._engine_lock = asyncio.Lock()
        self._static_steer_lock = asyncio.Lock()
        self._static_global_lease: _StaticDeltaLease | None = None
        # Lazily computed in `grad_support` / `residual_basis`, never here: a verdict must not be
        # part of loading.
        self._grad_support: GradSupport | None = None
        self._residual_basis: ResidualBasis | None = None
        self._trust_remote_code = trust_remote_code
        # Non-None while `set_steering` / `set_lens_intervention` have GLOBAL write-hooks
        # installed, holding a token minted at install time. Requests issued during that
        # window carry it as their cache salt, because the hooks change the KV they compute
        # and vLLM's block hash knows nothing about them. See `_prompt`.
        self._global_intervention: str | None = None
        # Per J_bar set, which layers are resident on the worker, after `set_lens_jacobians`.
        self._lens_jacobian_sets: dict[str, frozenset[int]] = {}

    def _apply_static_state(
        self,
        reads: Sequence[Address],
        writes: Sequence[Address],
        graph: bool,
        *,
        n_streams: int = 1,
    ) -> None:
        """Record static sites and, when graphs are on, lower ``max_num_batched_tokens`` to fit."""
        # KV-shared layers need the source layer's q/k/v as well (Gemma-4). Expand here so the
        # worker wrap set matches what capture_attention will harvest.
        expanded = list(reads)
        dims = getattr(self, "_attn_dims", None)
        if dims and any(a.name == "attn" for a in reads):
            seen = {(a.name, a.layer) for a in expanded}
            for address in list(expanded):
                if address.name != "attn" or address.layer is None:
                    continue
                for layer in attn_capture_layers(dims, [int(address.layer)]):
                    extra = Address("attn", int(layer))
                    if (extra.name, extra.layer) not in seen:
                        expanded.append(extra)
                        seen.add((extra.name, extra.layer))
        reads = expanded
        self._static_reads = frozenset(reads)
        self._static_writes = frozenset(writes)
        self._static_env = encode_static_env(reads, writes) if graph else ""
        self._static_self_test_done = False
        if not graph:
            return
        self._engine_kwargs["enforce_eager"] = False
        # Same condition as `apply_breakable_env`: only a non-empty static set turns torch.compile
        # off, and it is that combination a linear-attention trunk cannot survive.
        if reads or writes:
            self._refuse_static_where_vllm_reads_wrong()
            self._pin_decode_only_graphs_on_hybrid_trunk()
        # Reads only. A write allocates a `[1, width]` delta (see `static._alloc_site`), so it does
        # not scale with `max_num_batched_tokens` and has no business in a budget whose whole job is
        # to decide how large that may be. Counting writes here kept stepping the batch down for
        # buffers that are now a few kilobytes.
        n_bufs = sum(3 if a.name == "attn" else 1 for a in reads)
        if not n_bufs or not torch.cuda.is_available():
            return
        max_n = int(self._engine_kwargs.get("max_num_batched_tokens") or 8192)
        cfg = None
        hf_id = getattr(self, "hf_model_id", None)
        if hf_id:
            try:
                from transformers import AutoConfig

                cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=getattr(self, "_trust_remote_code", True))
            except Exception:
                cfg = None
        fitted = fit_max_num_batched_tokens(
            n_sites=n_bufs,
            width=static_read_width(reads, d_model=max(int(self._hidden_size), 1), n_streams=n_streams),
            max_n=max_n,
            device_memory=int(torch.cuda.get_device_properties(0).total_memory),
            gpu_memory_utilization=float(self._engine_kwargs.get("gpu_memory_utilization") or 0.9),
            # As stored, then narrowed if vLLM quantizes at load: a bf16 70B asked for as fp8 is
            # 68 GiB on the card, not 141, and the stored total refused a static set that fits.
            weight_bytes=quantized_on_load_bytes(
                estimate_weight_bytes(
                    self.num_hidden_layers,
                    self._hidden_size,
                    config=cfg,
                    hf_model_id=hf_id,
                ),
                cfg,
                self._engine_kwargs.get("quantization"),
            ),
            max_model_len=int(self._engine_kwargs.get("max_model_len") or max_n),
            kv_width=kv_cache_width(
                n_kv_heads=int(dims.get("n_kv_heads") or 0) if dims else 0,
                head_dim=int(dims.get("head_dim") or 0) if dims else 0,
                v_head_dim=int(dims.get("v_head_dim") or 0) if dims else 0,
                d_model=max(int(self._hidden_size), 1),
                latent_width=int(dims.get("kv_latent_width") or 0) if dims else 0,
            ),
            n_layers=self.num_hidden_layers,
            tensor_parallel_size=self.tensor_parallel_size,
            min_n=facts.min_batched_tokens(cfg) or 0,
        )
        if fitted > max_n:
            # The multimodal floor outranks the fit and the caller's pin alike, and it moves the
            # buffers the other way: on Qwen3.6-27B a 2048 pin became 8192 and 5 GiB of taps.
            logger.warning(
                "raising max_num_batched_tokens %s -> %s: this multimodal checkpoint will not start "
                "below it (facts.min_batched_tokens). Static buffers are sized at the raised value.",
                max_n,
                fitted,
            )
        elif fitted != max_n:
            logger.warning("lowering max_num_batched_tokens %s -> %s so static buffers fit", max_n, fitted)
        if fitted != max_n:
            self._engine_kwargs["max_num_batched_tokens"] = fitted

    def _refuse_static_where_vllm_reads_wrong(self) -> None:
        """Refuse a static set that would report a forward pass vLLM itself gets wrong.

        Raised here rather than after the engine exists, because it is decided by the checkpoint and
        the device: building 26B of weights first only delays the same answer. See
        :func:`~interp_engine.vllm_capture.static.sm100_cudagraph_refusal_reason`.
        """
        compilation = self._engine_kwargs.get("compilation_config")
        reason = sm100_cudagraph_refusal_reason(
            getattr(self, "hf_model_id", None),
            _device_capability(),
            cudagraph_mode=str(compilation.get("cudagraph_mode") or "") if isinstance(compilation, dict) else "",
            batch_invariant=os.environ.get("VLLM_BATCH_INVARIANT") == "1",
        )
        if reason:
            raise ValueError(reason)

    def _pin_decode_only_graphs_on_hybrid_trunk(self) -> None:
        """Capture graphs for decode only when the trunk is linear attention.

        See :func:`~interp_engine.vllm_capture.static.decode_only_graphs_reason` for the measurements;
        the short version is that a static set turns torch.compile off, and vLLM gets a
        GatedDeltaNet prefill wrong in that configuration -- wrong *served output*, not just wrong
        taps. An explicit ``cudagraph_mode`` from the caller wins, because someone pinning it is
        either reproducing this or has a newer vLLM where it is fixed.
        """
        dims = getattr(self, "_attn_dims", None)
        reason = decode_only_graphs_reason((dims or {}).get("layer_types"), self.num_hidden_layers)
        if reason is None:
            return
        compilation = self._engine_kwargs.get("compilation_config")
        if compilation is None:
            compilation = {}
            self._engine_kwargs["compilation_config"] = compilation
        if not isinstance(compilation, dict):
            logger.warning(
                "static on a linear-attention trunk wants cudagraph_mode=%s, but compilation_config "
                "is a %s rather than a dict, so it was left alone. %s",
                DECODE_ONLY_GRAPHS,
                type(compilation).__name__,
                reason,
            )
            return
        pinned = compilation.get("cudagraph_mode")
        if pinned:
            logger.warning("leaving caller's cudagraph_mode=%s in place, but note: %s", pinned, reason)
            return
        compilation["cudagraph_mode"] = DECODE_ONLY_GRAPHS
        logger.info("static: pinning cudagraph_mode=%s. %s", DECODE_ONLY_GRAPHS, reason)

    def configure_static(
        self,
        static_points: Sequence[Address | str | tuple[str, int]] | str,
        static_writes: Sequence[Address | str | tuple[str, int]] | None = None,
    ) -> None:
        """Bind a static set after construction, before the engine exists.

        Inference loads SAEs after ``VLLMModel.__init__``. Those hook names are the Phase 2 static
        set. The engine is lazy, so this must run before :meth:`warmup` / the first request.
        """
        if self.engine is not None:
            raise RuntimeError(
                "configure_static must run before the vLLM engine is built; static wraps are "
                "installed in Worker.load_model, which has already happened."
            )
        facts = read_residual_facts(self.hf_model_id, getattr(self, "_trust_remote_code", True))
        reads, writes, graph = resolve_static_points(
            static_points,
            n_layers=self.num_hidden_layers,
            n_streams=int(facts["n_residual_streams"] or 1),
            static_writes=static_writes,
        )
        if not graph:
            raise ValueError("configure_static needs a static set (a list, 'auto', or static_writes)")
        self._apply_static_state(reads, writes, True, n_streams=int(facts["n_residual_streams"] or 1))

    async def _ensure_engine(self) -> Any:
        # Every async method on this class comes through here, which makes it the one place the
        # engine's loop affinity can be checked once. It is checked on every call rather than
        # only at build time because the engine outlives the loop that built it: a caller who
        # initializes under `asyncio.run(...)` and then serves requests from a different loop
        # gets an engine no one is driving, and `collective_rpc` waits on that silently. The
        # `_engine_lock` below would also refuse a second loop, but only on the build path and
        # with asyncio's own wording, which names neither the model nor the way out.
        #
        # Only when a loop was recorded, which means only when the build below is what produced
        # `self.engine`. An engine assigned from outside belongs to a loop this class never saw.
        bound = self._engine_loop
        if bound is not None:
            refuse_foreign_loop(bound, f"the vLLM engine for {self.hf_model_id!r}")
        if self.engine is None:
            async with self._engine_lock:
                if self.engine is None:
                    from vllm import AsyncEngineArgs  # pyright: ignore[reportMissingImports]
                    from vllm.v1.engine.async_llm import AsyncLLM  # pyright: ignore[reportMissingImports]

                    env = getattr(self, "_static_env", "")
                    if env:
                        os.environ[STATIC_ENV] = env
                    else:
                        os.environ.pop(STATIC_ENV, None)
                    apply_breakable_env(
                        tuple(getattr(self, "_static_reads", ())),
                        tuple(getattr(self, "_static_writes", ())),
                    )
                    # The line below forks a child that suppresses stdout by descriptor, which a
                    # notebook kernel's stdout does not have. See `notebook_stdout`.
                    ensure_stdout_descriptor()
                    self.engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**self._engine_kwargs))
                    self._engine_loop = asyncio.get_running_loop()
        return self.engine

    async def warmup(self) -> None:
        """Build the engine now instead of on the first request, and run one throwaway decode.

        Construction is deliberately lazy (see ``_ensure_engine``), which means the first
        caller to touch this model pays several seconds of engine bring-up. Servers call
        this during startup so that cost lands before traffic; everyone else can ignore it,
        since every async method warms up on its own.

        Building the engine is not enough on its own, because several kernels are compiled the
        first time a *shape* is seen rather than at build time. vLLM's own profiling run is
        prefill-shaped, so the decode-only kernels -- Triton attention's split-softmax path and
        ``reduce_segments``, the Triton sampler -- were left to JIT inside the first real
        request, which is a latency spike vLLM itself logs a warning about. Two tokens is the
        cheapest generation that covers both shapes: the prefill, then one decode step.

        Kernel compile failure is not fatal: the engine is up, and those kernels will compile
        on the first request. A static self-test failure **is** fatal. Declaring a tap is not
        proof ``copy_`` / ``add_`` landed in the recorded graph; a dead write is fluent
        unsteered text. When static sites exist, warmup refuses rather than serve that.
        """
        engine = await self._ensure_engine()
        from vllm import SamplingParams  # pyright: ignore[reportMissingImports]

        token_ids = [int(t) for t in self.tokenizer.encode("Warmup.")] or [0]
        try:
            sp = SamplingParams(max_tokens=2, temperature=0.0)
            async for _ in engine.generate({"prompt_token_ids": token_ids}, sp, self._new_request_id("np-warmup")):
                pass
        except Exception:
            logger.warning(
                "vLLM warmup generation failed; its kernels will compile on the first request instead",
                exc_info=True,
            )
        await self._self_test_static(token_ids)
        await self._probe_resolvable()

    async def _probe_resolvable(self) -> None:
        """Ask the worker which points this checkpoint actually carries, and cache the answer.

        Being *hookable* is a property of the point and is on the client already; being *present* is
        a property of the checkpoint, and the modules are in another process. So :meth:`refuses`
        cannot answer the second half on its own, and answering only the first half over-advertises:
        it promises QK-norm on gpt2 and a router on a dense block, and the caller finds out from a
        worker exception several frames into a request it was told would work.

        Here rather than in :meth:`refuses` because the probe is an ``await`` and the verdict is
        sync, and here rather than at construction because there is no engine to ask yet. Warmup is
        already where a server pays deferred costs before it advertises anything, which is the same
        moment this needs to be true by.

        One round trip for every hookable point at every layer -- a module walk each, no forward, so
        a 60-layer checkpoint costs milliseconds. Asked for the whole set rather than a hand-picked
        "architecture-sensitive" subset, which would be a table to keep in step with the resolver.
        """
        addresses = self._hookable_addresses()
        if not addresses:
            return
        try:
            results = await self.engine.collective_rpc("resolvable_points", args=(addresses,))
        except Exception:
            # A failed probe leaves the verdict as the client half alone, which is what it was
            # before this existed. Not fatal: warmup's contract is to pay costs early, and a model
            # that cannot answer this can still capture.
            logger.warning("vLLM point probe failed; refuses() will answer from the point table alone")
            return
        self._resolvable = dict(results[0]) if results else {}

    def _hookable_addresses(self) -> list[str]:
        """Every hookable point at every layer it applies to, in wire form, for the probe."""
        out: list[str] = []
        for name in sorted(self.served_points):
            spec = point_spec(name, self.residual_basis.n_streams)
            if spec is not None and spec.scope is not Scope.LAYER:
                out.append(name)
            else:
                out.extend(f"{name}.{layer}" for layer in range(int(self.num_hidden_layers)))
        return out

    async def _self_test_static(self, token_ids: Sequence[int]) -> None:
        """Prove static ``copy_`` / ``add_`` ran on graph replay, or refuse to serve.

        One tap and one sentinel write. Hooked engines and ``static_points=[]`` skip this.
        Runs after the kernel warmup generate so the forwards here are replays, not the
        capture that recorded the graphs.
        """
        if getattr(self, "_static_self_test_done", False):
            return
        # Annotated because pyright reads `getattr(self, name, ())` as the default's type alone,
        # which makes the empty tuple the whole story and every loop below unreachable.
        reads: tuple[Address, ...] = tuple(sorted(getattr(self, "_static_reads", ()), key=str))
        writes: tuple[Address, ...] = tuple(sorted(getattr(self, "_static_writes", ()), key=str))
        if not reads and not writes:
            return
        from vllm import SamplingParams  # pyright: ignore[reportMissingImports]

        if reads:
            from interp_engine.vllm_capture.static import ATTN_STATIC_POINT

            site = next((a for a in reads if a.name != ATTN_STATIC_POINT), None)
            if site is not None:
                harvested = await self.capture(token_ids, [site])
                _assert_live_harvest(harvested[site], site)
                logger.info("static self-test: harvest at %s is live", site)
            else:
                layer = next(int(a.layer) for a in reads if a.layer is not None)
                attn = await self.capture_attention(token_ids, [layer])
                payload = attn[layer]["value"]
                _assert_live_harvest(payload, Address(ATTN_STATIC_POINT, layer))
                logger.info("static self-test: attn harvest at layer %s is live", layer)
        if writes:
            write_site = next((a for a in writes if a.name in _STATIC_SENTINEL_WRITE_POINTS), None)
            if write_site is None:
                raise RuntimeError(
                    "static self-test: static_writes has no residual-width site this warmup "
                    f"can prove ({[str(a) for a in writes]}). Pass a resid_post write, or omit "
                    "static_writes."
                )
            sp = SamplingParams(max_tokens=4, temperature=0.0)
            baseline = await self.generate_steered(token_ids, sp)
            steered = await self.generate_steered(
                token_ids, sp, steering_spec=_sentinel_steering(write_site, self.d_model)
            )
            if baseline == steered:
                raise RuntimeError(
                    f"static self-test: sentinel add_ at {write_site} did not change greedy "
                    "output. The static write is not in the replayed graph. A dead add_ is "
                    "fluent unsteered text. Refuse to serve. Check "
                    'VLLM_USE_BREAKABLE_CUDAGRAPH=1, or reload with backend="vllm", whose write '
                    "hooks do not depend on a recorded graph."
                )
            logger.info("static self-test: sentinel write at %s moved greedy output", write_site)
        self._static_self_test_done = True

    async def shutdown(self) -> None:
        """Tear down the vLLM EngineCore, releasing its VRAM. Idempotent.

        vLLM holds the KV cache in a child process that outlives a dropped Python
        reference, so letting the model go out of scope is NOT enough to free the device --
        the next engine bring-up in the same process would fight the orphaned allocation
        for free memory. Call this before loading another model.

        Deliberately NOT guarded by :func:`refuse_foreign_loop`, unlike every other async
        method: teardown has to be reachable from wherever the owner happens to be, and
        refusing it would trade a hang for leaked VRAM. ``AsyncLLM.shutdown`` is synchronous
        and reaps a child process, so it does not need this loop to be the engine's own.
        """
        engine, self.engine = self.engine, None
        self._engine_loop = None
        if engine is not None:
            engine.shutdown()

    @property
    def n_layers(self) -> int:
        return self.num_hidden_layers

    @property
    def d_model(self) -> int:
        return self._hidden_size

    @property
    def n_heads(self) -> int:
        """Query heads for the whole model, not this rank's share. See the protocol."""
        return int(self._attn_dims["n_heads"])

    @property
    def n_kv_heads(self) -> int:
        """Key/value heads for the whole model. See the protocol."""
        return int(self._attn_dims["n_kv_heads"])

    @property
    def head_dim(self) -> int:
        """Width of one attention head. See the protocol."""
        return int(self._attn_dims["head_dim"])

    def is_linear_attention_layer(self, layer: int) -> bool:
        """Whether ``layer`` computes no softmax attention. See the protocol."""
        return is_linear_attention_layer(self._attn_dims, layer)

    @property
    def grad_support(self) -> GradSupport:
        """What kind of gradients this model can provide. See :mod:`interp_engine.autograd_support`.

        Answers from the engine *kwargs* alone -- it never builds the engine or touches a worker, so
        ``/capabilities`` can report it on a lazily-constructed model. That costs nothing in
        precision here: ``through_forward`` is False on every vLLM configuration because the model
        runner's ``execute_model`` is ``@torch.inference_mode()``, and no per-layer attention-kernel
        detail can change that verdict. The remaining blockers are reported to make the error
        actionable, not because they are load-bearing.
        """
        if self._grad_support is None:
            compilation = self._engine_kwargs.get("compilation_config") or {}
            self._grad_support = vllm_grad_support(
                enforce_eager=self._engine_kwargs.get("enforce_eager"),
                cudagraph_mode=(
                    compilation.get("cudagraph_mode") if isinstance(compilation, dict) else None  # pyright: ignore[reportUnknownMemberType]
                ),
                quantization=self._engine_kwargs.get("quantization"),
            )
        return self._grad_support

    @property
    def hooks_available(self) -> bool:
        """Whether Python forward hooks run in this engine.

        False under ``enforce_eager=False``, where CUDA graph replay never calls the Python
        ``forward`` the hooks are attached to. That is **dynamic** hooks only -- static taps
        are :attr:`static_points` / :attr:`static_writes`. Answers from the engine kwargs,
        so ``/capabilities`` can report it before the engine exists.
        """
        return bool(self._engine_kwargs.get("enforce_eager"))

    @property
    def graph_replay(self) -> bool:
        return not bool(self._engine_kwargs.get("enforce_eager", True))

    @property
    def static_points(self) -> tuple[Address, ...]:
        return tuple(sorted(getattr(self, "_static_reads", ()), key=str))

    @property
    def static_writes(self) -> tuple[Address, ...]:
        return tuple(sorted(getattr(self, "_static_writes", ()), key=str))

    def refuses(self, point: Address | str | Point, layer: int | None = None) -> str | None:
        """Why this engine cannot produce ``point``, or None when it can. See the protocol.

        Three questions. Whether *any* vLLM engine can serve the point is the point table's
        business, and a fused engine that never forms the tensor is a different sentence from one
        that has it and baked no tap for it. Whether *this* engine can is about how it was built:
        hooked serves everything the table allows, and a graph engine serves the sites it declared.
        Whether this *checkpoint* carries the module is the worker's answer, cached by
        :meth:`_probe_resolvable` at warmup -- and **before warmup this method answers the first two
        only**, so it can say yes to QK-norm on a model that has none. A server should warm up
        before it advertises, which it has its own reasons to do anyway.

        Tensor parallelism is deliberately not a fourth question. The worker gathers the sharded
        points at collect (:mod:`interp_engine.vllm_capture._tp`), so a multi-GPU pod serves the
        same set a single-GPU one does -- and a caller narrowing by shard width here would refuse
        points that work.

        The answer is about the point, not about one method: the attention pair is never hookable on
        any backend and is reported here as :meth:`capture_attention` can serve it, which is how a
        caller gets it.
        """
        address = to_address(point if layer is None else (point, layer))  # pyright: ignore[reportArgumentType]
        if bad_layer := layer_out_of_range(address, self.n_layers):
            return bad_layer
        if address.name in ("attn_probs", "attn_scores"):
            try:
                # The pair is per-layer, so a caller who named no layer is asking whether the
                # engine recomputes attention at all. Layer 0 answers that.
                if not self._use_static_attn([address.layer if address.layer is not None else 0]):
                    self._require_hooks("Attention capture")
            except REFUSAL_ERRORS as exc:
                return str(exc)
            # The recompute rebuilds the softmax from captured q/k, so a config term it cannot
            # reproduce yields a plausible pattern that is not the model's. That is a refusal, not
            # a caveat: a caller cannot tell the difference by looking at the numbers.
            unsupported = tuple(getattr(self, "_attn_dims", {}).get("unsupported", ()))
            if unsupported:
                return (
                    f"the off-kernel attention recompute cannot reproduce this model's "
                    f"configuration: {'; '.join(unsupported)}"
                )
            return None
        try:
            _validate_hook_points([address], self.residual_basis, self.served_points)
            self._require_capture_points([str(address)], "Activation capture")
        except REFUSAL_ERRORS as exc:
            return str(exc)
        return getattr(self, "_resolvable", {}).get(format_address(address)) or None

    def serves(self, point: Address | str | Point, layer: int | None = None) -> bool:
        """Whether this engine can produce ``point``. See :meth:`refuses` for why not."""
        return self.refuses(point, layer) is None

    def describe(self) -> EngineDescription:
        """What this engine can serve, in one record. See :mod:`interp_engine.describe`."""
        return describe_model(self, self._backend_label(), native_residual=self.enable_extraction)

    def _backend_label(self) -> str:
        if not self.graph_replay:
            return "vllm"
        return "vllm-static" if self.static_points or self.static_writes else "vllm-generate"

    def _require_hooks(self, what: str) -> None:
        """Refuse a hook-dependent operation on a graph-replaying engine with no static site.

        This exists because the failure it replaces is not an error. Steering installs
        ``register_forward_hook``, so with graphs on the hook simply never fires: the request
        succeeds, returns fluent text, and is **unsteered**, with nothing anywhere to say so. Capture
        has :func:`_assert_points_captured` as a backstop and would at least raise, but only after
        paying for the forward and only naming the points, and :meth:`set_lens_intervention` installs
        for later requests so its failure surfaces somewhere else entirely.

        So the check is here, before the work, phrased in terms of the operation the caller asked
        for. Which backend the engine is was fixed when it was built, which makes this a
        deployment mistake rather than a request-level one -- hence naming ``backend=`` and not
        the request.
        """
        if self.hooks_available:
            return
        hf_id = getattr(self, "hf_model_id", None)
        subject = repr(hf_id) if hf_id else "the model"
        raise RuntimeError(
            f"{what} needs Python forward hooks, which this engine does not run: it replays CUDA "
            f"graphs, so vLLM never calls the Python forward the hooks attach to, and it declared "
            f"no static taps to serve the operation instead. Reload {subject} with one of:\n"
            f'    backend="vllm"         # hooked; every point, chosen per request\n'
            f'    backend="vllm-static"  # CUDA graphs over a tap set declared at load\n'
            f"Generation is unaffected on this engine, and so is capture_resid_post, which rides "
            f"vLLM's native extraction rather than hooks."
        )

    def _require_capture_points(self, pts: Sequence[str], what: str) -> None:
        """Allow capture when hooks run, or when every requested point is declared."""
        if self.hooks_available:
            return
        reads = getattr(self, "_static_reads", frozenset())
        if self.graph_replay and reads:
            missing = [to_address(p) for p in pts if to_address(p) not in reads]
            if missing:
                raise ValueError(
                    _static_miss_message(
                        what,
                        sorted(missing, key=str),
                        self.static_points,
                        getattr(self, "hf_model_id", None),
                    )
                )
            return
        self._require_hooks(what)

    def _require_static_writes(self, specs: Sequence[dict], what: str) -> None:
        """Allow static writes when hooks run, or when every write site is declared.

        Additive ``additive`` and the live-read ops (orthogonal, projection_cap, norm_scaled_add, ablate,
        swap) all ride the same static wrap. A site miss is a 400, not a silent no-op.
        """
        if self.hooks_available:
            return
        writes = getattr(self, "_static_writes", frozenset())
        if self.graph_replay and writes:
            for spec in specs:
                op = spec.get("op", SteerMethod.ADDITIVE)
                if op not in STATIC_WRITE_OPS:
                    raise RuntimeError(
                        f"{what} op {op!r} is not one a static write tap can apply; "
                        f"backend='vllm-static' serves {', '.join(sorted(STATIC_WRITE_OPS))}. "
                        f"Use backend='vllm' for this one."
                    )
                site = Address(str(spec.get("point") or "resid_post"), int(spec["layer"]))
                if not any(alias in writes for alias in resid_stream_aliases(site)):
                    raise ValueError(
                        _static_miss_message(
                            what,
                            [site],
                            self.static_writes,
                            getattr(self, "hf_model_id", None),
                            kwarg="static_writes",
                        )
                    )
            return
        self._require_hooks(what)

    def _require_additive_writes(self, specs: Sequence[dict], what: str) -> None:
        self._require_static_writes(specs, what)

    def _use_static_capture(self) -> bool:
        return self.graph_replay and bool(getattr(self, "_static_reads", ()))

    def _use_static_writes(self) -> bool:
        return self.graph_replay and bool(getattr(self, "_static_writes", ()))

    def _static_attn_layers(self) -> set[int]:
        return {int(a.layer) for a in getattr(self, "_static_reads", ()) if a.name == "attn" and a.layer is not None}

    def _use_static_attn(self, layers: Sequence[int]) -> bool:
        if not self.graph_replay:
            return False
        declared = self._static_attn_layers()
        if not declared:
            return False
        dims = getattr(self, "_attn_dims", None)
        if not dims:
            return False
        needed = attn_capture_layers(dims, layers)
        return all(int(layer) in declared for layer in needed)

    def _basis_if_loaded(self) -> ResidualBasis | None:
        cached = getattr(self, "_residual_basis", None)
        if cached is not None:
            return cached
        if getattr(self, "hf_model_id", None):
            return self.residual_basis
        return None

    def _steer_specs(self, steering_spec: Any) -> list[dict]:
        """The worker dicts for a steering spec, refused here if this model cannot serve it.

        One method rather than the conversion inlined at each of the four places that register a
        steer, because a client-side check only some of them make is worse than none: the ones that
        skipped it would fail inside a worker forward on another process instead, which is the failure
        mode :func:`_validate_hook_points` exists to keep capture out of.
        """
        from interp_engine.steer_specs import steering_spec_to_worker_specs

        specs = steering_spec_to_worker_specs(steering_spec)
        _validate_steer_points(specs, self._basis_if_loaded())
        return specs

    async def _register_static_write(
        self,
        rid: str,
        specs: list[dict],
        *,
        position_mask: Any = None,
        prompt_token_ids: Sequence[int] | None = None,
        generated: bool = True,
    ) -> None:
        """Per-request static write. ``position_mask`` becomes skip_positions on the wrap."""
        from interp_engine.steer import resolve_masked_positions

        ids = [int(t) for t in (prompt_token_ids or [])]
        skip = resolve_masked_positions(position_mask, prompt_token_ids=ids, tokenizer=getattr(self, "tokenizer", None))
        scope = {"steer_generated": bool(generated), "skip_positions": skip, "prompt_len": len(ids)}
        await self.engine.collective_rpc(
            "register_static_write",
            args=(rid, specs, skip, len(ids), scope),
        )

    async def _unregister_static_write(self, rid: str) -> None:
        await self.engine.collective_rpc("unregister_static_write", args=(rid,))

    async def _register_write(
        self, rid: str, steering: Any, prompt_token_ids: Sequence[int], *, what: str
    ) -> str | None:
        """Register a :class:`~interp_engine.steer.ActiveSteering` against ``rid``, scope included.

        The one place a request's steer is registered, so the capture and generation paths cannot
        disagree about what a block's ``position_mask`` or ``generated=False`` means here: masked
        prompt positions are skipped on the prefill, and a prompt-only steer leaves every decode
        step alone, on the hooked and the static write alike. Returns which of the two was used --
        ``"static"``, ``"hooks"`` or ``None`` for nothing to steer -- for :meth:`_unregister_write`.
        """
        from interp_engine.steer import resolve_masked_positions

        if steering is None or steering.is_empty():
            return None
        if not (self.hooks_available or self._use_static_writes()):
            self._require_hooks(what)
        worker_specs = self._steer_specs(steering.specs)
        self._require_static_writes(worker_specs, what)
        ids = [int(t) for t in prompt_token_ids]
        if self._use_static_writes():
            await self._register_static_write(
                rid,
                worker_specs,
                position_mask=steering.position_mask,
                prompt_token_ids=ids,
                generated=steering.generated,
            )
            return "static"
        skip = resolve_masked_positions(steering.position_mask, prompt_token_ids=ids, tokenizer=self.tokenizer)
        await self.engine.collective_rpc(
            "register_steering", args=(rid, worker_specs, skip, len(ids), bool(steering.generated))
        )
        return "hooks"

    async def _unregister_write(self, rid: str, registered: str | None) -> None:
        if registered == "static":
            await self._unregister_static_write(rid)
        elif registered == "hooks":
            await self.engine.collective_rpc("unregister_steering", args=(rid,))

    @property
    def residual_basis(self) -> ResidualBasis:
        """How this model's residual stream is structured. See :mod:`interp_engine.residual_basis`.

        Answers from the HF config alone, like :attr:`grad_support`, so ``/capabilities`` can report
        it without building the engine. The verdict differs from the eager one in exactly one way,
        and it is about this backend rather than the model: the capture wire key has no stream
        coordinate, so a stream cannot be asked for even where the model has several.
        """
        if self._residual_basis is None:
            self._residual_basis = vllm_residual_basis(**read_residual_facts(self.hf_model_id, self._trust_remote_code))
        return self._residual_basis

    # EagerModel-compatible tokenization (delegates to the Tokenize layer).
    def to_tokens(self, text, **kwargs):
        return self.tok.to_tokens(text, **kwargs)

    def to_str_tokens(self, text, **kwargs):
        return self.tok.to_str_tokens(text, **kwargs)

    def to_string(self, tokens):
        return self.tok.to_string(tokens)

    @property
    def tokenizer_prepends_bos(self) -> bool:
        return self.tok.tokenizer_prepends_bos

    @property
    def recommended_sampling(self) -> RecommendedSampling:
        """Read here, not asked of the engine: vLLM folds the file into its own request defaults
        only for a ``SamplingParams`` it builds itself, and this backend builds its own."""
        # `getattr`: a test double built without `__init__` has no slot yet.
        stated = getattr(self, "_recommended_sampling", None)
        if stated is None:
            stated = self._recommended_sampling = read_recommended_sampling(self.hf_model_id)
        return stated

    def sampling_settings(
        self,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
    ) -> SamplingSettings:
        """See the protocol."""
        return resolve_sampling(
            self.recommended_sampling,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            presence_penalty=presence_penalty,
        )

    async def generate(
        self,
        prompts,
        sampling_params: Any,
        *,
        steering_spec: Any = None,
        position_mask: Any = None,
        stream: bool = False,
        capture_points: Sequence[Address | str | tuple[str, int]] | None = None,
        capture_out: dict[Address, torch.Tensor] | None = None,
        drain_every: int = 64,
    ):
        """Generate from a text prompt, with optional steering.

        Accepts a prompt string (or ``[string]``); tokenizes and delegates to
        :meth:`generate_steered`. Returns the full text (``stream=False``) or an async
        generator of text deltas (``stream=True``). ``position_mask`` (a
        ``interp_engine.SteerMask`` preset or ``list[int]`` of positions) excludes prompt
        positions from steering (e.g. ``SteerMask.SPECIAL_TOKENS``).

        Note that this takes vLLM's ``SamplingParams`` but does NOT return vLLM's
        ``list[RequestOutput]`` -- it returns text, because that is what the steering
        endpoints want. For the vLLM-shaped result (``.text`` / ``.token_ids`` /
        ``.logprobs`` / ``.finish_reason``) use :meth:`generate_full`.
        """
        prompt = prompts[0] if isinstance(prompts, list | tuple) else prompts
        if isinstance(prompt, str):
            token_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        else:
            token_ids = list(prompt)
        return await self.generate_steered(
            token_ids,
            sampling_params,
            steering_spec=steering_spec,
            position_mask=position_mask,
            stream=stream,
            capture_points=capture_points,
            capture_out=capture_out,
            drain_every=drain_every,
        )

    async def generate_steered(
        self,
        prompt_token_ids: Sequence[int],
        sampling_params: Any,
        *,
        steering_spec: Any = None,
        position_mask: Any = None,
        generated: bool = True,
        stream: bool = False,
        capture_points: Sequence[Address | str | tuple[str, int]] | None = None,
        capture_out: dict[Address, torch.Tensor] | None = None,
        drain_every: int = 64,
    ):
        """VLLMSteerModel-style generation: apply an engine SteeringSpec, then generate.

        ``steering_spec`` is a ``interp_engine.SteeringSpec``, a list of them, or None. Returns
        the full text (stream=False) or an async generator of text deltas
        (stream=True). Steering is installed for the duration and cleared after.
        ``position_mask`` (``SteerMask`` preset or ``list[int]``) excludes prompt positions
        from steering; it's resolved here against the actual prompt token ids + tokenizer.
        ``generated=False`` confines the steer to the prompt. Single request-locked use.

        ``capture_points`` registers activation capture on the SAME request, so the
        generation's own forwards yield ``[prompt + generated - 1, width]`` per point
        instead of a caller re-prefilling the finished text (the final sampled token is
        never processed through the model). Rows are merged into ``capture_out``, which
        is complete once the returned generator is exhausted / the call returns. When a
        steering spec is also active the hook steers before it captures, so the captured
        rows are post-intervention -- which is what makes a steered generation double as
        its own post-cap read.

        ``drain_every`` bounds how long captured rows sit in worker memory: they are
        moved to the host after the first forward (the prefill, which is the bulk) and
        every ``drain_every`` streamed steps thereafter. Each drain is one
        ``collective_rpc``, so this trades a small per-request RPC count against holding
        a ``[prompt + generated, hidden]`` tensor on every GPU for the whole generation.
        """
        from interp_engine.steer import ActiveSteering
        from interp_engine.steer_specs import steering_specs

        await self._ensure_engine()
        specs = steering_specs(steering_spec)
        steered = any(not spec.is_empty() for spec in specs)
        capturing = bool(capture_points) and capture_out is not None
        if capturing and not (self.hooks_available or self._use_static_capture()):
            self._require_hooks("Capture during generation")
        token_ids = [int(t) for t in prompt_token_ids]
        rid = self._new_request_id("np-steer")
        prompt = self._prompt(token_ids, private_kv_for=rid if (capturing or steered) else None)
        pts: list[str] = []
        if capturing:
            assert capture_points is not None
            pts = _validate_hook_points(capture_points, self._basis_if_loaded(), self.served_points)
            self._require_capture_points(pts, "Capture during generation")
        static_cap = capturing and self._use_static_capture()
        steering = None
        if steered:
            steering = ActiveSteering(specs=specs, position_mask=position_mask, generated=generated)
        registered = await self._register_write(rid, steering, token_ids, what="Steered generation")
        if capturing and not static_cap:
            await self.engine.collective_rpc("register_capture", args=(rid, pts))
        elif static_cap:
            await self.engine.collective_rpc("register_static_capture", args=(rid, pts))

        async def _finish() -> None:
            async def collect() -> object:
                if not (capturing and capture_out is not None):
                    return None
                method = "collect_static" if static_cap else "collect_request"
                return await self.engine.collective_rpc(method, args=(rid,))

            payloads = await _settle(collect, lambda: self._unregister_write(rid, registered))
            if capturing and capture_out is not None:
                _merge_captures(capture_out, _decode_rank0(payloads))
                _assert_points_captured(capture_out, pts)
                _assert_full_width_captured(capture_out, self._hidden_size)

        if stream:

            async def _stream():
                prev = ""
                steps = 0
                try:
                    async for out in self.engine.generate(prompt, sampling_params, rid):
                        text = out.outputs[0].text
                        if len(text) > len(prev):
                            yield text[len(prev) :]
                            prev = text
                        if capturing and capture_out is not None:
                            steps += 1
                            if steps == 1 or steps % drain_every == 0:
                                await self._drain_into(rid, capture_out, static=static_cap)
                finally:
                    await _finish()

            return _stream()

        try:
            out = await self._run_one(prompt, sampling_params, request_id=rid)
            return out.outputs[0].text
        finally:
            await _finish()

    async def _generate_request_outputs(
        self,
        prompt_token_ids: Sequence[int],
        sampling_params: Any,
        *,
        steering: Any = None,
    ):
        """Stream this request's ``RequestOutput``s, with a per-request steer if one was given.

        ``steering`` is an :class:`~interp_engine.steer.ActiveSteering` or None. The
        register/generate/unregister dance, in one place, so that everything wanting a steered
        stream shares it. Extracted rather than copied because the ``finally`` is the load-bearing
        part: ``unregister_steering`` has to run on every exit path including a client
        disconnecting mid-stream, and a second hand-written copy of that is a hook leak onto every
        later request waiting to happen.

        Raw ``RequestOutput``s rather than a decoded shape, because the two callers want different
        things off them -- text deltas for the SSE path, per-token ids and logprobs for
        :meth:`generate_steps`.
        """
        await self._ensure_engine()
        steered = steering is not None and not steering.is_empty()
        token_ids = [int(t) for t in prompt_token_ids]
        rid = self._new_request_id("np-steer" if steered else "np-steps")
        prompt = self._prompt(token_ids, private_kv_for=rid if steered else None)
        registered = await self._register_write(rid, steering, token_ids, what="Steered generation")
        try:
            async for out in self.engine.generate(prompt, sampling_params, rid):
                yield out
        finally:
            await asyncio.shield(self._unregister_write(rid, registered))

    async def _drain_into(self, rid: str, capture_out: dict[Address, torch.Tensor], *, static: bool = False) -> None:
        """Move ``rid``'s captured rows so far to the host, leaving the hooks / static taps installed."""
        if static:
            payloads = await self.engine.collective_rpc("drain_static", args=(rid,))
        else:
            payloads = await self.engine.collective_rpc("drain_request", args=(rid,))
        _merge_captures(capture_out, _decode_rank0(payloads))

    async def _run_one(self, prompt: dict, sampling_params: Any, request_id: str | None = None):
        import uuid

        await self._ensure_engine()
        rid = request_id or f"np-{uuid.uuid4().hex}"
        final = None
        async for out in self.engine.generate(prompt, sampling_params, rid):
            final = out
        if final is None:
            raise RuntimeError("vLLM produced no output")
        return final

    @staticmethod
    def _new_request_id(prefix: str = "np") -> str:
        import uuid

        return f"{prefix}-{uuid.uuid4().hex}"

    def _prompt(self, token_ids: Sequence[int], *, private_kv_for: str | None = None) -> dict[str, Any]:
        """Build the vLLM prompt dict, deciding whether this request may share cached KV.

        Prefix caching is on engine-wide, and this is the one place that decides who opts out.
        Pass ``private_kv_for=<request id>`` for any request that reads the forward activations
        (capture, attention, native extraction) or changes them (steering, lens); leave it off
        for plain generation, which is what the caching is for.

        The mechanism is vLLM's ``cache_salt``. It goes into the ``extra_keys`` of the FIRST
        block's hash, and block hashes chain through ``parent_block_hash``, so a salt no other
        request has used makes every one of this request's block hashes unique. That isolates it
        in both directions, which is what correctness needs here: it cannot HIT a block computed
        by someone else (so a capture forwards every token, and a steered request computes its
        own steered KV rather than inheriting unsteered KV), and its blocks cannot be hit BY
        anyone else (so steered KV never gets served to a later plain request). Requests that
        pass no salt keep vLLM's ordinary hashing and go on sharing with each other.

        The request id is reused as the salt rather than minting a second token, so a cache-hit
        question can be traced back to the request that owned the blocks. Any unique string works.

        Isolation is per request, so two identical capture calls share nothing and each pays a
        full prefill. That is deliberate: making them share would mean deriving the salt from
        everything that affects the activations, including steering vectors, and a salt that is
        wrong in either direction is silent corruption. A missed cache hit only costs time.

        ``_global_intervention`` covers what a per-request salt cannot. ``set_steering`` and
        ``set_lens_intervention`` install write-hooks that apply to EVERY later request, so
        during that window even a plain generation computes intervened KV. Salting those
        requests with a token minted when the hooks went in keeps that KV from being served
        after ``clear_steering``, while still letting requests within one window share with each
        other -- they see the same hooks, so their blocks really are interchangeable.
        """
        prompt: dict[str, Any] = {"prompt_token_ids": [int(t) for t in token_ids]}
        salt = private_kv_for or self._global_intervention
        if salt is not None:
            prompt["cache_salt"] = salt
        return prompt

    async def generate_full(
        self,
        prompt_token_ids: Sequence[int],
        *,
        max_tokens: int = 200,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
        logprobs: int | None = None,
    ):
        """Non-streaming generate; returns the vLLM ``CompletionOutput``.

        The result exposes ``.text``, ``.token_ids``, ``.logprobs`` (when
        ``logprobs`` requested), and ``.finish_reason`` for the server adapter.
        """
        from vllm import SamplingParams  # pyright: ignore[reportMissingImports]

        settings = self.sampling_settings(
            temperature=temperature, top_k=top_k, top_p=top_p, presence_penalty=presence_penalty
        )
        out = await self._run_one(
            self._prompt(prompt_token_ids),
            SamplingParams(max_tokens=max_tokens, seed=seed, logprobs=logprobs, **settings.vllm_kwargs()),
        )
        return out.outputs[0]

    def _text_sampling(self, *, max_tokens: int, settings: SamplingSettings, seed: int | None) -> Any:
        """Sampling for the protocol's text methods, decoded the way eager decodes.

        Eager decodes every sampled id, special tokens included, so its text carries a chat turn's
        structure -- the channel markers and the end-of-turn token. vLLM's detokenizer drops those
        by default, which made the same prompt read differently across backends and left an
        assistant turn unrecoverable from this one. :meth:`generate_steps` decodes each id itself,
        so its steps concatenate to this text; ``generate_full`` keeps vLLM's defaults.
        """
        from vllm import SamplingParams  # pyright: ignore[reportMissingImports]

        return SamplingParams(max_tokens=max_tokens, seed=seed, skip_special_tokens=False, **settings.vllm_kwargs())

    async def generate_text(
        self,
        prompt_token_ids: Sequence[int],
        *,
        max_tokens: int = 200,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
    ) -> str:
        """Generate the completion text. Honors an open ``steer()`` context; see the protocol."""
        from interp_engine.steer import active_steering

        steering = active_steering(self)
        text = await self.generate_steered(
            prompt_token_ids,
            self._text_sampling(
                max_tokens=max_tokens,
                settings=self.sampling_settings(
                    temperature=temperature, top_k=top_k, top_p=top_p, presence_penalty=presence_penalty
                ),
                seed=seed,
            ),
            steering_spec=None if steering is None else steering.specs,
            position_mask=None if steering is None else steering.position_mask,
        )
        # `stream=False` returns the text; the union on the signature is the streaming form's.
        return str(text)

    async def generate_stream(
        self,
        prompt_token_ids: Sequence[int],
        *,
        max_tokens: int = 200,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
    ):
        """Yield decoded text deltas as generation proceeds (for SSE endpoints).

        Inside an open ``steer()`` context the request is steered, by the same per-request path
        ``generate_steered`` takes; nothing is installed on the shared engine. See the protocol.
        """
        from interp_engine.steer import active_steering

        steering = active_steering(self)
        deltas = await self.generate_steered(
            prompt_token_ids,
            self._text_sampling(
                max_tokens=max_tokens,
                settings=self.sampling_settings(
                    temperature=temperature, top_k=top_k, top_p=top_p, presence_penalty=presence_penalty
                ),
                seed=seed,
            ),
            steering_spec=None if steering is None else steering.specs,
            position_mask=None if steering is None else steering.position_mask,
            generated=True if steering is None else steering.generated,
            stream=True,
        )
        async for delta in deltas:
            yield delta

    #: vLLM's own default cap on how many logprobs a request may ask for. An engine built with a
    #: different ``max_logprobs`` overrides it; this is the value to compare against when the
    #: engine was built with the default, which is every engine this backend builds.
    DEFAULT_MAX_LOGPROBS = 20

    async def generate_steps(
        self,
        prompt_token_ids: Sequence[int],
        *,
        max_tokens: int = 64,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        stop_at_eos: bool = True,
        n_logprobs: int = 0,
        seed: int | None = None,
        steering_spec: Any = None,
        position_mask: Any = None,
        generated: bool = True,
    ):
        """Yield one :class:`~interp_engine.steer.GenStep` per generated token.

        The per-token twin of :meth:`generate_stream`, which yields decoded text deltas -- a
        delta is not a token (one token can decode to nothing until the next arrives) and carries
        neither the id nor the logprobs. This is what backs the free
        :func:`interp_engine.generate_stream` on this backend, so a caller gets the same
        ``GenStep`` stream here as on eager.

        ``GenStep.logits`` is always ``None``: the sampler runs in the worker and the logit tensor
        is never shipped out. ``n_logprobs`` is the portable way to ask what else was likely, and
        it is checked against the engine's cap up front rather than silently truncated.

        With ``steering_spec`` the steer is registered against THIS request only, so a request
        co-batched with it is unaffected -- unlike :meth:`set_steering`, which installs a hook
        over the whole forward. That is why the free ``steer()`` context routes here, and why an
        open ``steer()`` block is read when no spec is passed: the block cannot install anything
        on this backend, so the request has to carry it, ``position_mask`` and ``generated``
        included.
        """
        from interp_engine.steer import ActiveSteering, active_steering
        from interp_engine.steer_specs import steering_specs

        if steering_spec is not None:
            steering = ActiveSteering(
                specs=steering_specs(steering_spec), position_mask=position_mask, generated=generated
            )
        else:
            steering = active_steering(self)
        sampling = self._step_sampling(
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            presence_penalty=presence_penalty,
            stop_at_eos=stop_at_eos,
            n_logprobs=n_logprobs,
            seed=seed,
        )
        outputs = self._generate_request_outputs(prompt_token_ids, sampling, steering=steering)
        async for step in self._steps_from_outputs(outputs, n_logprobs):
            yield step

    async def generate_from_embeds(
        self,
        prompt_embeds: torch.Tensor,
        sampling_params: Any,
        *,
        request_id: str | None = None,
        stream: bool = False,
    ) -> Any:
        """Deprecated: use :meth:`generate_steps_from_embeds` or :meth:`sample_from_embeds`.

        Generates from ``prompt_embeds`` (``[num_tokens, hidden]``) with vLLM ``SamplingParams``.
        Returns the final ``RequestOutput``, or with ``stream=True`` an async generator of them.
        ``Any`` because vLLM is optional and ``stream`` selects the type.
        """
        warnings.warn(
            "VLLMModel.generate_from_embeds is deprecated; use generate_steps_from_embeds or sample_from_embeds.",
            DeprecationWarning,
            stacklevel=2,
        )
        prompt = self._embeds_prompt(prompt_embeds, "generate_from_embeds")
        await self._ensure_engine()
        prompt["prompt_embeds"] = prompt["prompt_embeds"].to(self.engine.model_config.dtype).contiguous()
        rid = request_id or self._new_request_id("np-embed")

        if stream:

            async def _stream():
                async for out in self.engine.generate(prompt, sampling_params, rid):
                    yield out

            return _stream()

        final = None
        async for out in self.engine.generate(prompt, sampling_params, rid):
            final = out
        if final is None:
            raise RuntimeError("vLLM produced no output for prompt_embeds request")
        return final

    async def generate_steps_from_embeds(
        self,
        prompt_embeds: torch.Tensor,
        *,
        max_tokens: int = 64,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        stop_at_eos: bool = True,
        n_logprobs: int = 0,
        seed: int | None = None,
        lora_path: str | None = None,
    ):
        """:meth:`generate_steps` over vLLM's ``EmbedsPrompt``. See the protocol.

        Needs an engine built with ``enable_prompt_embeds=True``, which is refused here by name
        rather than left to the input processor's error. The rows are cast to the engine's own
        model dtype, as vLLM's HTTP front end does for its clients, so a caller holding fp32
        activations need not know what ``dtype="auto"`` resolved to.

        An open ``steer()`` block is refused: the per-request steer is built around an ids prompt
        (its position mask resolves against the ids) and has not been carried to this one.

        ``lora_path`` runs this one request through a PEFT adapter in vLLM's layout (the engine
        needs ``max_lora_rank``). Closing the iterator early aborts the request in the engine.
        """
        from contextlib import aclosing

        prompt = self._embeds_prompt(prompt_embeds, "generate_steps_from_embeds")
        sampling = self._step_sampling(
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            presence_penalty=presence_penalty,
            stop_at_eos=stop_at_eos,
            n_logprobs=n_logprobs,
            seed=seed,
        )
        lora = self._lora_request(lora_path) if lora_path is not None else None
        await self._ensure_engine()
        prompt["prompt_embeds"] = prompt["prompt_embeds"].to(self.engine.model_config.dtype).contiguous()
        rid = self._new_request_id("np-embeds")
        async with aclosing(self.engine.generate(prompt, sampling, rid, lora_request=lora)) as outputs:
            async for step in self._steps_from_outputs(outputs, n_logprobs):
                yield step

    async def sample_from_embeds(
        self,
        prompt_embeds: torch.Tensor,
        *,
        n: int = 1,
        max_tokens: int = 64,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
        lora_path: str | None = None,
    ) -> list[EmbedsSample]:
        """``n`` completions of one embeds prompt, as ONE vLLM request with ``SamplingParams(n=n)``.

        What :func:`interp_engine.sample_from_embeds` runs on this backend. vLLM gives completion
        ``j`` the seed ``seed + j``, so a fixed ``seed`` repeats the set. Nothing streams: the
        request reports once, when all ``n`` are done, and vLLM does not detokenize along the way,
        which is most of the per-token cost of :meth:`generate_steps_from_embeds` at high
        concurrency. The same prompt checks and ``lora_path`` apply as there.
        """
        from vllm import SamplingParams  # pyright: ignore[reportMissingImports]
        from vllm.sampling_params import RequestOutputKind  # pyright: ignore[reportMissingImports]

        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        prompt = self._embeds_prompt(prompt_embeds, "sample_from_embeds")
        settings = self.sampling_settings(
            temperature=temperature, top_k=top_k, top_p=top_p, presence_penalty=presence_penalty
        )
        sampling = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            seed=seed,
            detokenize=False,
            output_kind=RequestOutputKind.FINAL_ONLY,
            **settings.vllm_kwargs(),
        )
        lora = self._lora_request(lora_path) if lora_path is not None else None
        await self._ensure_engine()
        prompt["prompt_embeds"] = prompt["prompt_embeds"].to(self.engine.model_config.dtype).contiguous()
        final = None
        async for out in self.engine.generate(prompt, sampling, self._new_request_id("np-sample"), lora_request=lora):
            final = out
        if final is None:
            raise RuntimeError("vLLM produced no output")
        samples = []
        for c in sorted(final.outputs, key=lambda c: c.index):
            ids = [int(t) for t in c.token_ids]
            # A stop token ends the ids; a stop string (never set here) would leave nothing to drop.
            stopped = c.finish_reason == "stop"
            if stopped and ids and not isinstance(c.stop_reason, str):
                ids = ids[:-1]
            text = self.tokenizer.decode(ids, clean_up_tokenization_spaces=False)
            samples.append(EmbedsSample(text=text, token_ids=ids, finish="eos" if stopped else "length"))
        return samples

    def _embeds_prompt(self, prompt_embeds: torch.Tensor, what: str) -> dict[str, Any]:
        """vLLM's ``EmbedsPrompt`` dict for ``prompt_embeds`` (on CPU, not yet cast), after the checks.

        An open ``steer()`` block is refused: the per-request steer is built around an ids prompt
        (its position mask resolves against the ids) and has not been carried to this one.
        """
        from interp_engine.steer import active_steering

        if active_steering(self) is not None:
            raise refuse(self, f"steer() around {what}", capability="steered_prompt_embeds")
        if not self.enable_prompt_embeds:
            raise ValueError(
                f"{what} needs an engine built with enable_prompt_embeds=True; this "
                "one was not. Pass it to load_model (or VLLMModel): vLLM's input processor rejects an "
                "embeds prompt on an engine that was not told to expect one."
            )
        prompt: dict[str, Any] = {"prompt_embeds": checked_prompt_embeds(prompt_embeds, self.d_model).to("cpu")}
        if self._global_intervention is not None:
            prompt["cache_salt"] = self._global_intervention
        return prompt

    def _lora_request(self, path: str) -> Any:
        """vLLM's ``LoRARequest`` for the adapter at ``path``, with one stable id per path."""
        from vllm.lora.request import LoRARequest  # pyright: ignore[reportMissingImports]

        if self.max_lora_rank is None:
            raise ValueError(
                "A LoRA request needs an engine built with max_lora_rank (vLLM's enable_lora); this "
                "one was not. Pass max_lora_rank to load_model (or VLLMModel)."
            )
        lora_id = self._lora_ids.setdefault(path, len(self._lora_ids) + 1)
        return LoRARequest(f"adapter-{lora_id}", lora_id, path)

    async def embed_rows(self, token_ids: Sequence[int]) -> torch.Tensor:
        """The input embeddings of ``token_ids`` ([k, d_model]), as the forward sees them."""
        await self._ensure_engine()
        results = await self.engine.collective_rpc("embed_rows", args=([int(t) for t in token_ids],))
        out = results[0] if isinstance(results, list | tuple) else results
        return decode_tensor_payload(out)

    def _step_sampling(
        self,
        *,
        max_tokens: int,
        temperature: float | None,
        top_k: int | None,
        top_p: float | None,
        presence_penalty: float | None,
        stop_at_eos: bool,
        n_logprobs: int,
        seed: int | None,
    ) -> Any:
        """The protocol's per-step sampling knobs, resolved, as vLLM's ``SamplingParams``.

        ``n_logprobs`` is checked against the engine's cap up front rather than silently truncated:
        vLLM would reject the request, so it is refused here where the number can be named.
        """
        from vllm import SamplingParams  # pyright: ignore[reportMissingImports]

        if n_logprobs > self.DEFAULT_MAX_LOGPROBS:
            raise ValueError(
                f"n_logprobs={n_logprobs} exceeds this engine's max_logprobs "
                f"({self.DEFAULT_MAX_LOGPROBS}). vLLM would reject the request rather than "
                "truncate, so it is refused here where the number can be named. Ask for at most "
                f"{self.DEFAULT_MAX_LOGPROBS}, or build the engine with a higher max_logprobs via "
                "extra_vllm_kwargs."
            )
        settings = self.sampling_settings(
            temperature=temperature, top_k=top_k, top_p=top_p, presence_penalty=presence_penalty
        )
        return SamplingParams(
            max_tokens=max_tokens,
            ignore_eos=not stop_at_eos,
            logprobs=n_logprobs or None,
            seed=seed,
            **settings.vllm_kwargs(),
        )

    async def _steps_from_outputs(self, outputs: AsyncIterator[Any], n_logprobs: int):
        """One ``GenStep`` per new token across a request's ``RequestOutput`` stream.

        vLLM reports the whole completion so far on every output, so this tracks how many ids it
        has already emitted and yields only the tail. Each id is decoded on its own, as eager does,
        so the steps concatenate to the completion text.
        """
        from interp_engine.steer import GenStep

        emitted = 0
        async for out in outputs:
            completion = out.outputs[0]
            token_ids, logprobs = completion.token_ids, completion.logprobs
            while emitted < len(token_ids):
                token_id = int(token_ids[emitted])
                yield GenStep(
                    token_id=token_id,
                    token_str=self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False),
                    logits=None,
                    logprobs=_step_logprobs(logprobs, emitted, n_logprobs),
                )
                emitted += 1

    async def capture_resid_post(
        self,
        prompt_token_ids: Sequence[int],
        layers: Sequence[int] | None = None,
    ) -> dict[int, torch.Tensor]:
        from vllm import SamplingParams  # pyright: ignore[reportMissingImports]

        # Native extraction reads what the forward produced, so like hook capture it needs every
        # token forwarded rather than served from cache.
        rid = self._new_request_id("np-resid")
        out = await self._run_one(
            self._prompt(prompt_token_ids, private_kv_for=rid),
            SamplingParams(max_tokens=1, temperature=0.0),
            request_id=rid,
        )
        resid = read_resid_post_from_output(out, self._layer_ids)
        if layers is not None:
            wanted = {int(x) for x in layers}
            resid = {k: v for k, v in resid.items() if k in wanted}
        return resid

    async def capture(
        self,
        prompt_token_ids: Sequence[int],
        points: Sequence[Address | str | tuple[str, int]],
        *,
        steering_spec: Any = None,
        detach: bool = True,
        rows: Sequence[int] | None = None,
    ) -> dict[Address, torch.Tensor]:
        """Async per-request worker-hook capture for a single prompt (concurrency-safe).

        Registers the points under a unique ``request_id``, runs that request, then
        collects only that request's rows (see the per-request demux in
        ``vllm_capture.requests``). Safe to call concurrently with other requests. When
        ``steering_spec`` (a ``SteeringSpec`` or a list of them) is given, the SAME request also
        steers, so the captured activations are post-cap (persona assistant-axis).

        With ``rows``, a static worker checks the full row count and then keeps only those
        rows, so a short capture of a long prompt costs one row per point to send back. Hooked
        workers send every row, and the rows are picked here.

        ``detach=False`` always raises here: the returned tensors are rebuilt from bytes on this
        side of the process boundary, so no graph reaches back into the worker's forward. They are
        ordinary tensors, though, so a caller can build their own graph on top -- which is the
        ``downstream`` half of :attr:`grad_support`.
        """
        if not detach:
            self.grad_support.require_through_forward()
        pts = _validate_hook_points(points, self._basis_if_loaded(), self.served_points)
        self._require_capture_points(pts, "Activation capture")
        n_prompt = len(prompt_token_ids)
        picked = checked_rows(rows, n_prompt)
        payload, static_cap = await self._captured_forward(prompt_token_ids, pts, steering_spec, picked)
        out = decode_capture_payload(payload)
        _assert_points_captured(out, pts)
        if picked is not None and static_cap:
            wrong = {str(pt): int(t.shape[0]) for pt, t in out.items() if int(t.shape[0]) != len(picked)}
            if wrong:
                raise RuntimeError(f"vLLM capture returned {wrong} rows for {len(picked)} requested rows.")
        else:
            _assert_full_prompt_captured(out, n_prompt)
            if picked is not None:
                out = {pt: t[picked] for pt, t in out.items()}
        _assert_full_width_captured(out, self._hidden_size)
        return out

    async def project(
        self,
        prompt_token_ids: Sequence[int],
        directions: Sequence[DirectionSet],
        *,
        steering_spec: Any = None,
    ) -> list[torch.Tensor]:
        """Per set, ``[n_prompt, k]`` float32: the worker projects its rows, so only the values cross.

        One forward with every set's point registered, as :meth:`capture` runs it, and one collect
        that projects on the worker (``vllm_capture.project``).
        """
        if not self._projects_on_worker:
            return await project_by_capture(self, prompt_token_ids, directions, steering_spec)
        addresses = check_directions(directions)
        pts = _validate_hook_points(list(dict.fromkeys(addresses)), self._basis_if_loaded(), self.served_points)
        self._require_capture_points(pts, "Projection")
        wire = [to_wire(s) for s in directions]
        payload, _ = await self._captured_forward(prompt_token_ids, pts, steering_spec, None, sets=wire)
        n_prompt = len(prompt_token_ids)
        out = []
        for i, address in enumerate(addresses):
            if str(i) not in payload:
                raise RuntimeError(f"vLLM captured no rows at {address} for DirectionSet {i}.")
            values = decode_tensor_payload(payload[str(i)])
            if int(values.shape[0]) != n_prompt:
                raise RuntimeError(
                    f"vLLM projected {int(values.shape[0])} rows at {address} for a {n_prompt}-token prompt."
                )
            out.append(values)
        return out

    async def _captured_forward(
        self,
        prompt_token_ids: Sequence[int],
        pts: list[str],
        steering_spec: Any,
        picked: list[int] | None,
        *,
        sets: list[dict] | None = None,
    ) -> tuple[Any, bool]:
        """One prompt's forward with ``pts`` registered; the rank-0 collect, and whether it was static.

        With ``sets`` (``directions.to_wire``), the collect is the worker's projection of the rows.
        """
        from vllm import SamplingParams  # pyright: ignore[reportMissingImports]

        from interp_engine.steer import steering_scope_for_call

        n_prompt = len(prompt_token_ids)
        await self._ensure_engine()
        steering = steering_scope_for_call(self, steering_spec, what="a capture")
        rid = self._new_request_id("np-cap")
        static_cap = self._use_static_capture()
        # The write first: its refusals all fire before any RPC, so nothing is left registered.
        registered = await self._register_write(rid, steering, prompt_token_ids, what="Steered capture")
        if static_cap:
            await self.engine.collective_rpc("register_static_capture", args=(rid, pts, picked, n_prompt))
        else:
            await self.engine.collective_rpc("register_capture", args=(rid, pts))
        try:
            await self._run_one(
                self._prompt(prompt_token_ids, private_kv_for=rid),
                SamplingParams(max_tokens=1, temperature=0.0),
                request_id=rid,
            )
        finally:

            def collect() -> Awaitable[Any]:
                if sets is not None:
                    return self.engine.collective_rpc("collect_projected", args=(rid, sets, static_cap))
                return self.engine.collective_rpc("collect_static" if static_cap else "collect_request", args=(rid,))

            payloads = await _settle(collect, lambda: self._unregister_write(rid, registered))
        return (payloads[0] if isinstance(payloads, list | tuple) else payloads), static_cap

    async def capture_generation(
        self,
        prompt_token_ids: Sequence[int],
        points: Sequence[Address | str | tuple[str, int]],
        *,
        max_tokens: int = 8,
        temperature: float = 0.0,
        seed: int | None = None,
        steering_spec: Any = None,
        lens_intervention: dict | None = None,
    ) -> tuple[Any, dict[Address, torch.Tensor]]:
        """Generate + capture ``points`` at prompt AND generated positions (decode-time).

        Returns ``(completion_output, {(name, layer): [prompt+generated, width]})``. The
        captured length is ``prompt_len + generated_len - 1``: the final sampled token is
        never fed back through the model, which is universal autoregressive behavior rather
        than a vLLM quirk.

        ``steering_spec``, or the open ``steer()`` block when none is passed, is applied during
        generation so captured residuals reflect the intervention; the block's ``position_mask``
        and ``generated`` are honoured. A lens intervention is a block of ``NormScaledAddSpec`` /
        ``AblateSpec`` / ``SwapSpec`` ops like any other. ``lens_intervention`` is the deprecated dict
        form of such a block. Single request-locked use.
        """
        from vllm import SamplingParams  # pyright: ignore[reportMissingImports]

        from interp_engine.steer import steering_scope_for_call

        pts = _validate_hook_points(points, self._basis_if_loaded(), self.served_points)
        self._require_capture_points(pts, "Capture during generation")
        await self._ensure_engine()
        steering = _legacy_lens_steering(
            lens_intervention, steering_spec, "capture_generation"
        ) or steering_scope_for_call(self, steering_spec, what="a generation capture")
        rid = self._new_request_id("np-capgen")
        static_cap = self._use_static_capture()
        registered = await self._register_write(rid, steering, prompt_token_ids, what="Steered generation")
        if static_cap:
            await self.engine.collective_rpc("register_static_capture", args=(rid, pts))
        else:
            await self.engine.collective_rpc("register_capture", args=(rid, pts))
        try:
            out = await self._run_one(
                self._prompt(prompt_token_ids, private_kv_for=rid),
                SamplingParams(max_tokens=max_tokens, temperature=temperature, seed=seed),
                request_id=rid,
            )
        finally:
            payloads = await _settle(
                lambda: self.engine.collective_rpc("collect_static" if static_cap else "collect_request", args=(rid,)),
                lambda: self._unregister_write(rid, registered),
            )
        caps = decode_capture_payload(payloads[0] if isinstance(payloads, list | tuple) else payloads)
        _assert_points_captured(caps, pts)
        _assert_full_width_captured(caps, self._hidden_size)
        return out.outputs[0], caps

    async def capture_generation_stream(
        self,
        prompt_token_ids: Sequence[int],
        points: Sequence[Address | str | tuple[str, int]],
        *,
        max_tokens: int = 8,
        temperature: float = 0.0,
        seed: int | None = None,
        steering_spec: Any = None,
        lens_intervention: dict | None = None,
    ):
        """Streaming :meth:`capture_generation`: yield ``(new_captures, token_ids)`` per step.

        ``new_captures`` holds the rows captured since the previous yield (keyed like
        :meth:`capture`; the prefill's prompt rows arrive in the first non-empty one) and
        ``token_ids`` is the generated ids so far. A consumer can therefore read out each
        position as the engine produces it instead of waiting for the whole generation --
        which is what makes lens read-outs stream token-by-token.

        A yield carries whatever is NEW, which may be rows, or ids, or both: an empty
        ``new_captures`` is normal and means the engine sampled a token whose rows an earlier
        drain already took. Consumers must handle it, because the ids are load-bearing on
        their own -- a position needs both its rows and its id, and this is the only place the
        ids come from. See :meth:`lens_capture_readout_stream` for what withholding them cost.

        The engine is never blocked on the consumer: it keeps generating and the hooks keep
        accumulating, so falling behind costs nothing but latency. ``steering_spec`` and the
        open ``steer()`` block behave exactly as in :meth:`capture_generation`, and so does the deprecated
        ``lens_intervention``.
        """
        from vllm import SamplingParams  # pyright: ignore[reportMissingImports]

        from interp_engine.steer import steering_scope_for_call

        pts = _validate_hook_points(points, self._basis_if_loaded(), self.served_points)
        self._require_capture_points(pts, "Streaming capture during generation")
        await self._ensure_engine()
        steering = _legacy_lens_steering(
            lens_intervention, steering_spec, "capture_generation_stream"
        ) or steering_scope_for_call(self, steering_spec, what="a streaming generation capture")
        rid = self._new_request_id("np-capgen")
        static_cap = self._use_static_capture()
        registered = await self._register_write(rid, steering, prompt_token_ids, what="Steered generation")
        if static_cap:
            await self.engine.collective_rpc("register_static_capture", args=(rid, pts))
        else:
            await self.engine.collective_rpc("register_capture", args=(rid, pts))

        sp = SamplingParams(max_tokens=max_tokens, temperature=temperature, seed=seed)
        prompt = self._prompt(prompt_token_ids, private_kv_for=rid)
        token_ids: list[int] = []
        tail: dict[Address, torch.Tensor] = {}
        completed = False
        # How many ids the consumer has been told about, so a step that drained no rows can
        # still report the token it sampled.
        reported = 0
        # Which points have produced rows at some point in the stream. A drain covers only the
        # rows since the last one, so no single payload is expected to hold every point -- the
        # union across the whole run is what can be checked, and only once it finishes.
        seen: set[Address] = set()
        try:
            async for out in self.engine.generate(prompt, sp, rid):
                token_ids = [int(t) for t in out.outputs[0].token_ids]
                if static_cap:
                    payloads = await self.engine.collective_rpc("drain_static", args=(rid,))
                else:
                    payloads = await self.engine.collective_rpc("drain_request", args=(rid,))
                drained = decode_capture_payload(payloads[0] if isinstance(payloads, list | tuple) else payloads)
                if drained or len(token_ids) > reported:
                    seen.update(drained)
                    reported = len(token_ids)
                    yield drained, token_ids
            completed = True
        finally:
            # Deregister on every exit path (including client disconnect), and keep whatever
            # the last forward appended after the final drain so no position is dropped.
            payloads = await _settle(
                lambda: self.engine.collective_rpc("collect_static" if static_cap else "collect_request", args=(rid,)),
                lambda: self._unregister_write(rid, registered),
            )
            tail = decode_capture_payload(payloads[0] if isinstance(payloads, list | tuple) else payloads)
        if completed and (tail or len(token_ids) > reported):
            seen.update(tail)
            yield tail, token_ids
        if completed:
            # Raising after the last yield surfaces on the consumer's final step, which is the
            # earliest the union above is known. A stream cut short by a disconnect is exempt:
            # it is allowed to be partial, and `completed` is what distinguishes the two.
            _assert_points_captured(seen, pts)

    async def lens_capture_readout_stream(
        self,
        prompt_token_ids: Sequence[int],
        points: Sequence[Address | str | tuple[str, int]],
        specs: list[dict],
        *,
        top_n: int,
        softcap: float | None = None,
        word_mask: torch.Tensor | None = None,
        chunk_positions: int = 8,
        skip_before: int = 0,
        max_tokens: int = 1,
        temperature: float = 0.0,
        seed: int | None = None,
        steering_spec: Any = None,
        lens_intervention: dict | None = None,
        stream_reduce: str = "none",
        stream_index: int | None = None,
    ):
        """Stream lens read-outs, yielding ``(first_position, top_idx, top_probs, token_ids)``.

        The fused form of :meth:`capture_generation_stream` + :meth:`decode_residuals_topk`: the
        captured rows are transported through the resident ``J_bar`` and unembedded in the worker,
        so only top-k crosses ``collective_rpc``. Driving the two separately sends the residuals
        out and straight back -- ~63 MB each way for a 96-position 64-layer read-out, which is
        where that path spent most of its time. Call :meth:`set_lens_jacobians` first; without it
        every layer reads out untransported.

        ``specs`` is one ``{"layers": [...], "jacobian": bool}`` per lens type. Each yield carries
        ``top_idx``/``top_probs`` lists holding one ``[n_new_positions * n_layers, top_n]`` tensor
        per spec, position-major, covering positions ``first_position`` onward.

        Positions are read out as the engine produces them, exactly as the capture stream hands
        rows over, so a consumer still emits token-by-token. ``token_ids`` is the generated ids so
        far; pairing them with positions is the caller's job, since generation runs ahead.
        ``skip_before`` drops leading positions unread for a caller replaying a prompt whose
        read-out it already holds.

        ``stream_reduce`` collapses a hyper-connection trunk's stream stack to the one ``d_model``
        vector the lens was fitted on -- ``points`` is ``resid_streams`` there, and its rows carry an
        extra axis the transport and the unembed have no place for. Required on such a trunk and
        refused on a conventional one, by :meth:`ResidualBasis.require_stream_reduction`, because a
        mismatch either way produces a believable shape rather than an error. Which reduction is
        correct is the lens's property, not the model's, so it has to be passed in.

        A yield may carry new ids and NO positions (``top_idx``/``top_probs`` present but with
        zero rows), which is what makes that pairing terminate. The engine outruns the read-out
        RPCs, so one call routinely takes the positions for several sampled tokens and every
        later call comes back empty; withholding those yields left the caller's last-known id
        list short of the positions it was already holding, and it dropped them -- a lens run
        asking for 5 tokens returned 3, and the count moved with how far the engine ran ahead.

        ``steering_spec`` and the open ``steer()`` block behave exactly as in
        :meth:`capture_generation`; a lens intervention is such a block. ``lens_intervention`` is its
        deprecated dict form.
        """
        from vllm import SamplingParams  # pyright: ignore[reportMissingImports]

        from interp_engine.steer import steering_scope_for_call

        pts = _validate_hook_points(points, self._basis_if_loaded(), self.served_points)
        self._require_capture_points(pts, "Streaming lens read-out")
        # The specs name layers; the worker rebuilds each capture key from this point name, so every
        # requested point has to be the same one (a lens reads one stream per layer).
        names = {to_address(p).name for p in pts}
        if len(names) != 1:
            raise ValueError(f"A lens read-out reads one point across layers; got {sorted(names)}")
        point = names.pop()
        # Gated before the request is registered and before the engine runs, so a mismatch costs no
        # forward and surfaces in the caller's own frame rather than out of a worker RPC.
        self.residual_basis.require_stream_reduction(stream_reduce, stream_index, point=point)
        await self._ensure_engine()
        steering = _legacy_lens_steering(
            lens_intervention, steering_spec, "lens_capture_readout_stream"
        ) or steering_scope_for_call(self, steering_spec, what="a lens read-out")
        rid = self._new_request_id("np-lensread")
        static_cap = self._use_static_capture()
        registered = await self._register_write(rid, steering, prompt_token_ids, what="A lens intervention")
        if static_cap:
            await self.engine.collective_rpc("register_static_capture", args=(rid, pts))
        else:
            await self.engine.collective_rpc("register_capture", args=(rid, pts))

        # Encoded once: the mask is a process-lifetime constant, and it rode along on every
        # read-out call when the client drove the chunking.
        mask_payload = encode_tensor_payload(word_mask.detach().to(torch.bool)) if word_mask is not None else None
        readout_spec = {
            "types": specs,
            "top_n": int(top_n),
            "softcap": softcap,
            "chunk_positions": int(chunk_positions),
            "point": point,
            "stream_reduce": stream_reduce,
            "stream_index": stream_index,
            # Only where the capture is stacked, so the worker's shape assertion is the trunk's
            # stream count exactly where that is what the axis holds, and absent where it is not.
            "n_streams": self.residual_basis.n_streams if self.residual_basis.stacked_at(point) else None,
            "skip_before": int(skip_before),
        }

        captured_rows = 0

        async def readout(final: bool) -> tuple[int, int, list[torch.Tensor], list[torch.Tensor]]:
            nonlocal captured_rows
            results = await self.engine.collective_rpc(
                "lens_capture_readout",
                args=(rid, readout_spec, mask_payload, final),
            )
            out = results[0] if isinstance(results, list | tuple) else results
            captured_rows += int(out["n_rows"])
            n = int(out["n_positions"])
            if n <= 0:
                # Zero-row tensors rather than empty lists: a yield can carry no positions (it
                # exists to report the ids), and a caller that reads `top_idx[0].shape[0]` to
                # count them should get 0 rather than an IndexError.
                return (
                    int(out["first_position"]),
                    0,
                    [torch.empty((0, int(top_n)), dtype=torch.int64) for _ in specs],
                    [torch.empty((0, int(top_n)), dtype=torch.float32) for _ in specs],
                )
            idx = [decode_tensor_payload(r["top_idx"]) for r in out["results"]]
            probs = [decode_tensor_payload(r["top_probs"]) for r in out["results"]]
            return int(out["first_position"]), n, idx, probs

        sp = SamplingParams(max_tokens=max(1, int(max_tokens)), temperature=temperature, seed=seed)
        prompt = self._prompt(prompt_token_ids, private_kv_for=rid)
        token_ids: list[int] = []
        completed = False
        # How many ids the consumer has been told about (see the note on empty yields above).
        reported = 0
        tail: tuple[int, int, list[torch.Tensor], list[torch.Tensor]] = (0, 0, [], [])
        try:
            async for out in self.engine.generate(prompt, sp, rid):
                token_ids = [int(t) for t in out.outputs[0].token_ids]
                first, n, idx, probs = await readout(final=False)
                if n or len(token_ids) > reported:
                    reported = len(token_ids)
                    yield first, idx, probs, token_ids
            completed = True
        finally:
            # Deregister on every exit path (including client disconnect), and read out whatever
            # the last forward appended after the final drain so no position is dropped.
            tail = await _settle(lambda: readout(final=True), lambda: self._unregister_write(rid, registered))
        if completed and (tail[1] or len(token_ids) > reported):
            yield tail[0], tail[2], tail[3], token_ids
        if completed and captured_rows == 0:
            raise RuntimeError(
                f"Lens read-out captured no positions for {sorted(pts)}. The worker hooks did not "
                "fire -- see capture_engine_kwargs (CUDA graphs and prefix caching both bypass them). "
                "This is the fused read-out's equivalent of the capture stream's empty-point check."
            )

    async def set_lens_intervention(
        self,
        specs: list[dict],
        steer_generated: bool,
        skip_positions: list[int],
        prompt_len: int,
    ) -> None:
        """Install GLOBAL jlens write-hooks (single-request; validation only).

        The server path opens a ``steer()`` block of ``NormScaledAddSpec`` / ``AblateSpec`` /
        ``SwapSpec`` ops around its capture, registered per request; this global variant keeps
        the lens wire format for the sync/validation scripts. Cleared by :meth:`clear_steering`.

        Pinned to the decoder layer's output, so a spec that names a point or a stream is refused
        rather than ignored: the per-request path honours both and this one has no way to, and a caller
        who cannot tell which of the two ran would have to discover that from the outputs.
        """
        aimed = sorted(
            {str(s["point"]) for s in specs if s.get("point") not in (None, "resid_post")}
            | {f"stream={s['stream']}" for s in specs if s.get("stream") is not None}
        )
        if aimed:
            raise ValueError(
                f"set_lens_intervention writes the decoder layer's output and cannot aim at {aimed}. "
                "Open a steer() block of SteeringSpec(point=..., stream=...) ops around the capture "
                "instead -- the per-request path installs a hook per site and honours the point (and "
                "stream) the spec names."
            )
        from interp_engine.vllm_capture.lens import lens_wire_to_steer_spec

        # The gate before the rename, so a backend that cannot write refuses as such.
        if not self._use_static_writes():
            self._require_hooks("A lens intervention")
        filled = [lens_wire_to_steer_spec({**s, "point": str(s.get("point") or "resid_post")}) for s in specs]
        if self._use_static_writes():
            self._require_static_writes(filled, "A lens intervention")
            await self._ensure_engine()
            prev = getattr(self, "_static_global_lease", None)
            if prev is not None:
                self._static_global_lease = None
                await prev.finish()
            lease = _StaticDeltaLease(
                self,
                filled,
                lens_scope={
                    "steer_generated": bool(steer_generated),
                    "skip_positions": [int(i) for i in (skip_positions or [])],
                    "prompt_len": int(prompt_len),
                },
            )
            await lease.start()
            self._static_global_lease = lease
            self._global_intervention = self._new_request_id("np-global-lens")
            return
        await self._ensure_engine()
        await self.engine.collective_rpc(
            "install_lens_intervention",
            args=(filled, steer_generated, skip_positions, prompt_len),
        )
        self._global_intervention = self._new_request_id("np-global-lens")

    async def decode_residuals(self, residuals: torch.Tensor, *, detach: bool = True) -> torch.Tensor:
        """Decode ``[n_rows, d_model]`` residuals -> ``[n_rows, vocab]`` logits.

        Reuses vLLM's own final norm + lm_head via the uniform ``compute_logits`` (no
        per-arch code, no extra weights). The logits come back with the model's configured
        final-logit softcap ALREADY applied, because vLLM applies it inside
        ``compute_logits``; do not apply it again. This is where this backend differs from
        :func:`interp_engine.lens.decode_residuals`, which returns raw logits.

        ``detach=False`` raises: the unembed runs in a worker process and the logits are rebuilt from
        bytes here, so there is no graph to connect back to the residuals you passed in.
        """
        if not detach:
            self.grad_support.require_through_forward()
        await self._ensure_engine()
        payload = encode_tensor_payload(residuals)
        results = await self.engine.collective_rpc("unembed", args=(payload,))
        return decode_tensor_payload(results[0] if isinstance(results, list | tuple) else results)

    async def decode_residuals_topk(
        self,
        residuals: torch.Tensor,
        *,
        top_n: int,
        softcap: float | None = None,
        word_mask: torch.Tensor | None = None,
        rows_per_group: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode residuals -> top-k ids/probs on the worker (see :func:`worker_lens_readout`).

        Returns ``(top_idx, top_probs)`` each ``[n_rows, top_n]``. Prefer this over
        :meth:`decode_residuals` for lens serving: it avoids shipping vocab-sized logits
        back over ``collective_rpc``.
        """
        await self._ensure_engine()
        payload = encode_tensor_payload(residuals)
        mask_payload = encode_tensor_payload(word_mask.detach().to(torch.bool)) if word_mask is not None else None
        results = await self.engine.collective_rpc(
            "lens_readout",
            args=(payload, int(top_n), softcap, mask_payload, int(rows_per_group)),
        )
        out = results[0] if isinstance(results, list | tuple) else results
        return decode_tensor_payload(out["top_idx"]), decode_tensor_payload(out["top_probs"])

    async def generate_with_lens(
        self,
        prompt_token_ids: Sequence[int],
        lenses: Sequence[LensSpec],
        *,
        point: str = "resid_post",
        top_n: int = 10,
        max_tokens: int = 0,
        temperature: float = 0.0,
        seed: int | None = None,
        word_mask: torch.Tensor | None = None,
        skip_before: int = 0,
        stream_reduce: str = "none",
        stream_index: int | None = None,
        jacobians: Mapping[int, torch.Tensor] | None = None,
        softcap: float | None = None,
        steering_spec: Any = None,
    ) -> AsyncIterator[LensStep]:
        """The lens at every position, streamed token by token. See the protocol.

        Read out in the worker, so only top-k crosses ``collective_rpc``. Given ``jacobians`` for a
        Jacobian lens, the rows come here instead, to be carried through them, and go back for the
        unembed: the path for a lens too large to sit on the worker.
        """
        from interp_engine import lens_stream

        here = jacobians is not None and any(s.jacobian for s in lenses)
        if here and any(s.jacobian and s.jacobian_set != lens_stream.DEFAULT_JACOBIAN_SET for s in lenses):
            raise ValueError(
                "jacobians= carries the default set to this side, but a named set stays on the "
                "worker: read named sets in a call that passes no jacobians"
            )
        held_sets = (
            ([lens_stream.DEFAULT_JACOBIAN_SET] if jacobians else [])
            if here
            else [name for name, held in self._lens_jacobian_sets.items() if held]
        )
        layers = lens_stream.prepare(
            self,
            lenses,
            top_n=top_n,
            point=point,
            stream_reduce=stream_reduce,
            stream_index=stream_index,
            jacobian_sets=held_sets,
        )
        kw: dict[str, Any] = {
            "point": point,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "skip_before": min(max(int(skip_before), 0), len(prompt_token_ids)),
            "stream_reduce": stream_reduce,
            "stream_index": stream_index,
            "steering_spec": steering_spec,
        }
        if not here:
            async for step in lens_stream.fused_steps(
                self, prompt_token_ids, lenses, layers, top_n=top_n, word_mask=word_mask, softcap=softcap, **kw
            ):
                yield step
            return
        assert jacobians is not None
        dtype = next(iter(jacobians.values())).dtype if jacobians else None
        topk = lens_stream.worker_topk(self, top_n=top_n, softcap=softcap, word_mask=word_mask, dtype=dtype)
        source = lens_stream.protocol_rows(self, prompt_token_ids, layers, **kw)
        async for step in lens_stream.read_out(
            source,
            lenses,
            prompt_len=len(prompt_token_ids),
            jacobians={lens_stream.DEFAULT_JACOBIAN_SET: jacobians},
            topk=topk,
        ):
            yield step

    async def set_lens_jacobians(self, jacobians: Mapping[int, torch.Tensor] | None, *, name: str = "default") -> int:
        """Make the Jacobian-lens set ``name`` resident on the worker(s). Returns bytes per rank.

        Hand this the whole lens once at startup and the read-out never has to ship residuals:
        :meth:`lens_capture_readout` transports and unembeds where the rows were captured. Pass
        ``None`` to release that set; other sets stay. Under tensor parallelism every rank holds a
        full copy (``J_bar`` is not sharded), so the return value is PER RANK, not in total.
        """
        await self._ensure_engine()
        payloads = (
            None
            if jacobians is None
            else {str(int(layer)): encode_tensor_payload(matrix) for layer, matrix in jacobians.items()}
        )
        results = await self.engine.collective_rpc("set_lens_jacobians", args=(payloads, name))
        out = results[0] if isinstance(results, list | tuple) else results
        if jacobians is None:
            self._lens_jacobian_sets.pop(name, None)
        else:
            self._lens_jacobian_sets[name] = frozenset(int(layer) for layer in jacobians)
        return int(out["bytes"])

    async def lens_transport(
        self, rows: torch.Tensor, layers: Sequence[int], *, jacobian_set: str = "default"
    ) -> tuple[torch.Tensor, list[bool]]:
        """Pull ``[k, d_model]`` rows back through each layer's resident ``J_bar``: ``rows @ J_bar``.

        Returns ``([n_layers, k, d_model]`` float32, ``per-layer transported flags)``. Layers with
        no fitted ``J_bar`` come back unchanged. See :func:`worker_lens_transport`, including why
        this is the transpose of what the read-out applies: this is the steering direction, and
        it exists as an RPC because the lens lives on the worker.
        """
        await self._ensure_engine()
        payload = encode_tensor_payload(rows)
        results = await self.engine.collective_rpc(
            "lens_transport", args=(payload, [int(x) for x in layers], jacobian_set)
        )
        out = results[0] if isinstance(results, list | tuple) else results
        return decode_tensor_payload(out["rows"]), [bool(flag) for flag in out["transported"]]

    async def unembed_rows(self, token_ids: Sequence[int]) -> torch.Tensor:
        """Return ``W_U[token_ids]`` ([k, d_model]) -- unembedding directions for jlens steering.

        ``W_U`` is ``lm_head.weight`` when present; on tied-embedding models such as
        Gemma 2 (no separate ``lm_head`` in vLLM) it is ``model.embed_tokens.weight``.

        Under tensor parallelism the head is vocab-sharded, so each worker returns only
        the rows it owns and this method merges them. That is what makes lens
        steer/ablate/swap work on multi-GPU pods (e.g. Llama 3.3 70B at TP=4).
        """
        from interp_engine.vllm_capture import merge_lm_head_row_payloads

        ids = [int(t) for t in token_ids]
        await self._ensure_engine()
        results = await self.engine.collective_rpc("lm_head_rows", args=(ids,))
        if not isinstance(results, list | tuple):
            results = [results]
        return merge_lm_head_row_payloads(ids, list(results))

    async def capture_attention(
        self, prompt_token_ids: Sequence[int], layers: Sequence[int]
    ) -> dict[int, dict[str, torch.Tensor]]:
        """Async attention probs + value per layer via off-kernel recompute (see sync variant)."""
        from vllm import SamplingParams  # pyright: ignore[reportMissingImports]

        from interp_engine.vllm_capture.static import ATTN_STATIC_POINT

        layers = [int(x) for x in layers]
        static_attn = self._use_static_attn(layers)
        if not static_attn:
            self._require_hooks("Attention capture")
        await self._ensure_engine()
        rid = self._new_request_id("np-attn")
        needed = attn_capture_layers(self._attn_dims, layers)
        # Asked before anything is registered: a layer with no attention op to read q/k off
        # (multi-head latent attention) would raise inside the worker, and under tensor parallelism
        # that raise leaves the other ranks' replies queued for the collect below to misread.
        verdict = (await self.engine.collective_rpc("resolvable_attn", args=(needed,)))[0]
        absent = {int(layer): why for layer, why in verdict.items() if why}
        if absent:
            layer, why = min(absent.items())
            raise ValueError(
                f"Attention capture cannot serve attn_scores at layer {layer} on the vLLM backend: {why}. "
                "Instead: capture_attention on an eager model, which forms the softmax explicitly."
            )
        if static_attn:
            pts = [format_address(Address(ATTN_STATIC_POINT, layer)) for layer in needed]
            await self.engine.collective_rpc("register_static_capture", args=(rid, pts))
        else:
            await self.engine.collective_rpc("register_attn", args=(rid, needed))
        try:
            await self._run_one(
                self._prompt(prompt_token_ids, private_kv_for=rid),
                SamplingParams(max_tokens=1, temperature=0.0),
                request_id=rid,
            )
        finally:
            payloads = await asyncio.shield(
                self.engine.collective_rpc("collect_static" if static_attn else "collect_attn_request", args=(rid,))
            )
        return recompute_attn_from_payloads(payloads, layers, self._attn_dims, self.tensor_parallel_size)

    async def set_steering(self, specs: list[dict]) -> None:
        """Install additive steering write-hooks on all workers (single-request use).

        ``specs`` items: ``{"layer", "point" ("resid_post"|"resid_pre"|"z"), "vector"
        (list[float]), "coeff"}``. Clear with :meth:`clear_steering`.
        """
        _validate_steer_points(specs, self._basis_if_loaded())
        self._require_static_writes(specs, "Steering")
        await self._ensure_engine()
        if self._use_static_writes():
            prev = getattr(self, "_static_global_lease", None)
            if prev is not None:
                self._static_global_lease = None
                await prev.finish()
            lease = _StaticDeltaLease(self, specs)
            await lease.start()
            self._static_global_lease = lease
        else:
            await self.engine.collective_rpc("install_steering", args=(specs,))
        self._global_intervention = self._new_request_id("np-global-steer")

    async def clear_steering(self) -> None:
        await self._ensure_engine()
        lease = getattr(self, "_static_global_lease", None)
        if lease is not None:
            self._static_global_lease = None
            await lease.finish()
        else:
            await self.engine.collective_rpc("clear_steering")
        # Requests from here on compute un-intervened KV again, so they go back to sharing the
        # cache with each other -- and the salt they no longer carry is what keeps them from
        # picking up blocks the intervention wrote.
        self._global_intervention = None
