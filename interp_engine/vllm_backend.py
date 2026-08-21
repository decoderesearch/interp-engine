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
from collections.abc import Iterable, Sequence
from typing import Any

import torch

from interp_engine import facts
from interp_engine._loop import refuse_foreign_loop
from interp_engine.address import Address, format_address, to_address
from interp_engine.autograd_support import GradSupport, vllm_grad_support
from interp_engine.notebook_stdout import ensure_stdout_descriptor
from interp_engine.points import d_model_wide, hyper_connection_names, refusal_reasons
from interp_engine.points import steer_refusal_reason as points_steer_refusal
from interp_engine.residual_basis import ResidualBasis, vllm_residual_basis
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
    apply_breakable_env,
    decode_only_graphs_reason,
    encode_static_env,
    estimate_weight_bytes,
    fit_max_num_batched_tokens,
    kv_cache_width,
    resid_stream_aliases,
    resolve_static_points,
    static_read_width,
    static_unsupported_reason,
)
from interp_engine.vllm_plugin import WORKER_EXTENSION_CLS

logger = logging.getLogger(__name__)

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
        # From here on a layer reuses an earlier layer's keys/values and has no v_proj to hook, so
        # `value`/DFA is unavailable there (Gemma-4). None when every layer projects its own.
        "first_kv_shared_layer": model_facts.first_kv_shared_layer,
        # The declared value-head width (MiMo-V2, DeepSeek MLA project a value unlike their q/k).
        # Equal to `head_dim` when the family declares nothing, which is why
        # `value_head_dim_for_layer` compares the two rather than testing this for truth.
        "v_head_dim": model_facts.v_head_dim,
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
        int(dims.get("n_kv_heads") or 0), layer, tuple(dims.get("per_layer_kv_heads") or ())
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
    """
    if int(tensor_parallel_size) > 1:
        # q/k/v are head-sharded across ranks, and we only read rank 0's payload, so the
        # tensors here hold 1/tp of the heads while ``dims`` describes the whole model.
        # The ``view`` below would fail on the element count anyway; raising here says why.
        raise RuntimeError(
            f"Attention recompute is not supported at tensor_parallel_size="
            f"{int(tensor_parallel_size)}: q/k/v are sharded by head across ranks, so "
            "rank 0 holds only its slice. Serve attention patterns and DFA from a "
            "single-GPU pod (or the eager backend)."
        )
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
        stated_kv = kv_heads_for_layer(dims, int(layer)) if dims.get("per_layer_kv_heads") else None
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
    points: Sequence[Address | str | tuple[str, int]], basis: ResidualBasis | None = None
) -> list[str]:
    """Check the requests against what worker hooks can serve, and return them in **wire** form.

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
    bad = sorted({a.name for a in addresses if a.name not in HOOK_CAPTURE_POINTS})
    if bad:
        # Each refused point quotes its own reason from the point table, which is the difference
        # between "nobody has implemented this yet" and "no such tensor exists in a fused engine" --
        # and is the difference between filing a bug and switching backend.
        raise ValueError(
            f"vLLM worker-hook capture cannot serve points {bad}:\n"
            f"{refusal_reasons(bad)}\n"
            f"Supported: {sorted(HOOK_CAPTURE_POINTS)}."
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
        # Independently of everything above, prefix caching keys on token IDs, which an
        # embeds prompt does NOT have; leaving it on hangs engine init (the sglang path used
        # disable_radix_cache=True for the same reason). A salt cannot help here -- there is
        # nothing to salt -- so this whole engine gives the feature up.
        kwargs["enable_prefix_caching"] = False
        # Accept EmbedsPrompt ({"prompt_embeds": [T, d]}) inputs -- powers NLA concept
        # injection (activation vector spliced into the prompt embedding sequence).
        kwargs["enable_prompt_embeds"] = True
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
        self.enable_extraction = enable_extraction
        self.enable_prompt_embeds = enable_prompt_embeds
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
            self._pin_decode_only_graphs_on_hybrid_trunk()
        n_bufs = sum(3 if a.name == "attn" else 1 for a in reads) + len(writes)
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
            weight_bytes=estimate_weight_bytes(
                self.num_hidden_layers,
                self._hidden_size,
                config=cfg,
                hf_model_id=hf_id,
            ),
            max_model_len=int(self._engine_kwargs.get("max_model_len") or max_n),
            kv_width=kv_cache_width(
                n_kv_heads=int(dims.get("n_kv_heads") or 0) if dims else 0,
                head_dim=int(dims.get("head_dim") or 0) if dims else 0,
                v_head_dim=int(dims.get("v_head_dim") or 0) if dims else 0,
                d_model=max(int(self._hidden_size), 1),
            ),
            n_layers=self.num_hidden_layers,
            tensor_parallel_size=self.tensor_parallel_size,
        )
        if fitted != max_n:
            logger.warning("lowering max_num_batched_tokens %s -> %s so static buffers fit", max_n, fitted)
            self._engine_kwargs["max_num_batched_tokens"] = fitted

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

        Additive ``add_`` and live-read ops (orthogonal, projection_cap, lens steer/ablate/swap)
        all ride the same static wrap. A site miss is a 400, not a silent no-op.
        """
        if self.hooks_available:
            return
        writes = getattr(self, "_static_writes", frozenset())
        if self.graph_replay and writes:
            for spec in specs:
                op = spec.get("op", "add")
                if op not in {"add", "orthogonal", "projection_cap", "steer", "ablate", "swap"}:
                    raise RuntimeError(
                        f"{what} op {op!r} is not one a static write tap can apply; "
                        f"backend='vllm-static' serves add, orthogonal, projection_cap, steer, "
                        f"ablate and swap. Use backend='vllm' for this one."
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

    def _lens_specs(self, lens_intervention: dict) -> list[dict]:
        """A lens intervention's specs, with the point filled in and checked like a steer's.

        A jlens intervention is a write, so it faces the same question a steer does -- can this point
        be written on this model -- and gets the same answer from the same place. The default is
        ``resid_post``, which is what jlens has always sent and what the eager path still hardcodes;
        naming an mHC collapse instead is how a swap/steer works on a hyper-connection trunk, where
        ``resid_post`` refers to nothing the caller can write.

        Filling the default in here rather than on the worker means the client-side refusal sees the
        point the worker will use, instead of passing an under-specified spec and being told about it
        from inside a forward in another process.
        """
        specs = [{**s, "point": str(s.get("point") or "resid_post")} for s in lens_intervention["specs"]]
        _validate_steer_points(specs, self.residual_basis)
        return specs

    async def _register_static_write(
        self,
        rid: str,
        specs: list[dict],
        *,
        position_mask: Any = None,
        prompt_token_ids: Sequence[int] | None = None,
        lens_scope: dict | None = None,
    ) -> None:
        """Per-request static write. ``position_mask`` becomes skip_positions on the wrap."""
        from interp_engine.steer import resolve_masked_positions

        if lens_scope is not None:
            skip = [int(i) for i in (lens_scope.get("skip_positions") or [])]
            prompt_len = int(lens_scope.get("prompt_len") or 0)
        else:
            ids = [int(t) for t in (prompt_token_ids or [])]
            skip = resolve_masked_positions(
                position_mask, prompt_token_ids=ids, tokenizer=getattr(self, "tokenizer", None)
            )
            prompt_len = len(ids)
        await self.engine.collective_rpc(
            "register_static_write",
            args=(rid, specs, skip, prompt_len, lens_scope),
        )

    async def _unregister_static_write(self, rid: str) -> None:
        await self.engine.collective_rpc("unregister_static_write", args=(rid,))

    def _lens_scope(self, lens_intervention: dict) -> dict:
        return {
            "steer_generated": bool(lens_intervention.get("steer_generated", False)),
            "skip_positions": [int(i) for i in (lens_intervention.get("skip_positions") or [])],
            "prompt_len": int(lens_intervention.get("prompt_len") or 0),
        }

    async def _register_static_lens(self, rid: str, lens_intervention: dict) -> None:
        specs = self._lens_specs(lens_intervention)
        self._require_static_writes(specs, "A lens intervention")
        await self._register_static_write(rid, specs, lens_scope=self._lens_scope(lens_intervention))

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
        stream: bool = False,
        capture_points: Sequence[Address | str | tuple[str, int]] | None = None,
        capture_out: dict[Address, torch.Tensor] | None = None,
        drain_every: int = 64,
    ):
        """VLLMSteerModel-style generation: apply an engine SteeringSpec, then generate.

        ``steering_spec`` is a ``interp_engine.SteeringSpec`` (or None). Returns
        the full text (stream=False) or an async generator of text deltas
        (stream=True). Steering is installed for the duration and cleared after.
        ``position_mask`` (``SteerMask`` preset or ``list[int]``) excludes prompt positions
        from steering; it's resolved here against the actual prompt token ids + tokenizer.
        Single request-locked use.

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
        from interp_engine.steer import resolve_masked_positions

        await self._ensure_engine()
        steered = steering_spec is not None and not steering_spec.is_empty()
        capturing = bool(capture_points) and capture_out is not None
        if capturing and not (self.hooks_available or self._use_static_capture()):
            self._require_hooks("Capture during generation")
        if steered and not (self.hooks_available or self._use_static_writes()):
            self._require_hooks("Steered generation")
        token_ids = [int(t) for t in prompt_token_ids]
        rid = self._new_request_id("np-steer")
        prompt = self._prompt(token_ids, private_kv_for=rid if (capturing or steered) else None)
        pts: list[str] = []
        worker_specs: list[dict] = []
        if capturing:
            assert capture_points is not None
            pts = _validate_hook_points(capture_points, self._basis_if_loaded())
            self._require_capture_points(pts, "Capture during generation")
        if steered:
            assert steering_spec is not None
            worker_specs = self._steer_specs(steering_spec)
            self._require_static_writes(worker_specs, "Steered generation")
        static_cap = capturing and self._use_static_capture()
        static_write = steered and self._use_static_writes()
        if static_write:
            await self._register_static_write(
                rid, worker_specs, position_mask=position_mask, prompt_token_ids=token_ids
            )
        if capturing and not static_cap:
            await self.engine.collective_rpc("register_capture", args=(rid, pts))
        elif static_cap:
            await self.engine.collective_rpc("register_static_capture", args=(rid, pts))
        if steered and not static_write:
            skip_positions = resolve_masked_positions(
                position_mask, prompt_token_ids=token_ids, tokenizer=self.tokenizer
            )
            await self.engine.collective_rpc(
                "register_steering",
                args=(rid, worker_specs, skip_positions, len(token_ids)),
            )

        async def _finish() -> None:
            if capturing and capture_out is not None:
                if static_cap:
                    payloads = await self.engine.collective_rpc("collect_static", args=(rid,))
                else:
                    payloads = await self.engine.collective_rpc("collect_request", args=(rid,))
                _merge_captures(capture_out, _decode_rank0(payloads))
                _assert_points_captured(capture_out, pts)
                _assert_full_width_captured(capture_out, self._hidden_size)
            if static_write:
                await self._unregister_static_write(rid)
            elif steered:
                await self.engine.collective_rpc("unregister_steering", args=(rid,))

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
        steering_spec: Any = None,
        position_mask: Any = None,
    ):
        """Stream this request's ``RequestOutput``s, with a per-request steer if one was given.

        The register/generate/unregister dance, in one place, so that everything wanting a steered
        stream shares it. Extracted rather than copied because the ``finally`` is the load-bearing
        part: ``unregister_steering`` has to run on every exit path including a client
        disconnecting mid-stream, and a second hand-written copy of that is a hook leak onto every
        later request waiting to happen.

        Raw ``RequestOutput``s rather than a decoded shape, because the two callers want different
        things off them -- text deltas for the SSE path, per-token ids and logprobs for
        :meth:`generate_steps`.
        """
        from interp_engine.steer import resolve_masked_positions

        await self._ensure_engine()
        steered = steering_spec is not None and not steering_spec.is_empty()
        if steered and not (self.hooks_available or self._use_static_writes()):
            self._require_hooks("Steered generation")
        token_ids = [int(t) for t in prompt_token_ids]
        rid = self._new_request_id("np-steer" if steered else "np-steps")
        prompt = self._prompt(token_ids, private_kv_for=rid if steered else None)
        worker_specs: list[dict] = []
        static_write = False
        if steered:
            assert steering_spec is not None
            worker_specs = self._steer_specs(steering_spec)
            self._require_static_writes(worker_specs, "Steered generation")
            if self._use_static_writes():
                await self._register_static_write(
                    rid, worker_specs, position_mask=position_mask, prompt_token_ids=token_ids
                )
                static_write = True
            else:
                skip_positions = resolve_masked_positions(
                    position_mask, prompt_token_ids=token_ids, tokenizer=self.tokenizer
                )
                await self.engine.collective_rpc(
                    "register_steering",
                    args=(rid, worker_specs, skip_positions, len(token_ids)),
                )
        try:
            async for out in self.engine.generate(prompt, sampling_params, rid):
                yield out
        finally:
            if static_write:
                await self._unregister_static_write(rid)
            elif steered:
                await self.engine.collective_rpc("unregister_steering", args=(rid,))

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
        temperature: float = 1.0,
        seed: int | None = None,
        logprobs: int | None = None,
    ):
        """Non-streaming generate; returns the vLLM ``CompletionOutput``.

        The result exposes ``.text``, ``.token_ids``, ``.logprobs`` (when
        ``logprobs`` requested), and ``.finish_reason`` for the server adapter.
        """
        from vllm import SamplingParams  # pyright: ignore[reportMissingImports]

        out = await self._run_one(
            self._prompt(prompt_token_ids),
            SamplingParams(max_tokens=max_tokens, temperature=temperature, seed=seed, logprobs=logprobs),
        )
        return out.outputs[0]

    async def generate_text(
        self,
        prompt_token_ids: Sequence[int],
        *,
        max_tokens: int = 200,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> str:
        out = await self.generate_full(prompt_token_ids, max_tokens=max_tokens, temperature=temperature, seed=seed)
        return out.text

    async def generate_from_embeds(
        self,
        prompt_embeds: torch.Tensor,
        sampling_params: Any,
        *,
        request_id: str | None = None,
        stream: bool = False,
    ) -> Any:
        """Generate from a prompt EMBEDDING sequence instead of token ids.

        ``prompt_embeds`` is ``[num_tokens, hidden]`` (per-token input embeddings) in the
        model dtype -- the vLLM ``EmbedsPrompt`` input, requires ``enable_prompt_embeds``.
        Powers NLA concept injection (an activation vector spliced into the prompt
        embeddings). Returns the final vLLM ``RequestOutput`` (``stream=False``) or an
        async generator of ``RequestOutput`` (``stream=True``); ``.outputs[0]`` exposes
        ``.text`` / ``.token_ids`` / ``.finish_reason``.

        Annotated ``Any`` because the return type is selected by ``stream`` and
        ``RequestOutput`` is not importable here (vLLM is an optional dependency), so the
        inferred union would otherwise make ``.outputs`` an error for every caller.
        """
        await self._ensure_engine()
        prompt = {"prompt_embeds": prompt_embeds}
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

    async def generate_stream(
        self,
        prompt_token_ids: Sequence[int],
        *,
        max_tokens: int = 200,
        temperature: float = 1.0,
        seed: int | None = None,
    ):
        """Yield decoded text deltas as generation proceeds (for SSE endpoints)."""
        import uuid

        from vllm import SamplingParams  # pyright: ignore[reportMissingImports]

        await self._ensure_engine()
        sp = SamplingParams(max_tokens=max_tokens, temperature=temperature, seed=seed)
        prompt = self._prompt(prompt_token_ids)
        prev = ""
        async for out in self.engine.generate(prompt, sp, f"np-{uuid.uuid4().hex}"):
            text = out.outputs[0].text
            if len(text) > len(prev):
                yield text[len(prev) :]
                prev = text

    #: vLLM's own default cap on how many logprobs a request may ask for. An engine built with a
    #: different ``max_logprobs`` overrides it; this is the value to compare against when the
    #: engine was built with the default, which is every engine this backend builds.
    DEFAULT_MAX_LOGPROBS = 20

    async def generate_steps(
        self,
        prompt_token_ids: Sequence[int],
        *,
        max_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        stop_at_eos: bool = True,
        n_logprobs: int = 0,
        seed: int | None = None,
        steering_spec: Any = None,
        position_mask: Any = None,
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
        over the whole forward. That is why the free ``steer()`` context routes here.
        """
        from vllm import SamplingParams  # pyright: ignore[reportMissingImports]

        from interp_engine.steer import GenStep

        if n_logprobs > self.DEFAULT_MAX_LOGPROBS:
            raise ValueError(
                f"n_logprobs={n_logprobs} exceeds this engine's max_logprobs "
                f"({self.DEFAULT_MAX_LOGPROBS}). vLLM would reject the request rather than "
                "truncate, so it is refused here where the number can be named. Ask for at most "
                f"{self.DEFAULT_MAX_LOGPROBS}, or build the engine with a higher max_logprobs via "
                "extra_vllm_kwargs."
            )
        sampling = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            # vLLM spells "no top-k / no top-p filtering" as -1 and 1.0, where the engine's own
            # `None` means "inherit a default"; the free function's `None` means "do not filter".
            top_k=-1 if top_k is None else int(top_k),
            top_p=1.0 if top_p is None else float(top_p),
            ignore_eos=not stop_at_eos,
            logprobs=n_logprobs or None,
            seed=seed,
        )

        emitted = 0
        async for out in self._generate_request_outputs(
            prompt_token_ids, sampling, steering_spec=steering_spec, position_mask=position_mask
        ):
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
    ) -> dict[Address, torch.Tensor]:
        """Async per-request worker-hook capture for a single prompt (concurrency-safe).

        Registers the points under a unique ``request_id``, runs that request, then
        collects only that request's rows (see the per-request demux in
        ``vllm_capture.requests``). Safe to call concurrently with other requests. When
        ``steering_spec`` (an engine ``SteeringSpec``) is given, the SAME request also
        steers, so the captured activations are post-cap (persona assistant-axis).

        ``detach=False`` always raises here: the returned tensors are rebuilt from bytes on this
        side of the process boundary, so no graph reaches back into the worker's forward. They are
        ordinary tensors, though, so a caller can build their own graph on top -- which is the
        ``downstream`` half of :attr:`grad_support`.
        """
        from vllm import SamplingParams  # pyright: ignore[reportMissingImports]

        if not detach:
            self.grad_support.require_through_forward()
        pts = _validate_hook_points(points, self._basis_if_loaded())
        self._require_capture_points(pts, "Activation capture")
        await self._ensure_engine()
        steered = steering_spec is not None and not steering_spec.is_empty()
        worker_specs: list[dict] = []
        static_write = False
        if steered:
            assert steering_spec is not None
            worker_specs = self._steer_specs(steering_spec)
            self._require_static_writes(worker_specs, "Steered generation")
        rid = self._new_request_id("np-cap")
        static_cap = self._use_static_capture()
        if static_cap:
            await self.engine.collective_rpc("register_static_capture", args=(rid, pts))
        else:
            await self.engine.collective_rpc("register_capture", args=(rid, pts))
        if steered and self._use_static_writes():
            await self._register_static_write(rid, worker_specs, prompt_token_ids=prompt_token_ids)
            static_write = True
        elif steered:
            await self.engine.collective_rpc("register_steering", args=(rid, worker_specs))
        try:
            await self._run_one(
                self._prompt(prompt_token_ids, private_kv_for=rid),
                SamplingParams(max_tokens=1, temperature=0.0),
                request_id=rid,
            )
        finally:
            if static_cap:
                payloads = await self.engine.collective_rpc("collect_static", args=(rid,))
            else:
                payloads = await self.engine.collective_rpc("collect_request", args=(rid,))
            if static_write:
                await self._unregister_static_write(rid)
            elif steered:
                await self.engine.collective_rpc("unregister_steering", args=(rid,))
        out = decode_capture_payload(payloads[0] if isinstance(payloads, list | tuple) else payloads)
        _assert_points_captured(out, pts)
        _assert_full_prompt_captured(out, len(prompt_token_ids))
        _assert_full_width_captured(out, self._hidden_size)
        return out

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

        Optional ``steering_spec`` (engine ``SteeringSpec``, additive/cap) OR
        ``lens_intervention`` (jlens steer/ablate/swap: ``{specs, steer_generated,
        skip_positions, prompt_len}``) is applied during generation so captured residuals
        reflect the intervention. Single request-locked use.
        """
        from vllm import SamplingParams  # pyright: ignore[reportMissingImports]

        pts = _validate_hook_points(points, self._basis_if_loaded())
        self._require_capture_points(pts, "Capture during generation")
        await self._ensure_engine()
        steered = steering_spec is not None and not steering_spec.is_empty()
        lens = bool(lens_intervention and lens_intervention.get("specs"))
        if lens and steered and self._use_static_writes():
            raise ValueError(
                'backend="vllm-static" cannot apply a steering spec and a lens intervention on '
                "the same request: both write the same static site. Run them as two requests, "
                'or reload with backend="vllm", whose per-request hooks can carry both.'
            )
        rid = self._new_request_id("np-capgen")
        static_cap = self._use_static_capture()
        worker_specs: list[dict] = []
        static_write = False
        if static_cap:
            await self.engine.collective_rpc("register_static_capture", args=(rid, pts))
        else:
            await self.engine.collective_rpc("register_capture", args=(rid, pts))
        if steered:
            assert steering_spec is not None
            worker_specs = self._steer_specs(steering_spec)
            self._require_static_writes(worker_specs, "Steered generation")
            if self._use_static_writes():
                await self._register_static_write(rid, worker_specs, prompt_token_ids=prompt_token_ids)
                static_write = True
            else:
                await self.engine.collective_rpc("register_steering", args=(rid, worker_specs))
        if lens:
            assert lens_intervention is not None
            if self._use_static_writes():
                await self._register_static_lens(rid, lens_intervention)
                static_write = True
            else:
                self._require_hooks("A lens intervention")
                await self.engine.collective_rpc(
                    "register_lens",
                    args=(
                        rid,
                        self._lens_specs(lens_intervention),
                        lens_intervention.get("steer_generated", False),
                        lens_intervention.get("skip_positions", []),
                        lens_intervention.get("prompt_len", 0),
                    ),
                )
        try:
            out = await self._run_one(
                self._prompt(prompt_token_ids, private_kv_for=rid),
                SamplingParams(max_tokens=max_tokens, temperature=temperature, seed=seed),
                request_id=rid,
            )
        finally:
            if static_cap:
                payloads = await self.engine.collective_rpc("collect_static", args=(rid,))
            else:
                payloads = await self.engine.collective_rpc("collect_request", args=(rid,))
            if static_write:
                await self._unregister_static_write(rid)
            elif steered or lens:
                await self.engine.collective_rpc("unregister_steering", args=(rid,))
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
        accumulating, so falling behind costs nothing but latency. Optional
        ``lens_intervention`` behaves exactly as in :meth:`capture_generation`.
        """
        from vllm import SamplingParams  # pyright: ignore[reportMissingImports]

        pts = _validate_hook_points(points, self._basis_if_loaded())
        self._require_capture_points(pts, "Streaming capture during generation")
        await self._ensure_engine()
        lens = bool(lens_intervention and lens_intervention.get("specs"))
        rid = self._new_request_id("np-capgen")
        static_cap = self._use_static_capture()
        static_write = False
        if static_cap:
            await self.engine.collective_rpc("register_static_capture", args=(rid, pts))
        else:
            await self.engine.collective_rpc("register_capture", args=(rid, pts))
        if lens and lens_intervention is not None:
            if self._use_static_writes():
                await self._register_static_lens(rid, lens_intervention)
                static_write = True
            else:
                self._require_hooks("A lens intervention")
                await self.engine.collective_rpc(
                    "register_lens",
                    args=(
                        rid,
                        self._lens_specs(lens_intervention),
                        lens_intervention.get("steer_generated", False),
                        lens_intervention.get("skip_positions", []),
                        lens_intervention.get("prompt_len", 0),
                    ),
                )

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
            if static_cap:
                payloads = await self.engine.collective_rpc("collect_static", args=(rid,))
            else:
                payloads = await self.engine.collective_rpc("collect_request", args=(rid,))
            tail = decode_capture_payload(payloads[0] if isinstance(payloads, list | tuple) else payloads)
            if static_write:
                await self._unregister_static_write(rid)
            elif lens:
                await self.engine.collective_rpc("unregister_steering", args=(rid,))
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
        """
        from vllm import SamplingParams  # pyright: ignore[reportMissingImports]

        pts = _validate_hook_points(points, self._basis_if_loaded())
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
        lens = bool(lens_intervention and lens_intervention.get("specs"))
        rid = self._new_request_id("np-lensread")
        static_cap = self._use_static_capture()
        static_write = False
        if static_cap:
            await self.engine.collective_rpc("register_static_capture", args=(rid, pts))
        else:
            await self.engine.collective_rpc("register_capture", args=(rid, pts))
        if lens and lens_intervention is not None:
            if self._use_static_writes():
                await self._register_static_lens(rid, lens_intervention)
                static_write = True
            else:
                self._require_hooks("A lens intervention")
                await self.engine.collective_rpc(
                    "register_lens",
                    args=(
                        rid,
                        self._lens_specs(lens_intervention),
                        lens_intervention.get("steer_generated", False),
                        lens_intervention.get("skip_positions", []),
                        lens_intervention.get("prompt_len", 0),
                    ),
                )

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
            tail = await readout(final=True)
            if static_write:
                await self._unregister_static_write(rid)
            elif lens:
                await self.engine.collective_rpc("unregister_steering", args=(rid,))
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

        The server path uses per-request lens registration inside
        :meth:`capture_generation`; this global variant remains for the sync/validation
        scripts. Cleared by :meth:`clear_steering`.

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
                "Pass the intervention to capture_generation / capture_generation_stream / "
                "lens_capture_readout_stream instead -- the per-request path installs a hook per site "
                "and honours the point (and stream) a spec names."
            )
        filled = [{**s, "point": str(s.get("point") or "resid_post")} for s in specs]
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
        self._require_hooks("A lens intervention")
        await self._ensure_engine()
        await self.engine.collective_rpc(
            "install_lens_intervention",
            args=(specs, steer_generated, skip_positions, prompt_len),
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

    async def set_lens_jacobians(self, jacobians: dict[int, torch.Tensor] | None) -> int:
        """Make the Jacobian-lens matrices resident on the worker(s). Returns bytes per rank.

        Hand this the whole lens once at startup and the read-out never has to ship residuals:
        :meth:`lens_capture_readout` transports and unembeds where the rows were captured. Pass
        ``None`` to release. Under tensor parallelism every rank holds a full copy (``J_bar`` is
        not sharded), so the return value is PER RANK, not in total.
        """
        await self._ensure_engine()
        payloads = (
            None
            if jacobians is None
            else {str(int(layer)): encode_tensor_payload(matrix) for layer, matrix in jacobians.items()}
        )
        results = await self.engine.collective_rpc("set_lens_jacobians", args=(payloads,))
        out = results[0] if isinstance(results, list | tuple) else results
        return int(out["bytes"])

    async def lens_transport(self, rows: torch.Tensor, layers: Sequence[int]) -> tuple[torch.Tensor, list[bool]]:
        """Pull ``[k, d_model]`` rows back through each layer's resident ``J_bar``: ``rows @ J_bar``.

        Returns ``([n_layers, k, d_model]`` float32, ``per-layer transported flags)``. Layers with
        no fitted ``J_bar`` come back unchanged. See :func:`worker_lens_transport`, including why
        this is the transpose of what the read-out applies: this is the steering direction, and
        it exists as an RPC because the lens lives on the worker.
        """
        await self._ensure_engine()
        payload = encode_tensor_payload(rows)
        results = await self.engine.collective_rpc("lens_transport", args=(payload, [int(x) for x in layers]))
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
            if static_attn:
                payloads = await self.engine.collective_rpc("collect_static", args=(rid,))
            else:
                payloads = await self.engine.collective_rpc("collect_attn_request", args=(rid,))
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
