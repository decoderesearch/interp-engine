"""``EagerModel``: a thin, autograd-capable wrapper over a raw HuggingFace CausalLM.

Loads the plain ``transformers`` model in standard eager PyTorch (op-by-op, autograd
intact) with ``no_processing`` semantics (no weight folding/centering — TransformerLens's
``from_pretrained_no_processing`` did the same). Because we run the real model and hook the
real submodules, every architecture gotcha (RoPE, RMSNorm offset, embed scaling, softcap,
masks) is applied for free inside ``forward()``.

The canonical identifier is the **raw HuggingFace repo id** (e.g. ``"openai-community/gpt2"``,
``"google/gemma-2-2b"``). There is no Neuronpedia/TransformerLens name aliasing here — the
caller passes the HF id directly.

Never imports from ``neuronpedia_inference``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import transformers
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, AutoConfig, AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    # transformers 5 renamed the config base class ``PretrainedConfig`` -> ``PreTrainedConfig``.
    # Importing the new spelling at runtime would raise on 4.57, which pyproject declares as the
    # floor and which downstream consumers really do pin -- circuit-tracer caps transformers at
    # <=4.57.3, because transformer-lens and nnsight do not support 5 yet. This name is only ever
    # an annotation, and annotations are strings here, so keeping the import type-only satisfies
    # both versions. Anything that needs the class at runtime must add a version fork instead.
    from transformers import PreTrainedConfig

from interp_engine import facts, moe_routing
from interp_engine.address import Address, to_address
from interp_engine.arch import ArchSpec, resolve_arch
from interp_engine.autograd_support import GradSupport, eager_grad_support
from interp_engine.chat_formatters import resolve_chat_formatter
from interp_engine.facts import factored_projection, text_config
from interp_engine.points import PointSpec, Scope, known_names, point_spec, points_for
from interp_engine.protocol import Completion
from interp_engine.residual_basis import ResidualBasis, eager_residual_basis
from interp_engine.tokenize import Tokenize

logger = logging.getLogger(__name__)

#: The mHC rows resolved together, off whichever hyper-connection modules the block spells.
#: `resid_streams` is not here: it is the block's own output, which `resid_post` already names.
_STREAM_POINTS: frozenset[str] = frozenset(
    f"{site}_stream_{quantity}" for site in facts.HYPER_CONNECTION_SITES for quantity in ("collapse", "write", "mix")
)


def _composite_text_config(hf_model_id: str, *, trust_remote_code: bool) -> Any | None:
    """The ``text_config`` of a composite (multimodal) config, or ``None`` if it isn't composite."""
    cfg = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=trust_remote_code)
    text_cfg = text_config(cfg)
    return text_cfg if text_cfg is not cfg else None


def resolve_trust_remote_code(hf_model_id: str, requested: bool | None) -> bool:
    """``requested``, or — when it is ``None`` — whether this checkpoint *needs* its bundled code.

    A checkpoint that ships modeling code and also has a class in the installed transformers is
    better loaded natively. The bundled copy is frozen at whatever transformers existed when the
    weights were uploaded, so it is the older of the two implementations and the one that breaks
    first: Phi-3-mini's, Nemotron-3-Nano's, Phi-mini-MoE's and DeepSeek-V2-Lite's all fail to
    import against current transformers, and all four have a native class that loads the same
    weights. Where transformers has no class for the family (EXAONE), the bundled code is the only
    implementation there is and has to be trusted.

    Asking transformers which case a checkpoint is in, rather than listing the checkpoints in
    either, is both shorter than the list and reaches mirrors and local copies of them.
    """
    if requested is not None:
        return requested
    try:
        cfg = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=False)
    except Exception:  # noqa: BLE001 - the config class itself is remote-only: nothing to prefer
        return True
    # `AutoConfig` succeeding only means transformers knows the *config*. The causal-LM mapping is
    # what says it also has a model to build: `PhiMoEConfig` is native while the class that
    # resolves from it is transformers' own `PhimoeForCausalLM`, so the config probe alone would
    # wrongly clear a checkpoint whose model still had to come from the hub. A composite
    # (multimodal) config is absent from this mapping too, which lands on the conservative answer.
    return type(cfg) not in MODEL_FOR_CAUSAL_LM_MAPPING


def _checkpoint_is_quantized(hf_model_id: str, *, trust_remote_code: bool) -> bool:
    """Whether the checkpoint carries its own ``quantization_config``, from the config alone.

    Config-only because the answer decides *how* to load, so it has to be known before the weights
    are read. Unreadable means "no": that lands on the plain load path, which is the one that reports
    a load failure with transformers' own message rather than one of ours about placement.
    """
    try:
        cfg = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=trust_remote_code)
    except Exception:  # noqa: BLE001 - not a fact about quantization; let the real load report it
        return False
    return facts.is_quantized(cfg)


def _load_hf_model(hf_model_id: str, load_kwargs: dict[str, Any], *, trust_remote_code: bool) -> nn.Module:
    """Load the raw HF model, transparently handling multimodal ``*ForConditionalGeneration`` repos.

    Plain decoder-only checkpoints load via ``AutoModelForCausalLM``. Multimodal /
    conditional-generation checkpoints (e.g. Qwen3.6-27B = ``Qwen3_5ForConditionalGeneration``,
    whose text dims are nested under ``text_config``) aren't mapped by ``AutoModelForCausalLM``, so
    we fall back to the concrete architecture class named in the config (then the image-text-to-text
    auto class). We load the whole checkpoint and only ever drive its text stack — ``resolve_arch``
    resolves the decoder trunk under ``model.language_model`` and the top-level ``lm_head``, so the
    vision/audio towers are never hooked or run (text-only forward passes ignore them).
    """
    try:
        return AutoModelForCausalLM.from_pretrained(hf_model_id, **load_kwargs)
    except AttributeError as err:
        # A composite config reached a text-only model class, which then failed reading a text
        # attribute off it. transformers narrows composite -> text itself, but only when the
        # mapped class's ``config_class`` IS the class registered as the config's ``text_config``
        # -- an object-identity comparison. vLLM registers its own config classes for some model
        # types (qwen3_5 among them) when an engine starts, which replaces that entry process-wide
        # and makes the comparison fail, so the narrowing is silently skipped. In a process that
        # has run vLLM, eager loading of such a model dies here. Doing the narrowing ourselves is
        # equivalent to what transformers does when the comparison holds.
        text_cfg = _composite_text_config(hf_model_id, trust_remote_code=trust_remote_code)
        if text_cfg is None:
            raise
        logger.info(
            "%s: %s; retrying with the text config passed explicitly (composite config was not narrowed)",
            hf_model_id,
            err,
        )
        return AutoModelForCausalLM.from_pretrained(hf_model_id, config=text_cfg, **load_kwargs)
    except ValueError as err:
        msg = str(err)
        # Only a genuine "this AutoModel can't map this architecture" error should trigger the
        # multimodal fallback; anything else (bad kwargs, corrupt weights) must surface.
        if "Unrecognized configuration class" not in msg and "AutoModelForCausalLM" not in msg:
            raise
        logger.info("AutoModelForCausalLM can't map %s; trying multimodal text-stack load", hf_model_id)

    cfg = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=trust_remote_code)
    last_err: Exception | None = None
    # Primary: the concrete architecture class named in the config (e.g. Qwen3_5ForConditionalGeneration).
    for arch_name in getattr(cfg, "architectures", None) or []:
        model_cls = getattr(transformers, arch_name, None)
        if model_cls is not None and hasattr(model_cls, "from_pretrained"):
            try:
                return model_cls.from_pretrained(hf_model_id, **load_kwargs)
            except Exception as e:  # noqa: BLE001 - fall through to the auto classes
                last_err = e
    # Fallback: the multimodal auto classes (these also yield a *ForConditionalGeneration).
    for auto_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        auto_cls = getattr(transformers, auto_name, None)
        if auto_cls is not None:
            try:
                return auto_cls.from_pretrained(hf_model_id, **load_kwargs)
            except Exception as e:  # noqa: BLE001
                last_err = e
    raise RuntimeError(f"Could not load {hf_model_id!r} as a causal LM or a multimodal text stack") from last_err


STR_TO_DTYPE: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _normalize_dtype(dtype: str | torch.dtype | None) -> torch.dtype | str | None:
    # "auto" is passed straight through to HF from_pretrained, which loads the checkpoint in its
    # native (config `torch_dtype`) precision — the right default for anything but tight fp32 parity.
    if dtype is None or dtype == "auto" or isinstance(dtype, torch.dtype):
        return dtype
    if dtype not in STR_TO_DTYPE:
        raise ValueError(f"Unknown dtype {dtype!r}; expected 'auto' or one of {list(STR_TO_DTYPE)}")
    return STR_TO_DTYPE[dtype]


def _require_hookable(module: nn.Module, name: str, layer: int | None, architecture: str) -> None:
    """Refuse a resolution that landed on a module no hook can fire on.

    A container -- an ``nn.ModuleList``, an ``nn.ModuleDict`` -- accepts ``register_forward_hook``
    and simply never calls it, because it has no ``forward``. So resolving to one produced a hook
    that silently did nothing: ``run_with_cache`` did eventually notice ("Captured nothing at ...")
    but anything that only *resolves* saw a success, which is how the coverage audit recorded a pass
    for LongcatFlash's attention points while they were bound to the ``ModuleList`` holding the two
    real attentions.

    Checked here, once, rather than at each of the resolver's thirty-odd returns.
    """
    if isinstance(module, nn.ModuleList | nn.ModuleDict) or type(module).forward is nn.Module.forward:
        where = f"{name!r}" + (f" at layer {layer}" if layer is not None else "")
        raise ValueError(
            f"{where} on {architecture} resolved to a {type(module).__name__}, which has no forward "
            "of its own: a hook registered there would never fire, and the capture would come back "
            "empty rather than wrong. This is a gap in this engine's resolution for the "
            "architecture, not something the caller can rephrase."
        )


class EagerModel:
    """Raw HF CausalLM + tokenizer + config-derived dims + canonical accessors."""

    def __init__(
        self,
        hf_model_id: str,
        *,
        device: str | None = None,
        dtype: str | torch.dtype | None = "float32",
        quantization_config: Any | None = None,
        attn_implementation: str | None = None,
        trust_remote_code: bool | None = None,
        tokenizer: Any | None = None,
        hf_model: nn.Module | None = None,
        model_kwargs: dict[str, Any] | None = None,
        device_map: str | dict[str, Any] | None = None,
        default_prepend_bos: bool = True,
        requires_grad: bool = False,
    ) -> None:
        self.hf_model_id = hf_model_id
        self._requested_device = device
        self._default_prepend_bos = default_prepend_bos
        self._requires_grad = requires_grad
        # Lazily computed in `grad_support`, never here: a verdict must not be part of loading.
        self._grad_support: GradSupport | None = None
        torch_dtype = _normalize_dtype(dtype)
        model_kwargs = dict(model_kwargs or {})
        # Resolved once, and only when something is actually fetched by id: a caller handing over
        # both an `hf_model` and a `tokenizer` gets no config read, which the meta-device coverage
        # audit depends on -- it builds synthetic trees under ids that resolve to nothing.
        #
        # Under its own name rather than reassigning the parameter, because the two are different
        # things: the parameter is a *request*, where None means "decide for me", and this is the
        # decision, which is always a bool. The initial value is never read -- both of its readers
        # below sit inside branches that imply the one that sets it.
        trust_remote: bool = False
        if hf_model is None or tokenizer is None:
            trust_remote = resolve_trust_remote_code(hf_model_id, trust_remote_code)

        if hf_model is None:
            load_kwargs: dict[str, Any] = {
                "dtype": torch_dtype,
                "trust_remote_code": trust_remote,
                **model_kwargs,
            }
            if quantization_config is not None:
                load_kwargs["quantization_config"] = quantization_config
            if attn_implementation is not None:
                load_kwargs["attn_implementation"] = attn_implementation
            # Multi-GPU sharding: with a device_map (e.g. "auto"/"balanced"), accelerate
            # places the layers across the visible GPUs itself. In that case we must NOT
            # do a manual .to(device) afterward (it would fight the accelerate placement /
            # try to pull the whole sharded model onto one GPU). The forward-hook capture /
            # steering layer is device-agnostic (accelerate moves activations between GPUs;
            # hooks fire on whichever device each module lives on).
            #
            # A checkpoint that ships quantized is placed at load time even when the caller named a
            # plain `device`, because "load, then move" is not a slower route to the same model here --
            # it is a different and much larger one. A quantizer with no kernels for the initial device
            # dequantizes to `dtype` to have something runnable, and CPU is such a device for every FP8
            # scheme, so the default route materializes bf16 weights at twice the checkpoint's size
            # before anything reaches the GPU: DeepSeek-V4-Flash loads as ~285 GiB of bf16 on the way
            # to a card that holds its 156 GiB of FP8 comfortably, and dies moving it. Handing
            # transformers the destination up front keeps the weights in the dtype they ship in, which
            # is also what its own warning on this path asks for.
            placement = device_map
            if (
                placement is None
                and device is not None
                and _checkpoint_is_quantized(hf_model_id, trust_remote_code=trust_remote)
            ):
                placement = device
                logger.info("%s is a quantized checkpoint: loading directly onto %s", hf_model_id, device)
            if placement is not None:
                load_kwargs["device_map"] = placement
            logger.info(
                "Loading raw HF model %s (%s%s)",
                hf_model_id,
                torch_dtype,
                f", device_map={placement}" if placement is not None else "",
            )
            hf_model = _load_hf_model(hf_model_id, load_kwargs, trust_remote_code=trust_remote)
            # Quantized / device_map loads place weights on-device themselves; only move otherwise.
            if quantization_config is None and placement is None and device is not None:
                hf_model = hf_model.to(device)  # type: ignore[assignment]

        self.hf_model: nn.Module = hf_model  # type: ignore[assignment]
        self.hf_model.eval()
        # Serving is forward-only (activations, attention, lens read-out, steering — no
        # backward pass), so the default freezes grads to make that contract explicit and
        # avoid building autograd graphs during capture/generation. `requires_grad=True`
        # opts into a differentiable model for research use; it costs activation memory on
        # every capture that keeps its graph, which is why serving does not pay it.
        self.hf_model.requires_grad_(requires_grad)
        self.config = self.hf_model.config

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=trust_remote)
        # Annotated `Any` (not inferred from the `Any | None` parameter): a tokenizer is always
        # present past this point, and callers should not have to narrow away a `None` that
        # cannot occur. It stays untyped because multimodal archs pass a processor here.
        self.tokenizer: Any = tokenizer

        self.arch: ArchSpec = resolve_arch(self.hf_model, self.config)
        self._attn_implementation = getattr(self.config, "_attn_implementation", attn_implementation)
        self.tok = Tokenize(
            self.tokenizer,
            default_prepend_bos=default_prepend_bos,
            device=str(self.device),
            # None for every family whose tokenizer carries its own chat template, which is
            # nearly all of them. Never raises: a model whose code-defined format cannot be
            # loaded still serves everything except chat input.
            formatter=resolve_chat_formatter(
                getattr(self.config, "architectures", None),
                hf_model_id,
                trust_remote_code=trust_remote,
            ),
        )

        # `requires_grad=True` IS a gradient request, so this is its point of use and the same gate
        # every other request goes through -- which is why it does not contradict "gradient support
        # never gates loading": a plain load never reaches this branch. Checked here, at the end,
        # because dtype="auto" is only resolved by actually loading the weights.
        if requires_grad:
            self.grad_support.require_through_forward()
            for caveat in self.grad_support.caveats:
                logger.warning("%s loaded with requires_grad=True: %s", hf_model_id, caveat)

    # --- tokenization (delegates to the Tokenize layer) ---------------------
    def to_tokens(self, text: str | list[str], **kwargs: Any) -> torch.Tensor:
        return self.tok.to_tokens(text, **kwargs)

    def to_str_tokens(self, text: str | torch.Tensor, **kwargs: Any) -> list[str]:
        return self.tok.to_str_tokens(text, **kwargs)

    def to_string(self, tokens: Any) -> str | list[str]:
        return self.tok.to_string(tokens)

    @property
    def default_prepend_bos(self) -> bool:
        return self._default_prepend_bos

    # --- classmethod convenience --------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        hf_model_id: str,
        *,
        dtype: str | torch.dtype | None = "auto",
        torch_dtype: str | torch.dtype | None = None,
        **kwargs: Any,
    ) -> EagerModel:
        """Load a model the way ``AutoModelForCausalLM.from_pretrained`` does.

        For code being ported off transformers: the call shape is familiar, and the result
        adds capture and steering. ``EagerModel(...)`` is the same thing -- this only spares
        you rewriting the call site.

        Two differences from the constructor's defaults, both to match what transformers
        does rather than what serving wants: ``dtype`` defaults to ``"auto"`` (the
        checkpoint's own precision) instead of ``"float32"``, and ``torch_dtype`` is accepted
        as the deprecated alias transformers still sees in the wild.
        """
        if torch_dtype is not None:
            if dtype not in (None, "auto"):
                raise ValueError("Pass either dtype or torch_dtype, not both (torch_dtype is deprecated)")
            dtype = torch_dtype
        return cls(hf_model_id, dtype=dtype, **kwargs)

    @classmethod
    def from_config_only(cls, hf_model_id: str) -> PreTrainedConfig:
        """Load just the config (used for cheap arch/quirk inspection without weights)."""
        return AutoConfig.from_pretrained(hf_model_id, trust_remote_code=True)

    # --- dims ----------------------------------------------------------------
    @property
    def n_layers(self) -> int:
        return self.arch.n_layers

    @property
    def n_heads(self) -> int:
        return self.arch.n_heads

    @property
    def n_kv_heads(self) -> int:
        return self.arch.n_kv_heads

    @property
    def head_dim(self) -> int:
        return self.arch.head_dim

    @property
    def d_model(self) -> int:
        return self.arch.d_model

    @property
    def vocab_size(self) -> int:
        return self.arch.vocab_size

    @property
    def device(self) -> torch.device:
        return next(self.hf_model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.hf_model.parameters()).dtype

    # --- capabilities --------------------------------------------------------
    @property
    def requires_grad(self) -> bool:
        """Whether this model was built to carry gradients (see :attr:`grad_support`)."""
        return self._requires_grad

    @property
    def grad_support(self) -> GradSupport:
        """What kind of gradients this model can provide. Computed on first access, then cached.

        Reads the *loaded* dtype rather than the requested one, so ``dtype="auto"`` is judged by what
        the checkpoint actually turned out to be.
        """
        if self._grad_support is None:
            self._grad_support = eager_grad_support(
                self._requires_grad,
                dtype=str(self.dtype).removeprefix("torch."),
                architectures=getattr(self.config, "architectures", None),
                quantization=self.quant_method,
            )
        return self._grad_support

    @property
    def quant_method(self) -> str | None:
        """The checkpoint's quantization scheme as transformers names it, or None if unquantized.

        Read off the *loaded* config rather than the ``quantization_config`` argument, so a
        pre-quantized checkpoint and a quantize-on-load are answered the same way -- transformers
        writes the resolved config back either way. Total by construction: an unrecognised shape reads
        as unquantized rather than raising, since this only feeds a capability verdict and a wrong
        refusal there would be worse than a missing caveat.
        """
        config = getattr(self.config, "quantization_config", None)
        if config is None:
            return None
        method = config.get("quant_method") if isinstance(config, dict) else getattr(config, "quant_method", None)
        if method is None:
            return None
        # Newer transformers uses a `QuantizationMethod` str-enum, whose `str()` is the member repr.
        return str(getattr(method, "value", method))

    @property
    def hooks_available(self) -> bool:
        """Always True: this backend holds the module tree and hooks it in-process.

        A constant, and still worth having, because it is what lets a caller ask "can this instance
        capture?" without first asking which backend it is. The vLLM twin is the one that can answer
        False (``enforce_eager=False`` -> CUDA graph replay skips the Python forward).
        """
        return True

    @property
    def graph_replay(self) -> bool:
        return False

    @property
    def static_points(self) -> tuple:
        return ()

    @property
    def static_writes(self) -> tuple:
        return ()

    @property
    def residual_basis(self) -> ResidualBasis:
        """How this model's residual stream is structured. See :mod:`interp_engine.residual_basis`.

        Recomputed per access rather than cached, unlike :attr:`grad_support` and unlike the vLLM
        twin. It is three scalars off ``arch.quirks`` into a frozen dataclass, so there is nothing to
        save -- while a cache would go stale the moment ``arch.quirks`` is replaced, which is how a
        family's structure gets simulated on an already-loaded model. ``resolve_point`` consults this
        on every call, so a stale verdict would be worse than a recomputed one.
        """
        return eager_residual_basis(
            n_residual_streams=self.arch.quirks.n_residual_streams,
            parallel_attn_mlp=self.arch.quirks.parallel_attn_mlp,
            architecture=self.arch.architecture,
        )

    @property
    def attn_implementation(self) -> str | None:
        return getattr(self.config, "_attn_implementation", self._attn_implementation)

    @property
    def eager_attention(self) -> bool:
        return self.attn_implementation == "eager"

    def _require_stream_coordinate(self, name: str, stream: int | None) -> None:
        """Whether ``name`` on this model may carry a stream -- see
        :meth:`~interp_engine.residual_basis.ResidualBasis.require_stream_coordinate`.

        A coordinate is a selector, so ignoring one is never a safe default: dropping ``stream=2``
        hands back a correct-looking tensor for a question nobody asked. The arithmetic is on the
        basis rather than here because the vLLM client asks the same question of the same verdict.
        """
        self.residual_basis.require_stream_coordinate(name, stream)

    def _require_block_boundary(self, name: str, layer: int, side: str) -> None:
        """Refuse a block-level residual point at a flattened position inside a block.

        ``resid_pre``/``resid_post`` are the block's own input and output, so on a trunk whose blocks
        run several sublayer pairs they exist only at the first and last position of each block. The
        residual *between* two pairs is formed inside one forward and no module boundary carries it.

        Refusing matters more here than usual, because the plausible wrong answer is so close: the
        block's output really is a residual of the right width at the right token positions -- just
        one or more sublayers later than the address asked for.
        """
        slot = self.arch.slot_for(layer)
        if slot.of == 1:
            return
        if side == "output" and slot.is_last_in_block:
            return
        if side == "input" and slot.is_first_in_block:
            return
        first, last = layer - slot.slot, layer - slot.slot + slot.of - 1
        raise ValueError(
            f"Layer {layer} of {self.arch.architecture} is position {slot.slot} of {slot.of} inside "
            f"decoder block {slot.block}, and {name!r} is the block's own {side}. The residual between "
            "two sublayer pairs is formed inside a single forward, so no module boundary carries it. "
            f"Use {name!r} at layer {first if side == 'input' else last} for this block's boundary, or "
            "the sublayer points ('attn_out', 'mlp_out'), which do exist at every position."
        )

    def _require_whole_feed_forward(self, layer: int) -> None:
        """Refuse ``mlp_out`` where ``layer.mlp`` is one branch of a two-branch feed-forward.

        Gemma-4's sparse layers keep the dense MLP and hang the router and experts *beside* it, then
        sum ``post_ffn_norm_1(mlp(x)) + post_ffn_norm_2(experts(x))`` inside the block's own forward.
        So ``layer.mlp`` is a complete module whose output is half of what this point names, at the
        right width and the right token positions, with nothing about it to notice -- which is the
        one failure mode this engine refuses on principle rather than serving.

        Refused rather than served from the sum, because the sum has no module boundary: it is a
        local of the block's forward, reachable only the way the kernel-local points are, and there
        is already a point that means the tensor a caller of this one wants. ``mlp_out_post`` is the
        ``post_feedforward_layernorm``'s output -- downstream of the sum, so it includes both
        branches -- and it is what keeps ``resid_post == resid_mid + mlp_out_post`` true here, so
        every residual decomposition is unaffected.

        Note what is NOT refused. ``mlp_act`` and the rest of the neuron basis are the dense
        branch's own internals and mean exactly what they mean elsewhere. ``mlp_in`` is the dense
        branch's input and is served: the block normalizes the same residual twice, once per branch,
        so there is no single tensor "the feed-forward reads" to prefer over it -- ``resid_mid`` is
        the quantity both branches are functions of.
        """
        if not self.arch.mlp_is_half_the_feed_forward(layer):
            return
        raise ValueError(
            f"Layer {layer} of {self.arch.architecture} runs its routed experts BESIDE the dense MLP "
            "rather than instead of it, and sums the two branches inside the block's forward. So "
            "'mlp_out' -- the output of `layer.mlp` -- is the dense branch alone: a correctly shaped "
            "d_model tensor that is half this layer's feed-forward, which is why it is refused rather "
            "than returned. Capture 'mlp_out_post' instead: it is the post-feedforward norm's output, "
            "downstream of the sum, so it carries both branches and is the layer's actual residual "
            "contribution (resid_post == resid_mid + mlp_out_post holds). The dense branch's own "
            "internals are unaffected -- 'mlp_in', 'mlp_pre', 'mlp_pre_linear' and 'mlp_act' all "
            "resolve here -- and the routed branch is described by the routing points."
        )

    # --- canonical hook-point resolution ------------------------------------
    def points(self) -> tuple[PointSpec, ...]:
        """Every point addressable on *this* model: the global table plus what its trunk adds.

        The accessor a caller enumerating points should use, rather than importing ``POINTS``
        directly -- that one cannot know how many residual streams it is looking at, and so cannot
        mention the seven hyper-connection rows a multi-stream trunk has.
        """
        return points_for(self.residual_basis.n_streams)

    def _resolve_hyper_connection(self, name: str, layer: int) -> tuple[nn.Module, str]:
        """The mHC points, off the hyper-connection modules a block carries two of.

        Where each quantity sits is :data:`facts.HYPER_CONNECTION_LAYOUTS`, because the two families
        that ship this trunk do not agree: both return their coefficients from the mHC module, in
        different orders, and only DeepSeek-V4 also returns the collapsed ``d_model`` vector, which
        on Motif 3 is instead the input its pre-sublayer norm is called on.

        Everything reachable here is read as module I/O rather than recomputed, and that is not a
        close call: the mix is a softmax followed by a Sinkhorn projection whose iteration count is a
        config field, and reproducing it slightly wrong would yield a doubly-stochastic-looking
        matrix that is not the model's.
        """
        site, _, quantity = name.partition("_stream_")
        found = self.arch.hyper_connection_boundary(layer, site, quantity)
        if found is None:
            raise ValueError(
                f"{name!r} is a hyper-connection point, and {self.arch.architecture} has no "
                f"{'attention' if site == 'attn' else 'MLP'} hyper-connection module on layer "
                f"{layer}. Only a trunk carrying several residual streams has these."
            )
        return found

    def resolve_point(self, name: str, layer: int | None = None, *, stream: int | None = None) -> tuple[nn.Module, str]:
        """Map a canonical hook name (+ optional coordinates) to ``(module, "input"|"output")``.

        The set of names is open: unknown names fall through to a direct module-path lookup
        so new consumers can tap points the core doesn't enumerate. ``interp_engine.points``
        declares what each *canonical* name means; this decides where it lives, including the
        refusals that have to explain themselves.

        ``stream`` is keyword-only and appended rather than inserted, so the ~90 positional
        ``(name, layer)`` call sites are unaffected.
        """
        self._require_stream_coordinate(name, stream)
        module, side = self._resolve_point(name, layer)
        _require_hookable(module, name, layer, self.arch.architecture)
        return module, side

    def derived_routing(self, name: str, layer: int | None) -> str | None:
        """The convention ``run_with_cache`` can rebuild ``name`` with, or None if it must be read.

        Not None only where all three hold: the point is one of the two halves of the top-k, this
        layer's block routes *inline* so no module boundary carries them, and this family's convention
        has been verified against its own router (:data:`facts.ROUTING_CONVENTIONS`). Where the router
        runs, this returns None and the point is read as always -- a recompute never displaces a read.

        A predicate rather than part of :meth:`resolve_point` because there is no address to return:
        the tensor is built after the pass from one that was captured, which is the capture path's job.
        """
        if name not in moe_routing.DERIVED_POINTS or layer is None:
            return None
        if self.arch.inline_routing_logits(layer) is None:
            return None
        return facts.routing_convention(self.arch.architecture)

    def _resolve_point(self, name: str, layer: int | None = None) -> tuple[nn.Module, str]:
        """The resolution itself. Wrapped by :meth:`resolve_point`, which validates the result."""
        if name == "embeddings":
            return self.arch.embed, "output"
        if name == "final_norm":
            return self.arch.final_norm, "output"
        if name == "lm_head":
            return self.arch.lm_head, "output"

        if layer is None:
            spec = point_spec(name, self.residual_basis.n_streams)
            if spec is not None and spec.scope is Scope.LAYER:
                raise ValueError(
                    f"Canonical point {name!r} is per-layer, so it needs a layer index. Without one "
                    "this would be read as a dotted module path and fail to find a module by that name."
                )
            # ad-hoc dotted module path, e.g. "model.layers.5.mlp.down_proj"
            return self.get_submodule(name), "output"

        if name == "resid_streams":
            # The same module and side as `resid_post`, deliberately: this is the whole
            # `(batch, seq, streams, d_model)` stack rather than one stream of it. Its own name
            # rather than "`resid_post` with no coordinate", because that is the request the basis
            # verdict refuses -- the refusal is what stops a d_model consumer meeting a stack, and
            # a caller that genuinely wants the stack should have to say so.
            if self.residual_basis.n_streams == 1:
                raise ValueError(
                    f"{self.arch.architecture} carries a single residual stream, so there is no "
                    f"stack for {name!r} to name. Use 'resid_post', which is that one stream."
                )
            self._require_block_boundary(name, layer, "output")
            return self.arch.block(layer), "output"
        if name == "resid_post":
            self._require_block_boundary(name, layer, "output")
            return self.arch.block(layer), "output"
        if name == "resid_pre":
            # The residual entering decoder layer `layer` (its forward input). For layer 0
            # this correctly includes positional embeddings / embed scaling that the bare
            # token-embedding output would miss.
            self._require_block_boundary(name, layer, "input")
            return self.arch.block(layer), "input"
        if name == "mlp_in":
            # Via `mlp_boundary`, not the MLP module directly: a block that inlines its projections
            # (OPT's `fc1`/`fc2`) has these two tensors but no module whose input/output they are.
            return self.arch.mlp_boundary(layer, "in")
        if name == "mlp_out":
            self._require_whole_feed_forward(layer)
            return self.arch.mlp_boundary(layer, "out")
        if name == "attn_in":
            # Symmetric with `mlp_in`: what the attention block reads, i.e. the normed residual.
            # On a `parallel_attn_mlp` architecture this is the same tensor as `mlp_in`.
            return self.arch.attn_boundary(layer, "in")
        if name == "attn_out":
            return self.arch.attn_boundary(layer, "out")
        if name == "resid_mid":
            # The residual *between* the two sublayers (TransformerLens `hook_resid_mid`):
            # `resid_pre + attn_out_post`. Taken at the input to the pre-MLP norm, so it is the
            # tensor the model actually formed rather than a sum we reconstructed -- and so it is
            # right on the families where those differ (a post-norm block's attention contribution
            # is normed, and `resid_mid` is downstream of that norm).
            #
            # With no pre-MLP norm (OLMo-2/3) the MLP reads the residual itself, so this aliases
            # `mlp_in` and the caller does not branch. On a parallel block nothing sequences the
            # sublayers, so the quantity does not exist -- refuse rather than return `resid_pre`,
            # which is what the pre-MLP norm's input *is* there and is wrong by a whole sublayer.
            self.residual_basis.require_sequential(name)
            if not self.arch.has_position_mixer(layer):
                # A block with one sublayer has no residual *between* two. Nemotron-H is built this way
                # -- each layer is an attention, or an MLP, or a Mamba2 mixer -- and there the pre-MLP
                # norm is called `norm`, which reads the block's input: returning its input would be
                # `resid_pre` under another name, and returning the MLP's would be the normed value.
                #
                # The test is whether anything mixes positions, not whether attention does: an LFM2
                # `conv` layer runs a short convolution where attention would go and is otherwise a
                # plain sequential block, so `resid_mid` is its `ffn_norm`'s input like anywhere else.
                raise ValueError(
                    f"Layer {layer} of {self.arch.architecture} is a feed-forward block with no "
                    "position-mixing sublayer, so there is no residual between two sublayers on it: "
                    "'resid_mid' is undefined here. Use 'resid_pre' (what this block reads) or "
                    "'resid_post' (after its contribution is added)."
                )
            norm = self.arch.pre_mlp_norm(layer)
            if norm is not None:
                return norm, "input"
            # No pre-MLP norm *found*, which has two causes and only one of them permits the alias to
            # `mlp_in`: the block genuinely has none (OLMo-2/3, which post-norm instead), or it has one
            # under a spelling `PRE_MLP_NORM_ATTRS` does not carry. Aliasing in the second case returns
            # the *normed* tensor under the residual's name, and nothing about the result looks wrong.
            # So require positive evidence -- a post-MLP norm, which is what leaves the MLP's input
            # unnormalized in the first place. (This call also raises the better message on a block with
            # no feed-forward sublayer at all, where the question does not arise.)
            boundary = self.arch.mlp_boundary(layer, "in")
            if self.arch.post_mlp_norm(layer) is not None:
                return boundary
            if not self.arch.has_mlp_module(layer):
                # Inlined projections (OPT, XGLM), which needs its own answer rather than the generic
                # "add the spelling" below: OPT's pre-MLP norm is `final_layer_norm`, which is also what
                # its *trunk* calls the model's final norm, and `do_layer_norm_before` decides whether it
                # runs before the MLP or after. So the name cannot go in a vocabulary -- it would bind
                # the wrong module on one of the two shapes.
                raise ValueError(
                    f"{self.arch.architecture} inlines its MLP projections on the decoder layer, so no "
                    "module's input is the residual between the sublayers, and this block's pre-MLP norm "
                    "is not identifiable by name: it is called 'final_layer_norm', as the trunk calls the "
                    "model's final norm, and `config.do_layer_norm_before` decides whether it runs before "
                    "the MLP or after it. Capture 'resid_pre'/'resid_post', or that norm by module path."
                )
            raise ValueError(
                f"Layer {layer} of {self.arch.architecture} has neither a pre-MLP norm this engine can "
                "name nor a post-MLP norm, so the residual between the sublayers cannot be identified: "
                f"the block's norms are {self.arch.block_norm_attrs(layer) or 'none'}. If one of those "
                "takes the residual and feeds the MLP, add its spelling to `facts.PRE_MLP_NORM_ATTRS`; "
                "if the block normalizes its MLP output, add that one to `facts.POST_MLP_NORM_ATTRS`. "
                "Meanwhile 'resid_pre' and 'resid_post' are unaffected, and that norm is capturable by "
                "module path."
            )
        if name in ("q_norm_in", "q_norm_out", "k_norm_in", "k_norm_out"):
            # QK-norm, inside the attention module and before RoPE. `*_in` is the raw projection
            # reshaped as the norm receives it; `*_out` is the normalized value RoPE then reads,
            # weight multiply included -- which is NOT TransformerLens' `hook_normalized` (that
            # fires before the weight). The norm's *scale* is an intermediate rather than a
            # boundary, so no hook can return it; recompute it from `*_in` with
            # `facts.rms_norm_eps` (see docs/ENGINE_HOOK_MAPPINGS.md).
            #
            # Shape is family-dependent -- per-head on Qwen3-style, flat on OLMo-2-style -- and
            # `arch.quirks.qk_norm` says which, because the conventions differ by a reshape.
            which, side = name.split("_norm_")
            return self.arch.qk_norm_module(layer, which), ("input" if side == "in" else "output")
        if name in ("mlp_act", "mlp_pre", "mlp_pre_linear"):
            # Inside the MLP, in the neuron basis (`d_mlp` wide, not `d_model`).
            #
            # `mlp_act` is the post-activation neuron vector -- TransformerLens' `mlp.hook_post`, the
            # basis MLP transcoders and neuron dashboards index. It is the down projection's INPUT
            # because no module output holds it: the activation function is applied inline, not by a
            # submodule, so `act_fn(gate) * up` is formed and consumed inside one forward.
            #
            # `mlp_pre` is what goes *into* the activation function and `mlp_pre_linear` the branch
            # that is multiplied by it instead of activated. Which projection each names depends on
            # gating (see `facts.is_gated_mlp`): on a gated MLP `gate_proj` and `up_proj`, on a plain
            # one the single `c_fc`-style projection and nothing.
            if name == "mlp_act":
                return self.arch.mlp_projection(layer, "down"), "input"
            # A dense MLP that keeps the two branches in one projection has no module output per
            # branch, so the address is the fused projection and `capture.run_with_cache` takes the
            # half this point names -- the same shape as `value` off a fused QKV, and for the same
            # reason: the tensor is read, and the split is the one the block's own forward does.
            if (fused := self.arch.fused_gate_up(layer)) is not None:
                return fused[0], "output"
            which = "pre_act" if name == "mlp_pre" else "pre_linear"
            return self.arch.mlp_projection(layer, which), "output"
        if name in ("router_logits", "expert_weights", "expert_indices"):
            # The MoE routing decision, all three from the router's own output tuple -- usually
            # `(logits, weights, indices)`, in whatever order this family returns them
            # (`facts.ROUTER_OUTPUTS`). Read rather than recomputed from the logits: every
            # family's convention differs (softmax before or after the top-k, renormalized or not,
            # sigmoid scoring, expert-group masking) and they all yield k weights summing to 1, so a
            # recomputation that assumed the wrong one would be plausible and silently different.
            #
            # `expert_indices` is integer-valued, so it is the one point whose tensor is not a
            # differentiable activation -- the top-k is where the model stops being continuous.
            #
            # A block whose forward routes inline never calls its router, but the recognized
            # replacements return the logits themselves, so those are read off the block instead of
            # being lost with the other two. Which is not the same as recomputing them: the tensor is
            # the one the kernel routed on, bit-identical to the router's own linear on gpt-oss's MXFP4
            # path. Only `router_logits` -- see `Arch.inline_routing_logits`. The other two are rebuilt
            # from it by `run_with_cache` where this family's convention is a verified one
            # (`derived_routing` below), which is a capture-path concern and has no address to return.
            if name == "router_logits" and (inline := self.arch.inline_routing_logits(layer)) is not None:
                return inline
            router = self.arch.moe_router(layer)
            # A family whose router returns *probabilities* where the tuple's logits slot is has its
            # logits one module deeper, and that projection's output is what its own softmax consumed
            # -- read, not recomputed. Gemma-4 only; see `facts.ROUTER_LOGITS_SUBMODULE`. Asked before
            # the tuple index so the slot that does not hold logits is never indexed for them.
            if name == "router_logits" and (attr := facts.router_logits_submodule(self.arch.architecture)):
                return getattr(router, attr), "output"
            element = facts.router_output_index(self.arch.architecture, name)
            return router, f"output:{element}"
        if name == "mlp_out_post":
            # The MLP's *residual contribution*: `resid_post = resid_mid + mlp_out_post` holds for
            # this and fails for raw `mlp_out` on any post-norm architecture. On a model with no
            # post-sublayer norm the two are the same tensor, so this aliases the raw point and
            # callers never have to branch on architecture.
            post = self.arch.post_mlp_norm(layer)
            return (post, "output") if post is not None else self.arch.mlp_boundary(layer, "out")
        if name == "attn_out_post":
            post = self.arch.post_attn_norm(layer)
            return (post, "output") if post is not None else self.arch.attn_boundary(layer, "out")
        if name == "z":
            # Per-head attention output pre-W_O (TransformerLens `hook_z`), i.e. the INPUT to the
            # attention output projection: [batch, pos, n_heads*head_dim]. This is what
            # attention-output SAEs are trained on. Symmetric with `value` (v_proj output).
            return self.arch.attn_out_proj(layer), "input"
        if name == "attn_gate":
            # Two shapes of gate. Where the family keeps a separate projection for it (Afmoe, Laguna)
            # its output *is* the point. Where the gate is packed into a double-width `q_proj`
            # (Qwen3-Next, Qwen3.5) this is that raw output, whose second per-head half is the gate;
            # post-process with `capture.attn_out_gate`, which owns the interleaved split.
            if not self.arch.quirks.gated_attn_out:
                raise ValueError(
                    f"{self.arch.architecture} does not gate its attention output, so there is no "
                    "'attn_gate' to capture."
                )
            standalone = self.arch.attn_out_gate_proj(layer)
            return (standalone or self.arch.q_proj(layer)), "output"
        if name == "value":
            # A norm between the projection and attention wins over the projection, because this point
            # means the tensor the attention pattern is applied to and the norm is the last thing that
            # touches it. Gemma-4 only, so far -- and there it is also the *whole* answer on a
            # `attention_k_eq_v` layer, which has no value projection to prefer it over: the value is
            # the key projection's output, read before k_norm and before RoPE, so nothing but this
            # norm's output distinguishes it from the key. See `ArchSpec.value_module`.
            if (normed := self.arch.value_module(layer)) is not None:
                return normed, "output"
            v = self.arch.v_proj(layer)
            if v is not None:
                return v, "output"
            fused = self.arch.fused_qkv_module(layer)
            if fused is not None:
                return fused, "output"
            # Multi-head *latent* attention is the usual reason, and where the factored pair is
            # actually found the refusal can name the latent's own module rather than describing the
            # pattern and leaving the caller to go looking for it.
            latent = factored_projection(self.arch.attn_module(layer), "kv")
            where = (
                f"Capture the latent, which on this model is {latent.latent_attr!r} on the attention module"
                if latent is not None
                else "Capture that latent by module path"
            )
            raise ValueError(
                f"No value projection to hook on layer {layer} of {self.arch.architecture}: neither a "
                "standalone v_proj nor a fused qkv. Multi-head *latent* attention is the usual reason "
                "(DeepSeek-V2/V3/V4, MiniCPM3, GLM-MoE): it carries a compressed kv latent and expands "
                f"it inside the forward, so no module's output is the value. {where}, or use 'z' -- "
                "the per-head attention output, which exists either way."
            )
        if name in _STREAM_POINTS:
            return self._resolve_hyper_connection(name, layer)

        spec = point_spec(name, self.residual_basis.n_streams)
        if spec is not None and not spec.module_resolved:
            # Declared, but no module boundary carries it -- the capture path special-cases these.
            raise ValueError(
                f"Canonical point {name!r} is not resolvable to a module: {spec.note or 'no module holds it'}. "
                "Request it through run_with_cache, which owns its capture path."
            )
        close = sorted(n for n in known_names() if n.startswith(name[:4]) or name.startswith(n[:4]))
        raise ValueError(
            f"Unknown canonical hook name {name!r}"
            + (f"; did you mean one of {close}?" if close else f" (known: {sorted(known_names())})")
        )

    def get_submodule(self, dotted_path: str) -> nn.Module:
        return self.hf_model.get_submodule(dotted_path)

    # --- InterpModel protocol ------------------------------------------------
    # Async wrappers over the sync free functions, so that code holding either backend
    # can call the same methods (see interp_engine.protocol). They return CPU tensors
    # WITHOUT a batch dimension, matching VLLMModel; the free functions stay batched and
    # on-device and remain the better entry point when you already know it's eager.
    #
    # The imports are function-local because capture/steer/lens all import this module,
    # so importing them at module scope would be circular.

    async def warmup(self) -> None:
        """No-op: ``__init__`` already loaded the weights. Present for protocol parity
        with vLLM, whose engine build is deferred to the first async call."""

    async def shutdown(self) -> None:
        """Move the weights off the accelerator so the next model can have its memory.

        Dropping the last Python reference is otherwise enough on this backend (unlike
        vLLM), but a caller holding the model in a local would keep the device allocation
        alive until the frame exits.
        """
        if self.device.type != "cpu":
            self.hf_model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _maybe_steer(self, steering_spec: Any, prompt_token_ids: torch.Tensor) -> AbstractContextManager[Any]:
        """Steering context for ``steering_spec``, or a no-op when there is nothing to steer."""
        from interp_engine.steer import steer, steering_spec_to_eager_specs

        specs = steering_spec_to_eager_specs(steering_spec) if steering_spec is not None else []
        if not specs:
            return nullcontext()
        return steer(self, specs, prompt_token_ids=prompt_token_ids)

    async def capture(
        self,
        prompt_token_ids: Any,
        points: Any,
        *,
        steering_spec: Any = None,
        detach: bool = True,
    ) -> dict[Address, torch.Tensor]:
        """Capture ``points`` over one prompt. See :meth:`interp_engine.protocol.InterpModel.capture`.

        ``detach=False`` keeps the autograd graph, and requires the model to have been built
        with ``requires_grad=True`` -- see :attr:`grad_support`. Results stay on the model's
        device in that case, since moving them to CPU is a graph node nobody asked for.
        """
        from interp_engine.capture import run_with_cache

        if not detach:
            self.grad_support.require_through_forward()
        addresses = [to_address(p) for p in points]
        input_ids = torch.tensor([[int(t) for t in prompt_token_ids]], device=self.device)
        with self._maybe_steer(steering_spec, input_ids):
            cache = run_with_cache(self, input_ids, addresses, detach=detach)
        if not detach:
            return {a: cache[a][0] for a in addresses}
        return {a: cache[a][0].cpu() for a in addresses}

    async def capture_generation(
        self,
        prompt_token_ids: Any,
        points: Any,
        *,
        max_tokens: int = 8,
        temperature: float = 0.0,
        seed: int | None = None,
        steering_spec: Any = None,
    ) -> tuple[Any, dict[Address, torch.Tensor]]:
        """Generate, capturing at prompt AND generated positions.

        See :meth:`interp_engine.protocol.InterpModel.capture_generation`. Implemented as
        generate-then-capture: one forward over ``prompt + generated[:-1]`` sees exactly the
        positions the generation loop processed, and a causal model's activation at each
        position does not depend on later tokens, so this equals capturing during the loop
        while costing one extra prefill instead of per-step bookkeeping. It is also the
        reference the vLLM decode-time capture is validated against
        (``scripts/vllm_capture_generation_check.py``).
        """
        prompt_ids = [int(t) for t in prompt_token_ids]
        with self._maybe_steer(steering_spec, torch.tensor([prompt_ids])):
            completion = self._generate_completion(prompt_ids, max_tokens, temperature, seed)

        # The last sampled token is never fed back through the model, so it has no
        # activations; dropping it here is what makes the captured length match vLLM's.
        gen_ids = list(completion.token_ids)
        processed = prompt_ids + gen_ids[: max(len(gen_ids) - 1, 0)]
        caps = await self.capture(processed, points, steering_spec=steering_spec)
        return completion, caps

    async def capture_attention(self, prompt_token_ids: Any, layers: Any) -> dict[int, dict[str, torch.Tensor]]:
        """Attention scores, probs and per-head values. See
        :meth:`interp_engine.protocol.InterpModel.capture_attention`.

        Requires the model to have been loaded with eager attention, since the probabilities come
        from ``output_attentions=True`` -- the refusal that says so is raised by the capture.
        """
        from interp_engine.capture import capture_attention_eager

        return capture_attention_eager(self, [int(t) for t in prompt_token_ids], layers)

    def _generate_completion(
        self, prompt_ids: list[int], max_tokens: int, temperature: float, seed: int | None
    ) -> Completion:
        from interp_engine.steer import generate_stream

        steps = list(
            generate_stream(
                self,
                torch.tensor([prompt_ids], device=self.device),
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
            )
        )
        return Completion(
            text="".join(s.token_str for s in steps),
            token_ids=[s.token_id for s in steps],
        )

    async def generate_text(
        self,
        prompt_token_ids: Any,
        *,
        max_tokens: int = 200,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> str:
        """Generate and return the completion text. See the protocol."""
        return self._generate_completion([int(t) for t in prompt_token_ids], max_tokens, temperature, seed).text

    async def generate_stream(
        self,
        prompt_token_ids: Any,
        *,
        max_tokens: int = 200,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield decoded text deltas, one per token. See the protocol.

        Shadows the module-level ``interp_engine.generate_stream`` free function by design:
        that one yields rich ``GenStep``s (logits, logprobs) and is what you want on eager
        specifically, while this yields the plain text deltas both backends agree on.
        """
        from interp_engine.steer import generate_stream

        for step in generate_stream(
            self,
            torch.tensor([[int(t) for t in prompt_token_ids]], device=self.device),
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
        ):
            yield step.token_str

    async def decode_residuals(self, residuals: torch.Tensor, *, detach: bool = True) -> torch.Tensor:
        """Decode residuals to logits with this model's own post-unembed arithmetic. See the protocol.

        The free function ``interp_engine.decode_residuals`` returns RAW logits and takes ``softcap``
        and ``multiplier`` explicitly; this reads the model's own ``final_logit_softcapping`` and
        :attr:`logit_multiplier` so that it agrees with the vLLM backend, which applies both inside
        its unembed.
        """
        from interp_engine.lens import decode_residuals

        return decode_residuals(
            self,
            residuals,
            softcap=self.final_logit_softcapping,
            detach=detach,
            multiplier=self.logit_multiplier,
        )

    @property
    def final_logit_softcapping(self) -> float | None:
        """The model's configured logit softcap (Gemma-2), or None when it has none."""
        cfg: Any = self.config
        if hasattr(cfg, "get_text_config"):
            cfg = cfg.get_text_config()
        cap = getattr(cfg, "final_logit_softcapping", None)
        return float(cap) if cap is not None else None

    @property
    def logit_multiplier(self) -> float | None:
        """The scalar this family multiplies its logits by after ``lm_head``, or None for unity.

        Cohere's ``logit_scale``, Granite's ``logits_scaling``, Falcon-H1's ``lm_head_multiplier`` and
        LLaDA's ``scale_logits``, normalized to one multiply. Read from ``arch.quirks`` rather than the
        config, so a family simulated by replacing ``quirks`` (which is how a structure gets tested on
        an already-loaded model) is answered consistently with every other point resolution.
        """
        return self.arch.quirks.logit_multiplier
