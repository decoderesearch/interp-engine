"""InterpEngine: standalone raw-transformers interpretability core.

Runs the plain HuggingFace model in eager PyTorch and captures/steers via our own
forward-hook layer. Eager (unfused) execution is what lets us read per-head attention
patterns, tap arbitrary hook points, and match reference numerics; serving is forward-only.
Keyed by the raw HuggingFace model id.
"""

from importlib.metadata import PackageNotFoundError, version

from interp_engine.address import (
    Address,
    AddressError,
    UnknownCoordinate,
    format_address,
    parse_address,
    to_address,
)
from interp_engine.arch import (
    ArchSpec,
    Quirks,
    resolve_arch,
    special_token_ids,
    special_token_positions,
)
from interp_engine.attn_config import unsupported_attn_config
from interp_engine.autograd_support import (
    BACKWARD_CAPABLE_ATTENTION_BACKENDS,
    GradientsUnsupported,
    GradSupport,
    eager_grad_support,
    vllm_grad_support,
)
from interp_engine.capture import (
    Cache,
    attn_out_gate,
    capture_attention,
    capture_generation,
    expert_assignment,
    head_contributions,
    is_rms_norm,
    per_head_value,
    pre_gain_normalized,
    rms_norm_parts,
    run_with_cache,
    split_fused_qkv,
)
from interp_engine.chat_compose import (
    ChatTurn,
    compose_assistant_turns,
    strip_wire_reasoning,
)
from interp_engine.chat_conventions import (
    REASONING_TAGS,
    TURN_END_TOKENS,
    ReasoningTags,
    added_vocab,
    detect_reasoning_tags,
    is_harmony,
)
from interp_engine.chat_formatters import (
    CODE_CHAT_FORMATS,
    ChatFormatter,
    ChatFormatterUnavailable,
    RenderedChat,
    resolve_chat_formatter,
)
from interp_engine.cuda_preflight import check_cuda_driver
from interp_engine.dispatch import CAPABILITIES, Capability, CapabilityUnsupported, TokensLike
from interp_engine.facts import rms_norm_eps_for_model
from interp_engine.hooks import HookManager
from interp_engine.lens import (
    apply_final_logit_softcap,
    apply_logit_transform,
    capture_residuals,
    decode_residuals,
    layer_logits,
)
from interp_engine.load import BACKENDS, VLLM_BACKENDS, load_model, vllm_installed
from interp_engine.mappers import (
    UnmappedHook,
    nnsight_accessor_to_point,
    point_to_nnsight_accessor,
    point_to_tlens_hook,
    tlens_hook_to_point,
    tlens_normalized_hook,
)
from interp_engine.model import EagerModel, HubKernelUnsupported, deepgemm_fallback_kwargs
from interp_engine.protocol import Completion, InterpModel, Point
from interp_engine.residual_basis import (
    RESIDUAL_POINTS,
    STREAM_REDUCTIONS,
    ResidualBasis,
    ResidualBasisUnsupported,
    eager_residual_basis,
    reduce_streams,
    vllm_residual_basis,
)
from interp_engine.select import BackendSelection, select_backend
from interp_engine.steer import (
    STEER_METHODS,
    GenStep,
    OrthogonalProjector,
    PositionMask,
    SteerMask,
    SteerSpec,
    generate_stream,
    projection_cap_delta,
    resolve_masked_positions,
    steer,
    steer_delta,
    steering_spec_to_eager_specs,
    top_logprobs,
    unit_vector,
)
from interp_engine.steer_specs import (
    AddSpec,
    LayerSteeringSpec,
    OrthogonalDecompSpec,
    ProjectionCapSpec,
    SteeringOp,
    SteeringSpec,
    steering_spec_to_worker_specs,
)
from interp_engine.sync import SyncModel, sync_model
from interp_engine.tokenize import (
    GeneratedTurnSpans,
    NoChatTemplateError,
    Tokenize,
    TokenSpan,
)
from interp_engine.vllm_backend import (
    VLLMModel,
    attn_capture_layers,
    is_linear_attention_layer,
    read_attn_dims,
    recompute_attn_from_payloads,
    sliding_window_for_layer,
)

# Only the pieces a caller driving their own vLLM engine needs: the point names, and the
# payload codecs for reading collective_rpc results back. The ``worker_*`` functions those
# RPCs run are NOT re-exported -- the supported way to reach them is by name through
# ``interp_engine.vllm_plugin``, and code that really wants the callables (to hand one to
# ``collective_rpc`` directly, which needs VLLM_ALLOW_INSECURE_SERIALIZATION) can import
# them from ``interp_engine.vllm_capture``. Same for the recompute internals.
from interp_engine.vllm_capture import (
    HOOK_CAPTURE_POINTS,
    decode_capture_payload,
    decode_tensor_payload,
    encode_tensor_payload,
)
from interp_engine.vllm_plugin import (
    WORKER_EXTENSION_CLS,
    InterpWorkerExtension,
    capture_engine_kwargs,
    native_extraction_engine_kwargs,
)

# Read from the installed distribution rather than hardcoded, so it cannot drift from
# pyproject.toml. Falls back for a source tree that was never installed.
try:
    __version__ = version("interp-engine")
except PackageNotFoundError:  # pragma: no cover -- source checkout, not installed
    __version__ = "0.0.0.dev0"

# Grouped by what you reach for, since alphabetical order buries the entry points among the
# helpers. Anything not listed here is reachable from its own module but is not API: it can
# change without a major version.
__all__ = [
    # The installed version, as `vllm` and `transformers` expose it.
    "__version__",
    # Loading a model -- start here.
    "BACKENDS",
    "VLLM_BACKENDS",
    "BackendSelection",
    "load_model",
    "select_backend",
    "vllm_installed",
    # The models, and the surface they share.
    "Completion",
    "EagerModel",
    "HubKernelUnsupported",
    "deepgemm_fallback_kwargs",
    "InterpModel",
    "VLLMModel",
    # That same surface without an event loop, for scripts and notebooks. The sync free
    # functions below dispatch through this, so most callers never name it.
    "SyncModel",
    "sync_model",
    # What one signature over two backends is allowed to do when only one of them can: refuse,
    # from a table, naming the capability. Never a warning and never a silent no-op.
    "CAPABILITIES",
    "Capability",
    "CapabilityUnsupported",
    "TokensLike",
    # Addressing a tensor: the type, its string form, and the refusals.
    "Address",
    "AddressError",
    "Point",
    "UnknownCoordinate",
    "format_address",
    "parse_address",
    "to_address",
    # Capture: one signature over either backend, dispatched on the model you pass.
    "Cache",
    "UnmappedHook",
    "attn_out_gate",
    "capture_attention",
    "capture_generation",
    "expert_assignment",
    "head_contributions",
    "is_rms_norm",
    "nnsight_accessor_to_point",
    "point_to_nnsight_accessor",
    "point_to_tlens_hook",
    "tlens_hook_to_point",
    "tlens_normalized_hook",
    "per_head_value",
    "pre_gain_normalized",
    "rms_norm_eps_for_model",
    "rms_norm_parts",
    "run_with_cache",
    "split_fused_qkv",
    # Gradients: whether they are available, and the refusal when they are not.
    "BACKWARD_CAPABLE_ATTENTION_BACKENDS",
    "GradSupport",
    "GradientsUnsupported",
    "eager_grad_support",
    "vllm_grad_support",
    # The residual stream's structure: how many, additive, sequenced -- and the refusals when not.
    "RESIDUAL_POINTS",
    "STREAM_REDUCTIONS",
    "ResidualBasis",
    "ResidualBasisUnsupported",
    "eager_residual_basis",
    "reduce_streams",
    "vllm_residual_basis",
    # Steering. Every method's arithmetic is one delta function, shared with the vLLM worker.
    "STEER_METHODS",
    "AddSpec",
    "LayerSteeringSpec",
    "OrthogonalDecompSpec",
    "OrthogonalProjector",
    "PositionMask",
    "ProjectionCapSpec",
    "SteerMask",
    "SteerSpec",
    "SteeringOp",
    "SteeringSpec",
    "projection_cap_delta",
    "resolve_masked_positions",
    "steer",
    "steer_delta",
    "steering_spec_to_eager_specs",
    "steering_spec_to_worker_specs",
    "unit_vector",
    # Generation (sync, eager).
    "GenStep",
    "generate_stream",
    "top_logprobs",
    # Lens read-out.
    "apply_final_logit_softcap",
    "apply_logit_transform",
    "capture_residuals",
    "decode_residuals",
    "layer_logits",
    # Architecture introspection.
    "ArchSpec",
    "HookManager",
    "Quirks",
    "attn_capture_layers",
    "read_attn_dims",
    "recompute_attn_from_payloads",
    "resolve_arch",
    "unsupported_attn_config",
    # Tokenization and chat templates.
    "CODE_CHAT_FORMATS",
    "ChatFormatter",
    "ChatFormatterUnavailable",
    "ChatTurn",
    "GeneratedTurnSpans",
    "NoChatTemplateError",
    "REASONING_TAGS",
    "ReasoningTags",
    "RenderedChat",
    "TURN_END_TOKENS",
    "TokenSpan",
    "Tokenize",
    "added_vocab",
    "compose_assistant_turns",
    "detect_reasoning_tags",
    "is_harmony",
    "resolve_chat_formatter",
    "special_token_ids",
    "special_token_positions",
    "strip_wire_reasoning",
    # Driving your own vLLM engine (see interp_engine.vllm_plugin).
    "HOOK_CAPTURE_POINTS",
    "InterpWorkerExtension",
    "WORKER_EXTENSION_CLS",
    "capture_engine_kwargs",
    "decode_capture_payload",
    "decode_tensor_payload",
    "encode_tensor_payload",
    "is_linear_attention_layer",
    "native_extraction_engine_kwargs",
    "sliding_window_for_layer",
    # Environment checks.
    "check_cuda_driver",
]
