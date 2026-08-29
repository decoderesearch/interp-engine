"""Build architecture families with no checkpoint, so resolution is testable where it lives.

The broad cross-architecture audit lives in ``interp-engine-validator``, over vLLM's registry
snapshot. This module is the engine's own much smaller version of the same idea, and it exists
because the split left this repo with no test that touches the families whose *resolution* it is
responsible for: a change to ``arch.py`` or ``model.resolve_point`` could regress them and only the
other repo would notice.

Two ways to build a family without downloading weights, and they answer different questions.

:func:`build_on_meta` -- **full size, no storage, structure only.** A meta tensor allocates nothing,
so a family builds in milliseconds at its config class's own defaults. That proves the part that
actually breaks: resolution walks the *module tree* and matches *attribute names*
(``facts.ATTN_ATTRS`` and friends), and a meta model's tree and names are the real ones. It proves
nothing about numerics, since there are no numbers.

Nothing is shrunk here, deliberately, and two families show why: shrinking Zamba2 fails its
``layer_types`` strict-dataclass validation outright (``num_hidden_layers`` must equal
``len(layer_types)``), and shrinking LongcatFlash fails inside attention on a rope dimension
mismatch. A shrink that "works" is worse than one that raises -- a shorter trunk truncates the
``layer_types`` pattern a hybrid family validates against, or drops the only attention layer out of a
mostly-SSM trunk, and then the probe reports a gap that is the probe's own fault.

:func:`shrunk_deepseek_v4` -- **tiny, real weights, on CPU, so numerics are checkable.** DeepSeek-V4
is the one family here whose trunk is homogeneous enough to shrink, and shrinking it is worth a lot:
the real checkpoints are 160 GB (V4-Flash) and 865 GB (V4-Pro), while this is ~0.6M parameters and
runs a forward in about a second. Parity is engine-against-raw-HF on *identical* weights, so every
question about the stream coordinate -- does ``Address("resid_post", 5, stream=2)`` return exactly
``hidden_states[:, :, 2, :]`` -- is weight-independent and answerable here rather than on a rented
8xH200. Random weights are not a limitation for that: the claim under test is which tensor an address
names, not what the model believes.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn as nn

#: Families this repo is responsible for resolving and that no checkpoint in the fast suite covers.
#: Each is a live gap or a recently-fixed one, so the list is the regression surface rather than a
#: sample -- see the plan's ``KNOWN_GAPS`` discussion.
UNCHECKPOINTED_FAMILIES: tuple[str, ...] = (
    "DeepseekV4ForCausalLM",
    "LongcatFlashForCausalLM",
    "Zamba2ForCausalLM",
    "HrmTextForCausalLM",
)


class NoTokenizer:
    """``EagerModel`` requires a tokenizer; nothing here tokenizes, and loading one is network.

    Deliberately not a stub with methods: if a test ever grows a step that tokenizes, an
    ``AttributeError`` naming this class is a better outcome than a result derived from a fake.
    """


def hf_class_for(arch: str) -> Any | None:
    """The ``transformers`` class for an architecture name, or ``None`` when it ships none."""
    import transformers

    return getattr(transformers, arch, None)


def build_on_meta(arch: str) -> nn.Module:
    """A meta-device instance of ``arch``, full size, from its config class's own defaults."""
    hf_class = hf_class_for(arch)
    if hf_class is None:
        raise LookupError(f"transformers has no class named {arch!r}")
    config = hf_class.config_class()
    config.architectures = [arch]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.device("meta"):
            return hf_class(config)


def eager_on_meta(arch: str) -> Any:
    """``build_on_meta`` wrapped in an ``EagerModel``, which is what resolution is tested through.

    ``device=None`` so nothing is moved (a meta tensor cannot be), and the architecture name stands
    in for the repo id -- no config is read by id when both model and tokenizer are supplied.
    """
    from interp_engine import EagerModel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return EagerModel(arch, hf_model=build_on_meta(arch), tokenizer=NoTokenizer(), device=None)


#: Dims for :func:`shrunk_deepseek_v4`. Every field that scales a weight is reduced; every field
#: that changes the *shape of the computation* is left at its default. ``hc_mult`` is the one that
#: matters most -- it stays 4, because the number of residual streams is the thing under test.
_DSV4_SHRUNK: dict[str, int] = {
    "hidden_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "head_dim": 64,
    "vocab_size": 512,
    "moe_intermediate_size": 64,
    "n_routed_experts": 4,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "q_lora_rank": 32,
    "o_lora_rank": 32,
    "o_groups": 2,
    "index_n_heads": 4,
    "index_head_dim": 32,
    "index_topk": 16,
    "qk_rope_head_dim": 16,
    "num_nextn_predict_layers": 1,
    "max_position_embeddings": 512,
}


def shrunk_deepseek_v4(*, seed: int = 0, dtype: torch.dtype = torch.float32) -> tuple[nn.Module, Any]:
    """A tiny DeepSeek-V4 with real random weights on CPU, plus its config.

    float32 by default: this exists to be compared against raw HF exactly, and bf16 round-off would
    put a tolerance between the two for no reason. V4 is eager-only upstream (its ``head_dim=512``
    exceeds every FlashAttention head-dim cap, and SDPA drops the learned sink term), so no
    ``attn_implementation`` is passed -- there is nothing to choose.
    """
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

    config = DeepseekV4Config(**_DSV4_SHRUNK)
    config.architectures = ["DeepseekV4ForCausalLM"]
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DeepseekV4ForCausalLM(config).to(dtype).eval()
    return model, config


def eager_shrunk_deepseek_v4(**kwargs: Any) -> Any:
    """:func:`shrunk_deepseek_v4` wrapped in an ``EagerModel``, ready to capture from."""
    from interp_engine import EagerModel

    model, _ = shrunk_deepseek_v4(**kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return EagerModel(
            "DeepseekV4ForCausalLM",
            hf_model=model,
            tokenizer=NoTokenizer(),
            device=None,
            dtype="float32",
        )


def motif_spelled_deepseek_v4(**kwargs: Any) -> Any:
    """A tiny DeepSeek-V4 whose mHC modules are renamed to Motif 3's spelling.

    A chimera, and it answers one question only: does the resolver take its addresses from the layout
    the *block* identifies? Motif 3 is the second family with a hyper-connection trunk, its smallest
    checkpoint is 315B, and its module layout differs from V4's by exactly enough to swap two tensors
    of the same shape -- so this pins which module and which side each point resolves to, and claims
    nothing about the numbers, which are V4's here.

    The layout's own provenance is separate and empirical: each coefficient tensor was recomputed
    from ``MHCLayer``'s projections on a tiny random-weight Motif built from its shipped
    ``modeling_motif.py`` and matched exactly, as was the collapse against its pre-sublayer norm's
    input. See the comment above :data:`interp_engine.facts.HYPER_CONNECTION_LAYOUTS`.
    """
    model = eager_shrunk_deepseek_v4(**kwargs)
    for block in model.arch.decoder_layers:
        block.mhc_attn, block.mhc_ffn = block.attn_hc, block.ffn_hc
        del block.attn_hc
        del block.ffn_hc
    return model


#: Dims for :func:`kv_shared_gemma4_on_meta`. Small everywhere it can be, because nothing here runs a
#: forward -- except ``num_hidden_layers``, which has to span two full periods of Gemma-4's 5:1
#: sliding-to-full pattern. Sharing is per layer *type*, so with only one full-attention layer the
#: shared one would have no source to name and the fixture would be testing that instead.
_GEMMA4_SHARED: dict[str, int] = {
    "num_hidden_layers": 12,
    "num_kv_shared_layers": 2,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 99,
}


def kv_shared_gemma4_on_meta() -> Any:
    """A meta-device Gemma-4 whose top layers reuse an earlier layer's keys/values, as an ``EagerModel``.

    Gemma-4's default config shares nothing, so the structure that matters here -- a layer built with
    no ``k_proj`` and no ``k_norm``, but with a ``q_norm`` of its own -- only appears once
    ``num_kv_shared_layers`` is set. The released checkpoints all set it; they are also gated and tens
    of gigabytes, so this builds the same module tree from transformers' own modeling code with no
    weights, which is all a structural question needs.
    """
    from interp_engine import EagerModel

    hf_class = hf_class_for("Gemma4ForCausalLM")
    if hf_class is None:
        raise LookupError("transformers has no Gemma4ForCausalLM")
    config = hf_class.config_class(**_GEMMA4_SHARED)
    config.architectures = ["Gemma4ForCausalLM"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.device("meta"):
            model = hf_class(config)
        return EagerModel("Gemma4ForCausalLM", hf_model=model, tokenizer=NoTokenizer(), device=None)


#: Dims for :func:`moe_gemma4_on_meta`. ``enable_moe_block`` and ``attention_k_eq_v`` are the subject
#: and the rest is as small as the modeling code accepts. ``num_hidden_layers`` spans one period of
#: Gemma-4's 5:1 sliding-to-full pattern so the fixture holds both kinds of attention layer: the flag
#: is model-wide but applies only to the full-attention ones, and a fixture of one kind could not tell
#: a per-layer answer from a per-model one.
_GEMMA4_MOE: dict[str, Any] = {
    "num_hidden_layers": 6,
    "hidden_size": 64,
    "intermediate_size": 128,
    "moe_intermediate_size": 32,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_global_key_value_heads": 1,
    "head_dim": 16,
    "vocab_size": 99,
    "enable_moe_block": True,
    "num_experts": 8,
    "top_k_experts": 2,
    "attention_k_eq_v": True,
}


def moe_gemma4_on_meta() -> Any:
    """A meta-device Gemma-4 whose layers hang experts beside the dense MLP, as an ``EagerModel``.

    Three structures no other family has, all of them switched on by config flags the default
    ``Gemma4TextConfig`` leaves off, and all three answered by the module tree with no weights:

    - ``enable_moe_block`` builds ``router``/``experts`` as *siblings* of ``layer.mlp``, so the MLP is
      one of two branches the block's forward sums;
    - the same flag keeps the dense MLP, so a sparse layer here still has a neuron basis;
    - ``attention_k_eq_v`` builds the full-attention layers with ``v_proj = None``, taking the key
      projection's output as the value.

    The checkpoint that has all three is the 26B, which is gated and 50-odd gigabytes. This is
    transformers' own modeling code at the same flags, which is what a structural question needs.
    """
    from interp_engine import EagerModel

    hf_class = hf_class_for("Gemma4ForCausalLM")
    if hf_class is None:
        raise LookupError("transformers has no Gemma4ForCausalLM")
    config = hf_class.config_class(**_GEMMA4_MOE)
    config.architectures = ["Gemma4ForCausalLM"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.device("meta"):
            model = hf_class(config)
        return EagerModel("Gemma4ForCausalLM", hf_model=model, tokenizer=NoTokenizer(), device=None)


#: Dims for :func:`shrunk_lfm2_moe`. ``layer_types`` is the point of the fixture and is spelled out
#: rather than shrunk: LFM2 interleaves short-convolution blocks with attention ones, and a fixture
#: with only the second kind would not have the block whose shape is under test. ``num_dense_layers``
#: stays 1 so layer 0 is dense and the later blocks are routed, matching the released checkpoints,
#: where every sampled layer happened to be a conv.
_LFM2_MOE_SHRUNK: dict[str, Any] = {
    "hidden_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 64,
    "moe_intermediate_size": 32,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "num_dense_layers": 1,
    "vocab_size": 256,
    "max_position_embeddings": 128,
    "layer_types": ["conv", "full_attention", "conv", "conv"],
}


#: Dims for :func:`shrunk_gpt_oss`. Every layer is sparse, as the released checkpoints are, and
#: ``num_experts_per_tok`` is spelled out because it *is* the fixture's subject: the top-k is what a
#: rebuilt routing decision has to get from the config.
_GPT_OSS_SHRUNK: dict[str, Any] = {
    "hidden_size": 64,
    "intermediate_size": 32,
    "head_dim": 16,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
    "vocab_size": 256,
    "max_position_embeddings": 128,
}


def shrunk_gpt_oss(*, seed: int = 0) -> Any:
    """A tiny real gpt-oss with float32 weights on CPU, as an ``EagerModel``.

    The family whose MoE block a quantizer replaces the ``forward`` of, at a size that builds in under a
    second against a 13 GB checkpoint. Real because the block's own arithmetic is what makes the fixture
    usable: ``GptOssDecoderLayer`` does ``hidden_states, _ = self.mlp(hidden_states)``, so a test can
    swap in an inline-routing ``forward`` that returns ``(out, router_logits)`` -- exactly the shape
    transformers' MXFP4 loader installs -- and the model still runs a real forward pass around it.
    """
    from transformers import GptOssConfig, GptOssForCausalLM

    from interp_engine import EagerModel

    config = GptOssConfig(**_GPT_OSS_SHRUNK)
    config.architectures = ["GptOssForCausalLM"]
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = GptOssForCausalLM(config).to(torch.float32).eval()
        return EagerModel("GptOssForCausalLM", hf_model=model, tokenizer=NoTokenizer(), device=None, dtype="float32")


#: Dims for :func:`shrunk_granite_moe`. ``num_local_experts`` and ``num_experts_per_tok`` are kept
#: far apart, because the fixture's subject is telling a ``[tokens, k]`` tensor from a
#: ``[tokens, n_experts]`` one and equal counts would make the two indistinguishable.
_GRANITE_MOE_SHRUNK: dict[str, Any] = {
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
    "vocab_size": 128,
    "max_position_embeddings": 64,
}


def shrunk_granite_moe(*, seed: int = 0) -> Any:
    """A tiny real GraniteMoE with float32 weights on CPU, as an ``EagerModel``.

    The family whose router returns the routing decision *backwards*:
    ``GraniteMoeTopKRouter.forward`` hands back ``(top_k_index, top_k_weights, router_logits)`` where
    every other family leads with the logits. Real rather than a stub with the same signature,
    because a stub would only restate the order this fixture exists to check against transformers.
    """
    from transformers import GraniteMoeConfig, GraniteMoeForCausalLM

    from interp_engine import EagerModel

    config = GraniteMoeConfig(**_GRANITE_MOE_SHRUNK)
    config.architectures = ["GraniteMoeForCausalLM"]
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = GraniteMoeForCausalLM(config).to(torch.float32).eval()
        return EagerModel(
            "GraniteMoeForCausalLM", hf_model=model, tokenizer=NoTokenizer(), device=None, dtype="float32"
        )


#: Dims for :func:`shrunk_opt`. Nothing here is unusual -- the fixture's subject is not a dimension
#: but a reshape in the block's own forward, which every OPT has at every size.
_OPT_SHRUNK: dict[str, Any] = {
    "hidden_size": 32,
    "ffn_dim": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "word_embed_proj_dim": 32,
    "vocab_size": 128,
    "max_position_embeddings": 64,
}


def shrunk_opt(*, seed: int = 0) -> Any:
    """A tiny real OPT with float32 weights on CPU, as an ``EagerModel``.

    The family that flattens mid-block: ``OPTDecoderLayer`` reshapes its hidden states to
    ``(batch * seq, d_model)`` before the feed-forward and back afterwards, so its MLP points arrive
    without a batch axis while its attention points keep one. Real and runnable because the claim is
    about the shape a forward pass actually produces, and 20k parameters is enough to produce it.
    """
    from transformers import OPTConfig, OPTForCausalLM

    from interp_engine import EagerModel

    config = OPTConfig(**_OPT_SHRUNK)
    config.architectures = ["OPTForCausalLM"]
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = OPTForCausalLM(config).to(torch.float32).eval()
        return EagerModel("OPTForCausalLM", hf_model=model, tokenizer=NoTokenizer(), device=None, dtype="float32")


#: Dims for :func:`shrunk_phi3`. The subject is the MLP's single ``gate_up_proj``, which every Phi-3
#: has at every size, so the only thing that matters here is that ``intermediate_size`` is even --
#: it is the width the two branches are cut out of.
_PHI3_SHRUNK: dict[str, Any] = {
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "vocab_size": 128,
    "max_position_embeddings": 64,
    # Phi-3's defaults point these at the released tokenizer's 32000, which is past the end of a
    # vocabulary this size -- the embedding refuses to build rather than the config being adjusted.
    "pad_token_id": 0,
    "eos_token_id": 1,
    "bos_token_id": 1,
}


def shrunk_phi3(*, seed: int = 0) -> Any:
    """A tiny real Phi-3 with float32 weights on CPU, as an ``EagerModel``.

    The family that fuses its two pre-activation projections into one ``gate_up_proj``. Real and
    runnable because the claim under test is an identity between three captured tensors --
    ``act(mlp_pre) * mlp_pre_linear == mlp_act`` -- and that is what says the halves were cut the
    way the block's own forward cuts them, rather than swapped.
    """
    from transformers import Phi3Config, Phi3ForCausalLM

    from interp_engine import EagerModel

    config = Phi3Config(**_PHI3_SHRUNK)
    config.architectures = ["Phi3ForCausalLM"]
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Phi3ForCausalLM(config).to(torch.float32).eval()
        return EagerModel("Phi3ForCausalLM", hf_model=model, tokenizer=NoTokenizer(), device=None, dtype="float32")


def shrunk_lfm2_moe(*, seed: int = 0) -> Any:
    """A tiny LFM2-MoE with real float32 weights on CPU, as an ``EagerModel``.

    Its conv blocks are the structure at issue: a short causal convolution stands where attention
    would, and the block is otherwise sequential (mix, add, ``ffn_norm``, feed-forward, add), so
    `resid_mid` exists there even though no attention does. float32 and real weights because the claim
    is an equality between a captured tensor and a d_model-wide sum. 166k parameters, so it builds in
    under a second and needs no download -- the released checkpoint is 16 GB.
    """
    from transformers import Lfm2MoeConfig, Lfm2MoeForCausalLM

    from interp_engine import EagerModel

    config = Lfm2MoeConfig(num_hidden_layers=len(_LFM2_MOE_SHRUNK["layer_types"]), **_LFM2_MOE_SHRUNK)
    config.architectures = ["Lfm2MoeForCausalLM"]
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Lfm2MoeForCausalLM(config).to(torch.float32).eval()
        return EagerModel("Lfm2MoeForCausalLM", hf_model=model, tokenizer=NoTokenizer(), device=None, dtype="float32")
