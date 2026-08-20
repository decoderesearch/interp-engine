"""Three arithmetic identities, checked on a tiny random model of every family the audit can build.

``tests/family_coverage.py`` proves that a point *resolves* -- that the engine found a module and can
hook it. That is the cheap half. It cannot tell whether the module it found is the right one, and the
failures that matter most are exactly the ones where it is not: a plausible tensor of the correct shape
and a believable magnitude, captured from one sublayer over. BLOOM's ``mlp_out`` was the whole residual
stream; GLM-4's ``attn_out_post`` was the next sublayer's normed input. Nothing about either looked
wrong.

So each family is asked to satisfy identities that only hold if the bindings are right:

- **DFA** -- ``probs @ value == z``. Ties the attention points together, and is the one that catches a
  wrong fused-QKV layout, a mis-split head, or a value the family scaled after projecting it.
- **residual decomposition** -- ``resid_pre + attn_out_post + mlp_out_post == resid_post``. Pins what
  the two ``*_out_post`` points mean, which is where post-sublayer norms and residual multipliers hide.
- **neuron basis** -- ``down_proj(mlp_act) == mlp_out``. Pins the MLP internals an SAE is trained on.

**Random weights, tiny configs, no downloads.** The identities are structural, so they hold or fail on
random weights exactly as they would on trained ones -- and a family's own config class supplies the
shapes, so nothing here is a checkpoint this suite has to fetch. What that cannot see is anything a
*trained* config turns on: Granite's ``residual_multiplier`` defaults to 1.0 and is 0.22 on the released
checkpoints, so the decomposition passes here either way. The engine reads such values from the config
rather than assuming the default, and :func:`test_a_multiplier_scales_the_contribution` sets one to
prove it.

**A refusal counts as a pass.** Not every family has all three quantities: a sparse MoE block has no
single down projection, MLA has no value to hook, a Mamba layer has no attention. The engine is
supposed to say so, with a reason. What this suite refuses to accept is a *number* that is wrong, or an
exception from three frames deeper that names nothing. :data:`EXCEPTIONS` records the families where an
identity genuinely cannot hold, each with why.
"""

from __future__ import annotations

import contextlib
import dataclasses
import warnings
from collections.abc import Callable
from typing import Any

import family_coverage as fc
import torch

# Shapes small enough to build in milliseconds, and every field that has to agree with another one.
# `head_dim` is 64 to match what several families' rope defaults assume (GPT-J's `rotary_dim` is 64, and
# a narrower head leaves a zero-width rotary section that fails inside the forward); the expert counts
# are here because a family whose default is 256 experts is several GB before anything runs.
TINY: dict[str, Any] = {
    "hidden_size": 256,
    "n_embd": 256,
    "d_model": 256,
    "num_attention_heads": 4,
    "n_head": 4,
    "n_heads": 4,
    "head_dim": 64,
    "num_key_value_heads": 2,
    "num_kv_heads": 2,
    "n_head_kv": 2,
    "kv_n_heads": 2,
    "num_hidden_layers": 2,
    "n_layer": 2,
    "n_layers": 2,
    "intermediate_size": 512,
    "n_inner": 512,
    "expansion_ratio": 2,
    "vocab_size": 128,
    "num_experts": 4,
    "num_local_experts": 4,
    "n_routed_experts": 4,
    "moe_num_experts": 4,
    "num_experts_per_tok": 2,
    # Group-limited routing (DeepSeek-V3 and the families that copied it) partitions the experts into
    # `n_group` groups, and the partition is integer division: 4 experts over the default 8 groups is
    # a group of zero, which fails inside the gate rather than at config time. One group of four is
    # the same routing the validator would get with the feature off.
    "n_group": 1,
    "topk_group": 1,
    "moe_intermediate_size": 128,
    "moe_ffn_hidden_size": 128,
    "shared_expert_intermediate_size": 128,
}

# Parameter budget, measured on `meta` before anything is materialized. A handful of families multiply
# some field of their own by a default this table does not shrink and land in the tens of billions.
BUDGET = 80_000_000

INPUT_IDS = torch.tensor([[3, 7, 11, 5, 9]])
TOLERANCE = 1e-4

INVARIANTS: tuple[str, ...] = ("dfa", "residual", "neuron")


# Families whose configs do not survive being miniaturized, so the validator cannot reach them at all.
# Not a support verdict: it says the *config class defaults* and these overrides do not compose, which
# is why each reason names the field. They are covered structurally by `family_coverage.audit()` and
# would need a real checkpoint to check numerically.
NOT_MINIATURIZABLE: dict[str, str] = {
    "Gemma3nForCausalLM": "its `layer_types` list is per-layer and validated against the full 35-layer "
    "pattern, so a 2-layer config fails inside __init__",
    "Gemma4ForCausalLM": "same per-layer validation as Gemma-3n, plus a kv-sharing index that has to "
    "point at a layer that exists",
    "InklingForConditionalGeneration": "the multimodal wrapper sizes its vision tower from fields this "
    "table does not shrink, so the tiny build is still ~1.7T parameters",
    "LongcatFlashForCausalLM": "its block count is derived, and a 2-layer config leaves the derived value at 0",
    "Mamba2ForCausalLM": "`num_heads` for the SSM is independent of `num_attention_heads` and the "
    "default does not divide this hidden size",
    "Zamba2ForCausalLM": "`layer_types` is a validated dataclass field whose default pattern is longer "
    "than a 2-layer trunk",
}

# Where an identity cannot hold on a family even with everything bound correctly. Each entry is a fact
# about the architecture, not a to-do: the point that would make it hold does not exist there.
EXCEPTIONS: dict[tuple[str, str], str] = {
    ("FalconH1ForCausalLM", "residual"): "a Mamba mixer runs alongside attention and adds into the same "
    "residual, so attention and the MLP do not account for the whole block; there is no `ssm_out` point "
    "to complete the sum with",
}


@dataclasses.dataclass
class Result:
    """One invariant on one family: ``held``, ``refused`` (with the engine's reason), or a mismatch."""

    invariant: str
    outcome: str  # "held" | "refused" | "mismatch" | "error"
    detail: str = ""

    @property
    def acceptable(self) -> bool:
        return self.outcome in ("held", "refused")


def config_for(hf_class: Any, arch: str) -> Any:
    """A tiny config for ``arch``, keeping only the fields its own config class declares."""
    probe = hf_class.config_class()
    fields = {
        name: value
        for name, value in TINY.items()
        # A read-only property (Falcon's `head_dim`) is derived from other fields and cannot be set.
        if hasattr(probe, name) and not isinstance(getattr(type(probe), name, None), property)
    }
    config = hf_class.config_class(**fields, pad_token_id=None)
    config.architectures = [arch]
    config._attn_implementation = "eager"
    return config


def tiny_model(arch: str) -> Any:
    """An ``EagerModel`` wrapping a tiny randomly initialized model of ``arch``.

    Tries grouped-query attention first, since that exercises the kv-repeat path, then falls back to
    one kv head per query head -- the MLA families (DeepSeek, MiniCPM3, GLM-MoE) derive their latent
    widths from the head count and do not survive an independent kv count.
    """
    from interp_engine import EagerModel

    found = fc.hf_class_for(arch)
    if found is None:
        raise RuntimeError(f"{arch} has no transformers class to build")
    _, hf_class = found

    errors: list[str] = []
    for grouped in (True, False):
        config = config_for(hf_class, arch)
        if not grouped and getattr(config, "num_key_value_heads", None):
            config.num_key_value_heads = config.num_attention_heads
        with torch.device("meta"):
            params = sum(p.numel() for p in hf_class(config).parameters())
        if params > BUDGET:
            raise RuntimeError(f"{arch} is {params / 1e6:.0f}M parameters even shrunk (budget {BUDGET / 1e6:.0f}M)")
        torch.manual_seed(0)
        try:
            model = hf_class(config).eval()
            # A config that builds but cannot run is no use to the checks below. `use_cache=False`
            # because a hybrid trunk asks its cache for a sequence length it has no cache to ask.
            model(INPUT_IDS, use_cache=False)
        except Exception as exc:  # noqa: BLE001 - reported, not handled
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        return EagerModel(arch, hf_model=model, tokenizer=_NoTokenizer(), device="cpu")
    raise RuntimeError(f"{arch} did not build tiny: {' / '.join(errors)}")


class _NoTokenizer:
    """`EagerModel` requires a tokenizer; nothing here tokenizes, and loading one would be network."""


# The activation a family applies to a standalone attention-output gate. Family knowledge, and it lives
# here rather than in the engine because the engine cannot read it: the block calls `torch.sigmoid`
# inline, so there is no module to inspect. Used to check that the module `attn_gate` resolves to is
# really the gate -- if it were not, applying the activation would not recover `z`.
GATE_ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "AfmoeForCausalLM": torch.sigmoid,
    "LagunaForCausalLM": torch.nn.functional.softplus,
}


def _first_layer_with_both_sublayers(model: Any) -> int:
    """A layer that has attention *and* a feed-forward, for the residual decomposition.

    Layer 0 is not always one: a hybrid trunk starts with a Mamba block, and Nemotron-H gives each
    layer a single sublayer. Falls back to 0 so the check still runs (and refuses) where none exists.
    """
    # Decided by whether the two points resolve, not by a sublayer predicate first: resolving *is* the
    # requirement, and the predicate that used to pre-filter this loop asked a broader question than the
    # name suggested (a convolutional mixer is a position mixer with no `attn_out` at all).
    for layer in range(model.arch.n_layers):
        with contextlib.suppress(Exception):
            model.resolve_point("attn_out_post", layer)
            model.resolve_point("mlp_out_post", layer)
            return layer
    return 0


def _first_dense_mlp_layer(model: Any) -> int:
    """A layer whose MLP is dense, for the neuron-basis check: a sparse block has no single projection.

    Many MoE families keep a dense prefix, so this reaches the neuron basis on families that would
    otherwise only ever refuse.
    """
    for layer in range(model.arch.n_layers):
        if not model.arch.is_moe_layer(layer) and model.arch.has_mlp_module(layer):
            return layer
    return 0


def check_dfa(model: Any) -> Result:
    """``probs @ value == z``: the attention points describe one consistent computation."""
    from interp_engine import per_head_value, run_with_cache

    layers = model.arch.softmax_attention_layers()
    if not layers:
        return Result("dfa", "refused", "no softmax-attention layer: nothing computes probabilities")
    layer = layers[0]
    cache = run_with_cache(model, INPUT_IDS, [("attn_probs", layer), ("value", layer), ("z", layer)])
    probs = cache.get("attn_probs", layer).float()
    z = cache.get("z", layer).float()
    value = per_head_value(model, cache, layer).float()
    if value.shape[2] != model.n_heads:  # GQA: each kv head serves several query heads
        value = value.repeat_interleave(model.n_heads // value.shape[2], dim=2)
    rebuilt = torch.einsum("bhqk,bkhd->bqhd", probs, value).reshape(*z.shape)

    if model.arch.quirks.gated_attn_out:
        rebuilt = _apply_gate(model, cache, layer, rebuilt, z)
    return _compare("dfa", rebuilt, z)


def _apply_gate(model: Any, cache: Any, layer: int, rebuilt: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """``z`` is post-gate on a gated-attention family, so the reconstruction has to be gated too."""
    from interp_engine import attn_out_gate, run_with_cache

    activation = GATE_ACTIVATIONS.get(model.arch.architecture)
    gate_cache = run_with_cache(model, INPUT_IDS, [("attn_gate", layer)])
    if activation is None:
        # The gate is packed into a double-width `q_proj`, a layout the engine owns end to end.
        gate = attn_out_gate(model, gate_cache, layer)
        return (rebuilt.reshape(*gate.shape) * gate).reshape(*z.shape)
    raw = activation(gate_cache.get("attn_gate", layer).float())
    head_dim = model.arch.value_head_dim_for_layer(layer)
    per_head = rebuilt.reshape(*z.shape[:2], model.n_heads, head_dim)
    # One scalar per head, or one per element of each head; both spellings exist.
    gate = raw.unsqueeze(-1) if raw.shape[-1] == model.n_heads else raw.reshape_as(per_head)
    return (per_head * gate).reshape(*z.shape)


def check_residual(model: Any) -> Result:
    """``resid_pre + attn_out_post + mlp_out_post == resid_post``: the two contributions are complete."""
    from interp_engine import run_with_cache

    layer = _first_layer_with_both_sublayers(model)
    names = ("resid_pre", "attn_out_post", "mlp_out_post", "resid_post")
    cache = run_with_cache(model, INPUT_IDS, [(name, layer) for name in names])
    got = {name: cache.get(name, layer).float() for name in names}
    rebuilt = got["resid_pre"] + got["attn_out_post"] + got["mlp_out_post"]
    return _compare("residual", rebuilt, got["resid_post"])


def check_neuron(model: Any) -> Result:
    """``down_proj(mlp_act) == mlp_out``: the neuron basis is the one the MLP actually uses."""
    from interp_engine import run_with_cache

    layer = _first_dense_mlp_layer(model)
    cache = run_with_cache(model, INPUT_IDS, [("mlp_act", layer), ("mlp_out", layer)])
    act, mlp_out = cache.get("mlp_act", layer), cache.get("mlp_out", layer)
    rebuilt = model.arch.mlp_projection(layer, "down")(act)
    return _compare("neuron", rebuilt.float(), mlp_out.float())


CHECKS: dict[str, Callable[[Any], Result]] = {
    "dfa": check_dfa,
    "residual": check_residual,
    "neuron": check_neuron,
}


def _compare(invariant: str, rebuilt: torch.Tensor, actual: torch.Tensor) -> Result:
    if rebuilt.shape != actual.shape:
        return Result(invariant, "mismatch", f"shape {tuple(rebuilt.shape)} != {tuple(actual.shape)}")
    diff = (rebuilt - actual).abs().max().item()
    scale = max(actual.abs().max().item(), 1.0)
    if diff <= TOLERANCE * scale:
        return Result(invariant, "held", f"max diff {diff:.3g}")
    return Result(invariant, "mismatch", f"max diff {diff:.3g} against a scale of {scale:.3g}")


def check(model: Any, invariant: str) -> Result:
    """Run one invariant, turning the engine's own refusal into a ``refused`` result.

    A ``ValueError`` from the engine is an answer: this family has no such quantity, and here is why.
    Anything else is a failure -- including an ``AttributeError``, which means a module was expected
    where the family has none and nobody said so.
    """
    try:
        return CHECKS[invariant](model)
    except ValueError as exc:
        return Result(invariant, "refused", str(exc).split(".")[0])
    except Exception as exc:  # noqa: BLE001 - reported as a failure, with its type
        return Result(invariant, "error", f"{type(exc).__name__}: {exc}")


def probed_architectures() -> list[str]:
    """Families the coverage audit could build, which are the ones the validator can miniaturize."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [row.arch for row in fc.audit() if row.status == "probed"]


def main() -> None:
    """Print the matrix: one line per family, one column per invariant."""
    warnings.simplefilter("ignore")
    counts: dict[str, int] = {}
    for arch in probed_architectures():
        if arch in NOT_MINIATURIZABLE:
            print(f"{arch:34s} skipped: {NOT_MINIATURIZABLE[arch][:60]}")
            counts["skipped"] = counts.get("skipped", 0) + 1
            continue
        try:
            model = tiny_model(arch)
        except Exception as exc:  # noqa: BLE001
            print(f"{arch:34s} BUILD {type(exc).__name__}: {str(exc)[:70]}")
            counts["unbuildable"] = counts.get("unbuildable", 0) + 1
            continue
        results = [check(model, name) for name in INVARIANTS]
        for result in results:
            counts[result.outcome] = counts.get(result.outcome, 0) + 1
        print(f"{arch:34s} " + " | ".join(f"{r.invariant}={r.outcome}" for r in results))
    print()
    for outcome, count in sorted(counts.items()):
        print(f"{outcome:12s} {count}")


if __name__ == "__main__":
    main()
