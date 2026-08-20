"""Which model families the engine resolves points on, answered from configs alone.

**The list is vLLM's.** What interp-engine has to cover is what a user can serve, and the serving
runtime's registry is the only definition of that which stays current without anyone curating it. So
``vllm_supported_archs.json`` is a snapshot of ``vllm.model_executor.models.registry``, and the audit
here is over that list rather than over the handful of checkpoints the rest of the suite loads.

**No weights.** A family is built from its config class's own defaults on the ``meta`` device, which
allocates nothing, so the whole registry is probed in seconds and a gap is found by opening a config
rather than by someone hitting it on a 70B checkpoint. What that proves is exactly the part that
breaks: resolution walks the *module tree* and matches *attribute names* (``facts.ATTN_ATTRS`` and
friends), and a meta-device model's tree and names are the real ones. What it cannot prove is
numerics -- for that a point has to be captured on real weights, which is
``tests/test_qkv_layout.py`` and the cross-engine sweep.

**Network for kernels, though never for weights.** Eight state-space families (Mamba, Mamba2,
FalconMamba, Jamba, Zamba2, Falcon-H1, Nemotron-H, Granite-4-hybrid) resolve a ``causal-conv1d`` /
``mamba-ssm`` kernel from the hub *during* ``__init__``, and it is a **version** they ask for, which
``kernels`` cannot map to a revision from the cache -- so ``HF_HUB_OFFLINE=1`` fails for them even
after a successful online run, not just on a cold cache. They report ``needs_download`` rather than
joining ``not_buildable``, because these eight are exactly the trunks whose sublayer points are the
interesting ones and an offline run would otherwise look complete while skipping the hybrid fifth of
the registry. CI runs online.

Five outcomes per family, and two of them are claims about interp-engine:

- **probed** -- the family built, so every point below is either resolved or refused with a reason.
- **not_loadable** -- ``EagerModel`` could not bind the tree at all, so *nothing* resolves. Ours, and
  the most severe kind of gap.
- **no_transformers_class** -- the family's HF modeling code ships *with the checkpoint*
  (``trust_remote_code``) rather than with transformers, so there is no class to instantiate from a
  config and nothing to probe. **Not a support verdict.** The eager backend loads these
  (``EagerModel`` passes ``trust_remote_code=True``) and the vLLM backend resolves against vLLM's own
  implementation, which is in the installed package; both then depend on whether those trees use names
  the vocabularies know, and answering that would mean downloading either code or weights, which is
  what this audit is defined not to do.
- **not_buildable** -- the config class's defaults do not survive its own model's ``__init__`` (a rope
  field left ``None``, an attention sub-config missing ``rope_theta``). A transformers-side limit.
- **needs_download** -- construction reached the hub and could not. Not a verdict either way: rerun
  with ``HF_HUB_OFFLINE=0``, as CI does.

Refresh the snapshot after a vLLM upgrade (needs a venv with vLLM installed, not the test venv)::

    .venv-vllm/bin/python tests/family_coverage.py --refresh

and print the matrix (in the test venv, which needs only transformers)::

    .venv-cmp/bin/python tests/family_coverage.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import warnings
from collections.abc import Iterable, Sequence
from typing import Any

SNAPSHOT = pathlib.Path(__file__).with_name("vllm_supported_archs.json")

# Points every decoder-only family must resolve, because each is a *module boundary* that any
# transformer has: the block's input and output, the two sublayers' inputs and outputs, the attention
# output projection's input (`z`), and the three model-level points. A family that misses one of
# these is missing a module we could not name, which is the failure this audit exists to surface.
CORE_POINTS: tuple[str, ...] = (
    "embeddings",
    "resid_pre",
    "attn_in",
    "attn_out",
    "attn_out_post",
    "z",
    "mlp_in",
    "mlp_out",
    "mlp_out_post",
    "resid_post",
    "final_norm",
    "lm_head",
)

# Points that legitimately do not exist on some architectures, so their absence is reported and not
# asserted: `resid_mid` (none on a parallel block), `value` (none as a separable tensor under MLA),
# and the neuron basis (none at a block boundary on a sparse MLP, none per branch on a fused one).
CONDITIONAL_POINTS: tuple[str, ...] = ("resid_mid", "value", "mlp_pre", "mlp_pre_linear", "mlp_act")

# Probed on a softmax-attention layer rather than on layer 0, which on a hybrid trunk is usually a
# linear-attention or SSM mixer -- where these points *correctly* do not exist, and reporting that as
# a gap would bury the real ones.
_ATTENTION_POINTS = frozenset({"attn_in", "attn_out", "attn_out_post", "z", "value"})
_MODEL_LEVEL_POINTS = frozenset({"embeddings", "final_norm", "lm_head"})

# Not an exception message: reported when a trunk has no attending layer to probe at all, which is a
# fact about the family rather than a failure to resolve anything.
NO_ATTENTION_LAYERS = "no softmax-attention layer in this trunk"


# --- what does not hold, and which kind of "does not hold" it is ------------------------------
#
# Three tables, read both by ``test_family_coverage.py`` (which pins them: an entry that starts passing
# has to be deleted) and by ``models_status.py`` (which turns them into the documented support tiers).
# They live here rather than in either consumer so the doc and the assertions cannot disagree about
# which families are covered.

# Families whose module tree interp-engine cannot address, with the structural reason. Not name gaps:
# each of these is a shape the canonical points do not describe, so "fixing" it means deciding what a
# point should *mean* there, not adding a spelling.
KNOWN_GAPS: dict[str, str] = {
    "HrmTextForCausalLM": (
        "the model does not load: trunk discovery wants one `.layers`/`.h`/`.blocks` and finds two "
        "stacks side by side (`model.H_module.layers` / `model.L_module.layers`), so it raises "
        "`Could not locate transformer trunk` and every point is lost at once, before addressing is "
        "reached. The narrower reason this entry used to give -- that a layer index cannot name a "
        "position in a single pass -- no longer holds: flattened order counts re-entries, and "
        "`(H_cycles * (L_cycles + 1)) * num_layers_per_stack == num_hidden_layers == 128` on the "
        "default config, so the indices do line up. What is left is the loader and the re-entry "
        "support behind it; until then the flattening tripwire refuses rather than mis-indexing"
    ),
    "DeepseekV4ForCausalLM": (
        "the trunk carries `hc_mult` residual streams rather than one (manifold-constrained "
        "hyper-connections), so no tensor between the blocks is the residual stream the three resid "
        "points name -- see `test_a_hyper_connection_trunk_refuses_the_residual_points`. The engine "
        "can now address one (`Address('resid_post', 5, stream=2)`) and has family points for the "
        "collapse/write/mix tensors, but this probe asks for the bare point, which is still refused "
        "and should be: there is no default stream. vLLM serves all seven of those family points, "
        "measured on V4-Flash at layers 0/21/42, and still cannot address a stream through the resid "
        "points themselves -- which is a fact about the residual rather than about the wire. Capture "
        "`resid_streams` and index it instead. Note that vLLM's decoder layer also returns a "
        "(num_tokens, hc_mult, hidden_size) stack and it is NOT that point: vLLM defers each "
        "sublayer's write into the next sublayer's kernel, so the returned stack is the one the MLP "
        "read rather than the one the block produced -- resid_mid in stream form. The block's output "
        "stack reaches no boundary at all and is read off the next layer's kernel; see "
        "interp_engine.vllm_capture.mhc and interp_engine.residual_basis.vllm_residual_basis"
    ),
}

# Core points a family does not *have*, as opposed to cannot reach. A state-space block is a norm and
# a mixer: no attention, and no feed-forward either, so seven of the twelve core points describe parts
# it does not contain. These families still load, and the residual stream -- where interpretability
# work on them actually happens -- is capturable, which is the reason to cover them at all.
SUBLAYER_POINTS = frozenset({"attn_in", "attn_out", "attn_out_post", "z", "mlp_in", "mlp_out", "mlp_out_post"})
ATTENTION_POINTS_ONLY = frozenset({"attn_in", "attn_out", "attn_out_post", "z"})

ARCHITECTURAL_ABSENCES: dict[str, tuple[frozenset[str], str]] = {
    "MambaForCausalLM": (SUBLAYER_POINTS, "a MambaBlock is `norm` + `mixer`: neither sublayer exists"),
    "Mamba2ForCausalLM": (SUBLAYER_POINTS, "a Mamba2Block is `norm` + `mixer`: neither sublayer exists"),
    "FalconMambaForCausalLM": (SUBLAYER_POINTS, "a FalconMambaBlock is `norm` + `mixer`"),
    "GraniteMoeHybridForCausalLM": (
        ATTENTION_POINTS_ONLY,
        "the *default config* is all `linear_attention`, so this probe's trunk has no attending layer "
        "-- a shipped granite-4.0-h checkpoint interleaves `full_attention` and those layers resolve, "
        "which `test_granite_hybrid_attention_resolves_where_a_checkpoint_has_it` shows",
    ),
}

# Families with no capturable `value`, so no DFA. Pinned separately from the core points because the
# tensor genuinely does not exist rather than being unreachable: multi-head latent attention stores a
# compressed kv latent and expands it inside the forward.
KNOWN_NO_VALUE: dict[str, str] = {
    "DeepseekV2ForCausalLM": "MLA: kv latent expanded in the forward",
    "DeepseekV3ForCausalLM": "MLA: kv latent expanded in the forward",
    "DeepseekV32ForCausalLM": "MLA: kv latent expanded in the forward",
    "DeepseekV4ForCausalLM": "MLA: kv latent expanded in the forward",
    "Glm4MoeLiteForCausalLM": "MLA: kv latent expanded in the forward",
    "GlmMoeDsaForCausalLM": "MLA: kv latent expanded in the forward",
    "MiniCPM3ForCausalLM": "MLA: kv latent expanded in the forward",
    "LongcatFlashForCausalLM": "MLA: kv latent expanded in the forward",
    # transformers 5.15 added this one, and its own docstring says what it is: "Multi-headed Latent
    # Attention (MLA) from Deepseek V2". Same absence as the rest of this list, for the same reason.
    # (AXK2 is MLA too, and is not here because this table is scoped to what vLLM serves.)
    "AXK1ForCausalLM": "MLA: kv latent expanded in the forward",
}

# The eight state-space families above fetch a `causal-conv1d`/`mamba-ssm` kernel inside `__init__`, by
# version rather than by revision -- which `kernels` cannot resolve from the cache, so an offline run
# skips them however warm it is. CI runs online (`HF_HUB_OFFLINE: '0'`) and audits them there.
NEEDS_NETWORK = "builds a hub kernel; rerun with HF_HUB_OFFLINE=0"


@dataclasses.dataclass
class Coverage:
    """One family's result. ``points`` maps a point to ``"ok"`` or to why it did not resolve."""

    arch: str
    status: str
    hf_class: str | None = None
    detail: str = ""
    points: dict[str, str] = dataclasses.field(default_factory=dict)

    @property
    def core_gaps(self) -> list[str]:
        return [p for p in CORE_POINTS if self.points.get(p, "unprobed") != "ok"]


class _NoTokenizer:
    """`EagerModel` requires a tokenizer; nothing here tokenizes, and loading a real one is network.

    Deliberately not a stub with methods: if the audit ever grows a step that tokenizes, an
    ``AttributeError`` naming this class is a better outcome than results derived from a fake.
    """


def snapshot() -> dict[str, Any]:
    return json.loads(SNAPSHOT.read_text())


def text_generation_archs() -> list[str]:
    """The vLLM architectures that generate text, which is the scope of the canonical points.

    Embedding, reranking and classification architectures are excluded at the source: they are
    encoder stacks with pooling heads, so most of the points below are not merely missing but
    meaningless. The multimodal list is excluded for a different reason -- its text half is one of
    these families, and it is the *wrapper* that varies, which config defaults do not exercise.

    Includes the families vLLM serves through its transformers backend rather than a native
    implementation: vLLM serves them, so they are in scope, and having no native implementation is a
    fact about vLLM and not about the module tree we hook.
    """
    snap = snapshot()
    return sorted(set(snap["text_generation"]) | set(snap["transformers_backend"]))


# Registry names that are a *former* spelling of a family transformers now ships under its own name,
# where a checkpoint written against the old name loads on the new class today with no remote code.
# Without these, a family we do cover reads as unauditable. Case is handled separately (below), which
# is enough for `MPTForCausalLM` -> `MptForCausalLM` but not for a rename.
LEGACY_NAMES: dict[str, str] = {
    "RWForCausalLM": "FalconForCausalLM",  # Falcon's pre-upstreaming name; falcon-rw-* still use it
    "StableLMEpochForCausalLM": "StableLmForCausalLM",  # stablelm-3b-4e1t's original remote-code name
    "Fairseq2LlamaForCausalLM": "LlamaForCausalLM",  # a converted Llama; `model_type` is still llama
    "GritLM": "MistralForCausalLM",  # Mistral plus a pooling head vLLM adds
}


def hf_class_for(arch: str) -> tuple[str, Any] | None:
    """The ``transformers`` class for a vLLM architecture name, or ``None`` if there is none.

    Matched case-insensitively, because a few registry entries are the ``trust_remote_code``
    capitalization of a family transformers ships (``MPTForCausalLM`` is ``MptForCausalLM``), and
    through :data:`LEGACY_NAMES` for the ones that were renamed outright. Both exist so that a
    covered family is not reported as unauditable over its spelling.
    """
    import transformers

    for candidate in (arch, LEGACY_NAMES.get(arch)):
        if candidate and (cls := getattr(transformers, candidate, None)) is not None:
            return candidate, cls
    lowered = arch.lower()
    for name in dir(transformers):
        if name.lower() == lowered and (cls := getattr(transformers, name, None)) is not None:
            return name, cls
    return None


def build_on_meta(hf_class: Any, arch: str) -> Any:
    """A meta-device instance of ``hf_class``, built from its config class's own defaults.

    Nothing is shrunk, not even the layer count: a meta tensor allocates no storage, so the whole
    registry probes in seconds at full size, and every kind of shrinking turned out to change what is
    being probed. A shorter trunk truncates the ``layer_types`` pattern a hybrid family validates
    against (reported as unbuildable, which is the probe's fault), or drops the only attention layer
    out of a mostly-SSM trunk (reported as a missing attention module, which is worse).
    """
    import torch

    config = hf_class.config_class()
    config.architectures = [arch]
    with torch.device("meta"):
        return hf_class(config)


def _reached_the_hub(exc: BaseException) -> bool:
    """Whether a build failure was a download, read off the traceback rather than the message.

    By frame, because the fetch is several libraries deep and raises whatever suits it -- ``kernels``
    raises a bare ``ValueError`` when a version is uncached and the hub is offline, which is
    indistinguishable from a config error by type or text.
    """
    fetchers = ("kernels", "huggingface_hub")
    for err in (exc, exc.__cause__, exc.__context__):
        tb = getattr(err, "__traceback__", None)
        while tb is not None:
            module = tb.tb_frame.f_globals.get("__name__", "")
            if module.split(".", 1)[0] in fetchers:
                return True
            tb = tb.tb_next
    return False


def _layers_to_try(model: Any, point: str) -> Sequence[int | None]:
    if point in _MODEL_LEVEL_POINTS:
        return (None,)
    if point in _ATTENTION_POINTS:
        return model.arch.softmax_attention_layers()
    return range(model.arch.n_layers)


def probe(arch: str, points: Iterable[str] = (*CORE_POINTS, *CONDITIONAL_POINTS)) -> Coverage:
    """Resolve each point on ``arch``, reporting rather than raising.

    A point counts as resolved if it resolves on *any* layer it applies to, since a mixed trunk is
    covered when the block types that have the point have it -- and the reported reason is the last
    layer's, which on a uniform trunk is every layer's.
    """
    from interp_engine import EagerModel

    found = hf_class_for(arch)
    if found is None:
        return Coverage(arch, status="no_transformers_class")
    name, hf_class = found
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            hf_model = build_on_meta(hf_class, arch)
        except Exception as exc:
            status = "needs_download" if _reached_the_hub(exc) else "not_buildable"
            return Coverage(arch, status=status, hf_class=name, detail=f"{type(exc).__name__}: {exc}")
        try:
            # Separated from the build above because the two failures mean opposite things: one is
            # transformers not instantiating its own defaults, the other is a tree interp-engine
            # cannot walk -- a gap, and the most severe kind, since no point resolves at all.
            model = EagerModel(arch, hf_model=hf_model, tokenizer=_NoTokenizer(), device=None)
        except Exception as exc:
            return Coverage(arch, status="not_loadable", hf_class=name, detail=f"{type(exc).__name__}: {exc}")

    result = Coverage(arch, status="probed", hf_class=name)
    for point in points:
        layers = list(_layers_to_try(model, point))
        if not layers:
            result.points[point] = NO_ATTENTION_LAYERS
            continue
        for layer in layers:
            try:
                model.resolve_point(point, layer)
                result.points[point] = "ok"
                break
            except Exception as exc:
                # Recorded by *kind* rather than by concrete class: this column's whole meaning is
                # "a ValueError is a fact about the architecture, an AttributeError is a module this
                # engine failed to name", and the engine is free to raise a more specific ValueError
                # subclass (it now does -- `ResidualBasisUnsupported`) without that meaning changing.
                # Keying on the exact class name made every such refinement look like a new failure
                # mode here. The subclass name is not lost information worth keeping: the message
                # that follows already says what the refusal is.
                kind = "ValueError" if isinstance(exc, ValueError) else type(exc).__name__
                result.points[point] = f"{kind}: {exc}"
    return result


def audit(archs: Iterable[str] | None = None) -> list[Coverage]:
    return [probe(arch) for arch in (archs if archs is not None else text_generation_archs())]


# --- CLI ---------------------------------------------------------------------


def installed_archs(registry: Any) -> dict[str, list[str]]:
    """The three lists the snapshot records, read off a live vLLM registry.

    Shared by ``--refresh`` and the test that audits the committed snapshot against an installed
    vLLM, so the two cannot compare unlike things. They nearly did: the scope
    :func:`text_generation_archs` reports is the *union* of the native table and the
    transformers-backend one, and checking that union against ``_TEXT_GENERATION_MODELS`` alone reads
    every transformers-backend family as one vLLM dropped -- five of them at 0.26.0, on a snapshot
    taken from that same version. Invisible for as long as the test venv had no vLLM to skip on.
    """
    return {
        "text_generation": sorted(registry._TEXT_GENERATION_MODELS),
        "multimodal": sorted(registry._MULTIMODAL_MODELS),
        "transformers_backend": sorted(
            arch
            for arch, (_, backend) in registry._TRANSFORMERS_SUPPORTED_MODELS.items()
            # The rest of that table maps to a multimodal backend class, and is in `multimodal`.
            if backend in ("TransformersForCausalLM", "TransformersMoEForCausalLM")
        ),
    }


def _refresh() -> None:
    """Rewrite the snapshot from an installed vLLM. Run with a vLLM venv's interpreter."""
    import vllm
    from vllm.model_executor.models import registry

    payload = {
        "_comment": (
            "Snapshot of vLLM's model registry -- the list of families interp-engine aims to cover. "
            "Regenerate with `<vllm venv>/bin/python tests/family_coverage.py --refresh`; see "
            "tests/family_coverage.py for what is audited against it."
        ),
        "vllm_version": vllm.__version__,
        **installed_archs(registry),
    }
    SNAPSHOT.write_text(json.dumps(payload, indent=1) + "\n")
    archs = set(payload["text_generation"]) | set(payload["transformers_backend"])
    print(f"wrote {SNAPSHOT} from vllm {vllm.__version__}: {len(archs)} text-generation archs")


def _print_matrix() -> None:
    reports = audit()
    probed = [r for r in reports if r.status == "probed"]
    for report in reports:
        if report.status != "probed":
            print(f"{report.arch:38s} {report.status:22s} {report.detail[:80]}")
    print()
    for report in probed:
        gaps = report.core_gaps
        conditional = [p for p in CONDITIONAL_POINTS if report.points[p] != "ok"]
        flag = "GAP " if gaps else "    "
        print(f"{flag}{report.arch:38s} core: {'all' if not gaps else ' '.join(gaps)}  |  no: {' '.join(conditional)}")
    print()
    statuses = ("probed", "not_loadable", "no_transformers_class", "not_buildable", "needs_download")
    counts = {status: sum(r.status == status for r in reports) for status in statuses}
    print(f"{len(reports)} architectures: " + ", ".join(f"{v} {k}" for k, v in counts.items()))
    print(f"{sum(not r.core_gaps for r in probed)}/{len(probed)} probed families resolve every core point")
    for report in probed:
        for point in report.core_gaps:
            print(f"  {report.arch} {point}: {report.points[point][:110]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="rewrite the snapshot from an installed vLLM")
    args = parser.parse_args()
    if args.refresh:
        _refresh()
    else:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
        _print_matrix()
