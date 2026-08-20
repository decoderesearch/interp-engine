"""The point table is the single source of truth -- these are the checks that make that true.

A point missing from one of the table's consumers fails *silently*: absent from
``HOOK_CAPTURE_POINTS`` it is merely refused under vLLM, absent from the width guard merely
unchecked, absent from the reshape set it merely comes back with its batch and sequence axes fused.

Each test below pins the table against one consumer, so a half-added point fails at the consumer
that would otherwise have ignored it -- including the **documentation**, whose footnote markers are
checked here rather than maintained by hand.
"""

from __future__ import annotations

import pathlib
import re

import pytest
from harness import GPT2, load_model

from interp_engine import mappers, points
from interp_engine.capture import _TOKEN_FLATTENED_POINTS
from interp_engine.vllm_capture import (
    _GLOBAL_POINTS,
    _INPUT_POINTS,
    _OUTPUT_POINTS,
    HOOK_CAPTURE_POINTS,
    MHC_KERNEL_POINTS,
    STEERABLE_POINTS,
)
from interp_engine.vllm_capture.requests import _DEMUX_MHC_HOOKS, _DEMUX_OUT_HOOKS, _DEMUX_PRE_HOOKS

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

#: The samples site's vocabulary page. Outside ``docs/``, so nothing else in this file reaches it,
#: and it tables every point by name -- the one list a reader consults to find out what may be asked
#: for at all.
ADDRESSES_PAGE = ROOT / "visualizer-web/docs-site/docs/addresses.md"

#: How the support table in ``docs/SUPPORTED_POINTS.md`` spells each :class:`points.VllmSupport`.
_SUPPORT_MARKS = {
    "✅": points.VllmSupport.HOOKS,
    "♻\ufe0f": points.VllmSupport.RECOMPUTE,
    "❌": points.VllmSupport.NONE,
}

#: Every place a point *count* is stated in prose, as ``(doc, pattern, which count)``. The table
#: below is parsed row by row, but "34 standardized points" in a sentence is a separate claim that
#: nothing else checks -- and it went stale once already, at 31, while the table beside it was right.
#: A reworded sentence fails here too, deliberately: an unmatched pattern is an unchecked claim.
_PROSE_COUNTS = (
    ("README.md", r"\((\d+) 'points'/addresses", "declared"),
    ("README.md", r"supports (\d+) standardized points", "declared"),
    ("README.md", r"(\d+) of them on vLLM", "served"),
    ("docs/SUPPORTED_POINTS.md", r"The (\d+) canonical points", "declared"),
    ("docs/SUPPORTED_POINTS.md", r"vLLM backend serves (\d+) of them", "served"),
    ("docs/SUPPORTED_POINTS.md", r"(\d+) of those by recompute", "recomputed"),
    ("visualizer-web/docs-site/docs/addresses.md", r"The (\d+) points", "declared"),
    ("visualizer-web/docs-site/docs/addresses.md", r"(\d+) on every model", "global"),
    ("visualizer-web/docs-site/docs/addresses.md", r"(\d+) more that need a hyper-connection trunk", "conditional"),
)


def test_every_declared_point_is_known_to_the_resolver():
    """The drift test the table exists for: a new row without a resolver branch fails here.

    Refusals are *expected* on a model that lacks the structure -- gpt2 has no QK-norm, no MoE and no
    gated MLP -- so the assertion is not "it resolves" but "the resolver recognized the name". The
    unknown-name branch is the failure being detected, and it is distinguishable because it says so.
    """
    model = load_model(GPT2, device="cpu")
    for point in points.POINTS:
        if point.scope is points.Scope.GLOBAL or not point.module_resolved:
            continue
        try:
            module, side = model.resolve_point(point.name, 0)
        except ValueError as exc:
            assert "Unknown canonical hook name" not in str(exc), (
                f"{point.name} is declared in points.POINTS but resolve_point has no branch for it"
            )
            assert str(exc).strip(), f"{point.name} refused with an empty message"
            continue
        assert module is not None and side.split(":")[0] in ("input", "output")


def test_points_not_carried_by_a_module_say_so_instead_of_looking_unknown():
    """`attn_probs` and `attn_scores` are declared but not module-resolved, and the two differ."""
    model = load_model(GPT2, device="cpu")
    for name in ("attn_probs", "attn_scores"):
        assert points.point_spec(name) is not None and not points.point_spec(name).module_resolved
        with pytest.raises(ValueError, match="not resolvable to a module"):
            model.resolve_point(name, 0)


def test_a_layer_point_without_a_layer_is_refused_not_read_as_a_module_path():
    """The typo trap the open point set creates: without this branch `resolve_point("resid_mid")`
    falls through to `get_submodule` and fails with a message about module paths for what is really a
    missing argument."""
    model = load_model(GPT2, device="cpu")
    with pytest.raises(ValueError, match="per-layer"):
        model.resolve_point("resid_mid")


def test_a_misspelled_point_gets_a_suggestion():
    model = load_model(GPT2, device="cpu")
    with pytest.raises(ValueError, match="did you mean"):
        model.resolve_point("resid_mids", 0)


def test_the_tlens_mapper_covers_every_point_or_declares_it_unmapped():
    """Coverage, not correctness: adding a point must force a decision about translation.

    Over the conditional rows as well as the global ones. Scoping this to `known_names()` was how
    all seven mHC points came to be neither mapped nor declared unmapped once TransformerLens 3
    shipped a DeepSeek-V4 bridge that names them: a family point could be added, and a foreign name
    for it could appear, without anything here failing.

    This is also where the dependency direction is visible -- `mappers` imports the point set, so the
    engine's table carries no foreign names (AGENTS.md).
    """
    every_point = points.known_names() | points.hyper_connection_names()
    mapped = set(mappers._POINT_TO_TLENS)
    declared = mapped | set(mappers.UNMAPPED_TLENS)
    assert every_point == declared, (
        f"undecided: {sorted(every_point - declared)}; stale: {sorted(declared - every_point)}"
    )
    assert not (mapped & mappers.UNMAPPED_TLENS), "a point cannot be both mapped and unmapped"


def test_every_vllm_hookable_point_has_a_mechanism_on_the_vllm_tree():
    """A point declared servable but given no mechanism would be accepted and then never captured.

    Three mechanisms, not two: a point is read from a module's input, from its output, or -- for the
    five mHC quantities that are locals of a decoder layer's forward -- off a wrapped mHC kernel call.
    The three are disjoint because each one installs something different, and a point in the wrong set
    installs a tap that cannot fire.
    """
    assert points.vllm_hookable() == HOOK_CAPTURE_POINTS
    assert HOOK_CAPTURE_POINTS == (_INPUT_POINTS | _OUTPUT_POINTS | MHC_KERNEL_POINTS)
    mechanisms = (_INPUT_POINTS, _OUTPUT_POINTS, MHC_KERNEL_POINTS)
    for i, first in enumerate(mechanisms):
        for second in mechanisms[i + 1 :]:
            assert not (first & second), f"a point cannot be served two ways: {sorted(first & second)}"


def test_the_per_request_demux_serves_every_hookable_point():
    """The mechanism tables above cover the single-request path; `VLLMModel` uses the demux instead.

    Which is the path that matters and the one that had drifted: `attn_out` was declared hookable
    and resolvable, so `_validate_hook_points` accepted it on the client, and the refusal came from
    a `ValueError` raised inside a forward on the worker. The two tables are pinned separately
    because they went out of step, not because one implies the other.
    """
    served = set(_DEMUX_PRE_HOOKS) | set(_DEMUX_OUT_HOOKS) | set(_DEMUX_MHC_HOOKS)
    assert served == HOOK_CAPTURE_POINTS, (
        f"unserved by the demux: {sorted(HOOK_CAPTURE_POINTS - served)}; "
        f"not a hookable point: {sorted(served - HOOK_CAPTURE_POINTS)}"
    )
    assert set(_DEMUX_PRE_HOOKS) == _INPUT_POINTS, "the demux reads a point from the wrong side"
    assert set(_DEMUX_OUT_HOOKS) == _OUTPUT_POINTS, "the demux reads a point from the wrong side"
    assert set(_DEMUX_MHC_HOOKS) == MHC_KERNEL_POINTS, "the demux serves an mHC point the wrong way"


def test_the_steerable_set_is_the_hookable_one_minus_what_cannot_be_written_or_said():
    """Two exclusions from capture's set, and a caller told "not steerable" needs to know which.

    The coefficients are not activations -- there is no additive edit to a doubly stochastic matrix
    that means what a steer means -- and the trunk-level points a worker spec has no way to *name*,
    since it carries the layer as an ``int``. The first is permanent and the second is a wire change,
    so they are pinned separately rather than as one blob of exceptions.

    Deliberately not a claim that every member is steerable on every model: ``resid_mid`` depends on
    the family's block and ``resid_streams`` on the kernel's halves composing, both refused at
    registration on the worker where a model is in hand.
    """
    assert STEERABLE_POINTS < HOOK_CAPTURE_POINTS
    unwritable = points.mhc_coefficient_names() | _GLOBAL_POINTS
    assert unwritable == HOOK_CAPTURE_POINTS - STEERABLE_POINTS
    assert points.mhc_coefficient_names() == {
        f"{site}_stream_{kind}" for site in ("attn", "mlp") for kind in ("write", "mix")
    }
    assert {"resid_post", "resid_pre", "resid_streams", "attn_stream_collapse", "mlp_stream_collapse"} <= (
        STEERABLE_POINTS
    )


def test_the_width_guard_covers_the_residual_points_and_not_the_sharded_ones():
    """`z`/`value` must stay out: `n_heads * head_dim` equals `hidden_size` on Llama but not Gemma-3,
    so checking them against `hidden_size` would fail for a reason unrelated to sharding."""
    wide = points.d_model_wide()
    assert {"resid_pre", "resid_mid", "resid_post", "mlp_in", "mlp_out", "attn_out"} <= wide
    assert not ({"z", "value", "mlp_act", "mlp_pre", "attn_probs"} & wide)


def test_the_reshape_set_is_exactly_the_router_points():
    assert points.token_flattened() == _TOKEN_FLATTENED_POINTS
    assert {"router_logits", "expert_weights", "expert_indices"} == _TOKEN_FLATTENED_POINTS


def test_every_eager_only_point_explains_which_kind_of_limit_it_is():
    """ "Unimplemented" and "unreachable" are the difference between filing a bug and switching
    backend, so the taxonomy is enforced rather than left to prose.

    The hyper-connection rows are checked alongside the global ones because they are the rows most
    likely to get this wrong: no one runs a ~300B model to find out, so a note claiming "unreachable"
    can sit there being false for as long as nobody rereads vLLM's source. One did.
    """
    for point in (*points.POINTS, *points.HYPER_CONNECTION_POINTS):
        if point.vllm is not points.VllmSupport.NONE:
            continue
        assert point.note.startswith(("unimplemented", "unreachable", "see ")), (
            f"{point.name}'s note does not say which kind of limit it is: {point.note!r}"
        )


def test_every_see_reference_resolves_and_none_of_them_chain():
    """The table's one abbreviation, kept honest: four QK-norm rows share one paragraph by reference.

    A dangling reference would print `see mlp_pre` at a user; a chain would print `as X: see Y`.
    """
    for point in points.POINTS:
        if not point.note.startswith("see "):
            continue
        referent = point.note.removeprefix("see ").partition(";")[0].strip()
        target = points.point_spec(referent)
        assert target is not None, f"{point.name} refers to {referent!r}, which is not a point"
        assert not target.note.startswith("see "), f"{point.name} -> {referent} is a chain of references"
        assert "see " not in points.reason(point.name), f"{point.name}'s resolved reason still defers"


def test_a_vllm_refusal_quotes_each_points_own_reason():
    from interp_engine.vllm_backend import _validate_hook_points

    with pytest.raises(ValueError) as exc:
        _validate_hook_points([("mlp_pre", 0), ("expert_weights", 1), ("nonsense", 2)])
    message = str(exc.value)
    assert "gate_up_proj" in message, "mlp_pre's own note was not quoted"
    assert "FusedMoE" in message, "expert_weights' own note was not quoted"
    assert "not a canonical point name" in message, "an unknown name was silently dropped"


def test_the_sharded_set_is_the_narrowed_axes_and_not_just_the_non_d_model_ones():
    """`router_logits` is the point this distinction exists for.

    It is not `d_model` wide, so the old "everything except `d_model_wide()`" proxy called it
    sharded and a multi-GPU pod refused it -- but vLLM routes with a `ReplicatedLinear`, so every
    rank computes the whole thing. Same for a hyper-connection trunk's streams.
    """
    sharded = points.tp_sharded()
    assert {"z", "value", "mlp_act", "q_norm_in", "k_norm_out", "lm_head"} <= sharded
    assert not ({"router_logits", "resid_post", "attn_in", "resid_streams"} & sharded)


def test_the_mapping_doc_marks_exactly_the_eager_only_points():
    """The footnote markers in the docs table, checked against the table in code.

    The marker means "capturable eagerly but not under vLLM", which is `VllmSupport.NONE` --
    `attn_probs` is unmarked because it is served by recompute, and that distinction is exactly what
    a reader of the table is trusting.
    """
    text = (DOCS / "ENGINE_HOOK_MAPPINGS.md").read_text()
    rows = re.findall(r'\| `cache\["(\w+)"[^|]*\|', text)
    assert len(rows) > 15, f"the table parse found only {len(rows)} rows; did its format change?"

    # Resolved over the conditional rows too, as the SUPPORTED_POINTS.md test is: the page tables the
    # seven hyper-connection points separately from the main table, and `point_spec` at one stream
    # cannot see them -- so without this they would read as names that are not points at all.
    every = {p.name: p for p in (*points.POINTS, *points.HYPER_CONNECTION_POINTS)}

    starred = set(re.findall(r'`cache\["(\w+)"[^|]*?\]`\s+`\*`', text))
    for name in rows:
        spec = every.get(name)
        assert spec is not None, f"the doc names `{name}`, which is not a canonical point"
        assert (name in starred) == (spec.vllm is points.VllmSupport.NONE), (
            f"`{name}` is {'starred' if name in starred else 'unstarred'} in ENGINE_HOOK_MAPPINGS.md "
            f"but declared {spec.vllm.name} in points.POINTS"
        )

    for point in points.POINTS:
        assert point.name in text, f"{point.name} is not documented in the mapping table at all"


def test_the_support_table_doc_says_what_the_point_table_declares():
    """``docs/SUPPORTED_POINTS.md``'s eager-vs-vLLM table, checked against the table it summarises.

    It is what a reader consults to decide which backend to run, so a point that quietly became
    servable (or stopped being) while the tick stayed put is worse than no table: it is a claim
    about the engine that the engine does not make. Parsed rather than maintained, for the same
    reason the mapping doc's footnote markers are.
    """
    text = (DOCS / "SUPPORTED_POINTS.md").read_text()
    # Whitespace-tolerant on purpose: a markdown formatter pads these cells to align the columns, so
    # a pattern written against single spaces silently matches only the widest rows and the test
    # passes having checked three of them. The first cell holds names, most of them wrapped in a
    # reference link to the point on interp-engine.org, so it is matched as "whatever contains
    # backticked names" and the names are picked out of it below -- the alternative is this pattern
    # having to know the doc's link style, and breaking when a row gains or loses a link.
    rows = re.findall(r"^\|([^|]*`\w+`[^|]*)\|[^|]+\|([^|]+)\|([^|]+)\|", text, re.MULTILINE)
    assert len(rows) > 20, f"the table parse found only {len(rows)} rows; did its format change?"

    # Looked up over the conditional rows too: the table's last seven are the hyper-connection ones,
    # and those are not in the global set `point_spec` searches at one stream.
    every = {p.name: p for p in (*points.POINTS, *points.HYPER_CONNECTION_POINTS)}

    documented: set[str] = set()
    for names, eager, vllm in rows:
        assert eager.strip() == "✅", f"{names} is not marked capturable eagerly, which every point is"
        declared = _SUPPORT_MARKS.get(vllm.strip())
        assert declared is not None, f"{names}'s vLLM cell is {vllm.strip()!r}, not one of {sorted(_SUPPORT_MARKS)}"
        for name in re.findall(r"`(\w+)`", names):
            spec = every.get(name)
            assert spec is not None, f"SUPPORTED_POINTS.md names `{name}`, which is not a canonical point"
            assert spec.vllm is declared, (
                f"SUPPORTED_POINTS.md marks `{name}` {vllm.strip()} but points.POINTS declares it {spec.vllm.name}"
            )
            documented.add(name)

    assert set(every) == documented, (
        f"missing from the SUPPORTED_POINTS.md table: {sorted(set(every) - documented)}; "
        f"no longer a point: {sorted(documented - set(every))}"
    )


def test_the_samples_page_names_every_point_and_no_others():
    """``addresses.md``'s two tables, checked against the registry they transcribe.

    It carries no support column -- that claim is SUPPORTED_POINTS.md's, and checked above -- so
    this is the narrower assertion that the names are the point set. A point added without a row
    here is one the samples site tells a reader does not exist, which is the same silent failure
    as a point missing from any other consumer.
    """
    rows = re.findall(r"^\|([^|]*`\w+`[^|]*)\|", ADDRESSES_PAGE.read_text(), re.MULTILINE)
    assert len(rows) > 20, f"the table parse found only {len(rows)} rows; did the page's format change?"

    listed = {name for row in rows for name in re.findall(r"`(\w+)`", row)}
    every = {p.name for p in (*points.POINTS, *points.HYPER_CONNECTION_POINTS)}
    assert listed == every, (
        f"missing from addresses.md: {sorted(every - listed)}; no longer a point: {sorted(listed - every)}"
    )


@pytest.mark.parametrize(("doc", "pattern", "which"), _PROSE_COUNTS)
def test_a_point_count_stated_in_prose_matches_the_registry(doc: str, pattern: str, which: str):
    """A sentence saying "34 standardized points" is a claim the table beside it does not keep honest."""
    every = (*points.POINTS, *points.HYPER_CONNECTION_POINTS)
    expected = {
        "declared": len(every),
        "global": len(points.POINTS),
        "conditional": len(points.HYPER_CONNECTION_POINTS),
        "served": sum(p.vllm is not points.VllmSupport.NONE for p in every),
        "recomputed": sum(p.vllm is points.VllmSupport.RECOMPUTE for p in every),
    }[which]

    found = re.findall(pattern, (ROOT / doc).read_text())
    assert found, f"{doc} no longer states its {which} point count as /{pattern}/ -- reword the pattern or the doc"
    assert all(int(n) == expected for n in found), (
        f"{doc} says {found} points {which}, but points.POINTS declares {expected}"
    )
