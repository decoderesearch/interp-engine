"""`docs/MODELS_STATUS.md` and the README's support summary say only what the suite can still show.

A support matrix goes stale silently and is read as a promise, so these tests fail on each way it can
drift apart from its sources: a family that changed tier, an architecture vLLM added or renamed, a
checkpoint added to or dropped from the comparison sweep, or the two documents disagreeing with each
other about a count.

Not a byte-for-byte comparison against a regenerated doc, because generation needs the hub for eight
kernel-fetching families and an offline run would demote them -- which would make the test fail for the
one reason that is not a drift. Each property below is checked directly instead, and offline it is the
audit-derived ones that skip rather than the whole test.
"""

from __future__ import annotations

import json
import re

import family_coverage as fc
import invariants
import models_status as ms
import pytest


@pytest.fixture(scope="session")
def doc() -> str:
    return ms.DOC.read_text()


@pytest.fixture(scope="session")
def readme() -> str:
    return ms.README.read_text()


@pytest.fixture(scope="session")
def coverage() -> dict[str, fc.Coverage]:
    return {report.arch: report for report in fc.audit()}


def _known_architectures() -> set[str]:
    """Every architecture the doc could be about: vLLM's text-generation list plus the sweep's wrappers.

    The sweep contributes three multimodal wrappers (Gemma-3/4 and Qwen3.5 are served as
    `*ForConditionalGeneration`), which are verified numerically and belong in the doc even though the
    audit's scope is the text-generation registry.
    """
    return set(fc.text_generation_archs()) | set(ms.sweep_architectures().values())


def _architectures_in(text: str) -> set[str]:
    """Which known architectures `text` names. Matched with its backticks, so no name is a prefix of
    another's match (`MptForCausalLM` and `MPTForCausalLM` are both real, and both in the registry)."""
    return {arch for arch in _known_architectures() if f"`{arch}`" in text}


def test_every_architecture_vllm_serves_is_named_exactly_once(doc: str) -> None:
    """The completeness property, and the one that catches a vLLM upgrade.

    A new family that nobody has classified is the failure mode this whole doc exists to prevent: it
    would be absent from the page rather than listed as unknown, and absence reads as "not applicable"
    to everyone except the person who added it.
    """
    listed = _architectures_in(doc)
    missing = [arch for arch in fc.text_generation_archs() if arch not in listed]
    assert not missing, f"not in docs/MODELS_STATUS.md; regenerate it: {missing}"


def test_no_architecture_is_in_two_tiers(doc: str) -> None:
    """Tiers are claims of different strength, so an architecture in two of them makes both unreadable."""
    # The doc carries three cross-cutting tables that name a family already tiered elsewhere. They are
    # facts about a family rather than claims about it, so a second mention is the point, not a drift.
    swept = ms.uncorroborated(ms.verified_families(ms.verified_checkpoints(), ms.sweep_architectures()))
    cross_cutting = set(fc.KNOWN_NO_VALUE) | set(fc.ARCHITECTURAL_ABSENCES) | set(swept)
    sections = _tier_sections(doc)
    seen: dict[str, str] = {}
    for tier, body in sections.items():
        for arch in _architectures_in(body):
            if arch in seen and arch not in cross_cutting:
                pytest.fail(f"{arch} is under both '{seen[arch]}' and '{tier}'")
            seen.setdefault(arch, tier)


def _tier_sections(doc: str) -> dict[str, str]:
    """The doc split at its `##` headings, keyed by heading text."""
    parts = re.split(r"\n## ", doc)
    return {chunk.split("\n", 1)[0].strip(): chunk for chunk in parts[1:]}


def test_the_tier_a_family_is_in_is_the_tier_the_audit_puts_it_in(doc: str, coverage: dict[str, fc.Coverage]) -> None:
    """The doc is generated, so this catches an *edited* doc and a stale one alike."""
    stalled = [arch for arch, report in coverage.items() if report.status == "needs_download"]
    if stalled:
        pytest.skip(f"{fc.NEEDS_NETWORK}: {len(stalled)} families did not build")

    verified = set(ms.corroborated(ms.verified_families(ms.verified_checkpoints(), ms.sweep_architectures())))
    sections = _tier_sections(doc)
    headings = {
        ms.VERIFIED: "Verified numerically",
        ms.RESOLVES: "Resolves, unverified numerically",
        ms.UNAUDITED: "Unaudited",
        ms.BROKEN: "Does not work",
    }
    misplaced = []
    for arch in fc.text_generation_archs():
        expected = ms.tier_of(arch, coverage.get(arch), verified)
        # A tier's section carries subsections (the absence tables), so membership is checked against
        # the tier's *own* table rows -- the lines that name exactly one architecture first.
        if arch not in _table_subjects(sections[headings[expected]]):
            misplaced.append((arch, expected))
    assert not misplaced, f"regenerate docs/MODELS_STATUS.md; these are in the wrong tier: {misplaced[:8]}"


def _table_subjects(section: str) -> set[str]:
    """Architectures a section's table rows are *about*, i.e. named in the first cell."""
    subjects = set()
    for line in section.splitlines():
        if line.startswith("| `"):
            subjects |= _architectures_in(line.split("|")[1])
    return subjects


def test_the_verified_tier_is_exactly_the_committed_comparison_results() -> None:
    """Tier 1 is the strongest claim in the repo, so it is pinned to the evidence on disk.

    Both directions matter. A checkpoint whose cells were deleted must lose its claim, and a checkpoint
    added to the sweep must gain a family in the map -- otherwise its rows are captured, scored and
    published in the README table while the support doc goes on not mentioning its architecture.
    """
    sweep = set(json.loads(ms.SWEEP.read_text()))
    architectures = ms.sweep_architectures()
    assert set(architectures) == sweep, (
        "comparison/sweep_architectures.json is out of step with sweep_models.json; refresh it with "
        "`tests/models_status.py --refresh-architectures`"
    )

    by_arch: dict[str, list[str]] = {}
    for hf_id, arch in architectures.items():
        by_arch.setdefault(arch, []).append(hf_id)
    dupes = {arch: ids for arch, ids in by_arch.items() if len(ids) > 1}
    # Llama 3.1 8B stays in the sweep next to 3.3 70B so static can score a small LlamaForCausalLM.
    #
    # Gemma 4 is the other exception, and for a different reason: one checkpoint per architecture
    # assumes the class identifies the wiring, and here it does not. The 26B declares the same class
    # as the 31B and turns on 128 routed experts with `enable_moe_block`, so scoring only the 31B
    # would leave the routing points -- and a feed-forward that is two branches rather than one --
    # unexercised on the whole family while the table reported the class as covered.
    allowed = {
        "LlamaForCausalLM": {"meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.3-70B-Instruct"},
        "Gemma4ForConditionalGeneration": {"google/gemma-4-26B-A4B-it", "google/gemma-4-31B"},
    }
    unexpected = {arch: ids for arch, ids in dupes.items() if set(ids) != allowed.get(arch)}
    assert not unexpected, f"sweep lists more than one checkpoint per architecture: {unexpected}"


def test_a_verified_family_agrees_with_at_least_one_independent_engine() -> None:
    """What "verified" has to mean, checked rather than assumed.

    interp-engine's own two backends agreeing says nothing about correctness -- they share this
    repository's understanding of the architecture. The claim is only worth making when an
    independently written implementation reproduced the same numbers.
    """
    swept = ms.verified_families(ms.verified_checkpoints(), ms.sweep_architectures())
    published = ms.corroborated(swept)
    unsupported = {
        arch: entry["agreeing"] for arch, entry in published.items() if not entry["agreeing"] & ms.INDEPENDENT_ENGINES
    }
    assert not unsupported, f"only interp-engine's own engines agree here: {unsupported}"


def test_a_swept_family_with_no_independent_result_is_reported_rather_than_promoted(doc: str) -> None:
    """The demotion has to leave a trace, or the sweep's failures become invisible by being excluded.

    Filtering the verified tier is only half a fix: a family whose comparisons all errored would then
    look identical to one nobody ever ran, and the errors are the actionable part. So the doc names
    them, and the count below is what keeps the filter from quietly swallowing a growing pile.

    The list is empty as of the sweep that fixed nnsight's loading of DeepSeek-V2, LFM2-MoE and
    Nemotron-H, and the generator drops the section with it -- so the assertion in that case is that
    the section is gone rather than empty, and the guard survives the next family that needs it.
    """
    swept = ms.verified_families(ms.verified_checkpoints(), ms.sweep_architectures())
    unconfirmed = ms.uncorroborated(swept)
    sections = _tier_sections(doc)
    if not unconfirmed:
        assert "Swept, and nothing independent came back" not in sections
        return

    reported = sections["Swept, and nothing independent came back"]
    verified_section = sections["Verified numerically"]
    for arch in unconfirmed:
        assert f"`{arch}`" in reported, f"{arch} was demoted out of verified without being reported"
        assert arch not in _architectures_in(verified_section)


def test_the_readme_summary_counts_what_the_doc_lists(readme: str, doc: str) -> None:
    """Two documents, one set of numbers. A README summary that quietly disagrees is worse than none."""
    block = readme[readme.index(ms.START) : readme.index(ms.END)]
    counted = {tier: int(n) for tier, n in re.findall(r"\| \*\*(\w+)\*\* \| (\d+) \|", block)}
    assert set(counted) == set(ms.TIER_SUMMARY), counted

    declared = {tier: int(n) for tier, n in re.findall(r"\*\*(\w+)\*\* --.*?\((\d+) architectures\)", doc)}
    assert counted == declared, "README counts and MODELS_STATUS.md counts disagree; regenerate both"
    assert sum(counted.values()) >= len(fc.text_generation_archs())


def test_every_tier_is_described_the_same_way_in_both_documents(readme: str, doc: str) -> None:
    """Both texts render `TIER_SUMMARY`, so a hand-edited description in either one is a drift."""
    for tier, summary in ms.TIER_SUMMARY.items():
        assert summary in doc, f"{tier} description missing from the doc"
        assert summary in readme, f"{tier} description missing from the README"


def test_the_gap_tables_reach_the_doc_verbatim(doc: str) -> None:
    """The reasons are written once, in `family_coverage.py`, where the tests that pin them live.

    Paraphrasing them into the doc is how a documented limitation and the code's own account of it come
    apart, and the doc is the copy people read.
    """
    for arch, why in fc.KNOWN_GAPS.items():
        assert why in doc, f"{arch}: KNOWN_GAPS reason is not in the doc"
    for arch, (_, why) in fc.ARCHITECTURAL_ABSENCES.items():
        assert why in doc, f"{arch}: ARCHITECTURAL_ABSENCES reason is not in the doc"


def test_the_invariant_column_says_what_the_sweep_enforces(doc: str) -> None:
    """The column is rendered from the invariant sweep's own tables, so it cannot promise more.

    A family with an exception must be named as one in the doc, and a family with none must not be:
    "invariants hold" is a claim `tests/test_invariants.py` fails on, and the doc is where it is read.
    """
    for arch in ms.tiers({r.arch: r for r in fc.audit()}, set()).get(ms.RESOLVES, []):
        excepted = sorted(name for family, name in invariants.EXCEPTIONS if family == arch)
        rendered = ms._invariants(arch)
        assert bool(excepted) == rendered.startswith("all but"), f"{arch}: {rendered}"
        for name in excepted:
            assert f"`{name}`" in rendered


def test_the_doc_is_marked_generated(doc: str) -> None:
    """It is rewritten wholesale, so an edit to it is lost work -- which the first line has to say."""
    assert doc.startswith(ms.GENERATED)
