"""Every ``interp_engine`` name in a doc code fence must still import.

The docs carry hand-written snippets, and a snippet is the one kind of code no refactor updates.
Renaming ``Point`` to ``Address`` -- which happened -- rewrites every call site the type checker can
see and leaves ``docs/`` describing an API that no longer exists. That is worse than an out-of-date
sentence, because the reader's next step is to paste it.

So this parses the fences rather than the prose, and asserts three things per snippet: it is valid
Python, every name it imports from the package exists, and (for the top-level namespace) that name is
in ``__all__``. The third is the one with teeth beyond rot: it keeps the docs describing the API
surface rather than whatever happens to be reachable, since anything not in ``__all__`` may change
without a major version and should not be the thing a reader copies.

Deliberately NOT checked: that a snippet runs. Most need weights, a GPU, or an event loop, and a
guard that needs an 8B download is a guard that gets skipped. Import-level checking is cheap, runs on
CPU, and catches the whole class of failure that renames cause.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import re

import pytest

import interp_engine

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Every doc a reader might paste from. Kept as an explicit list so a new doc is a deliberate
#: addition rather than something a glob quietly starts (or stops) covering.
DOC_FILES = (
    "README.md",
    "AGENTS.md",
    "docs/README.md",
    "docs/USAGE.md",
    "docs/SUPPORTED_POINTS.md",
    "docs/AGENT_INTEGRATION.md",
    "docs/PORTING.md",
    "docs/GRADIENTS.md",
    "docs/PERFORMANCE.md",
    "docs/ARCHITECTURE_QUIRKS.md",
    "docs/COMPATIBILITY.md",
    "docs/ENGINE_HOOK_MAPPINGS.md",
    "docs/INTERNALS.md",
    "benchmarks/README.md",
)

#: The samples site, which is nothing *but* fences a reader pastes -- so it is the doc that rot
#: hurts most. Globbed rather than listed: a page there is one file with one job, and adding one
#: should not need an edit here. Its own build would not catch a rename, since it never imports
#: the package.
SAMPLE_DOCS = tuple(sorted(str(p.relative_to(ROOT)) for p in (ROOT / "visualizer-web/docs-site/docs").glob("*.md")))

#: Submodules the docs may import from directly. The package's own ``__init__`` says these two are
#: reachable-but-not-re-exported on purpose (the worker RPC names, and the payload codecs), so a doc
#: that shows tier-1 integration has to name them.
ALLOWED_SUBMODULES = frozenset({"interp_engine.vllm_plugin", "interp_engine.vllm_capture"})

_FENCE = re.compile(r"^```(\w*)\n(.*?)^```", re.MULTILINE | re.DOTALL)


def _fences(text: str) -> list[tuple[int, str]]:
    """``(line_number, source)`` for every ```python fence, so a failure can name a line."""
    out = []
    for match in _FENCE.finditer(text):
        if match.group(1) != "python":
            continue
        out.append((text[: match.start()].count("\n") + 1, match.group(2)))
    return out


def _engine_imports(tree: ast.Module) -> list[tuple[str, str]]:
    """``(module, name)`` for every ``interp_engine`` import in the snippet.

    ``name`` is empty for a plain ``import interp_engine[.sub]``, which asserts only that the module
    itself resolves.
    """
    found: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "interp_engine" or module.startswith("interp_engine."):
                found.extend((module, alias.name) for alias in node.names)
        elif isinstance(node, ast.Import):
            found.extend((alias.name, "") for alias in node.names if alias.name.split(".")[0] == "interp_engine")
    return found


DOC_FENCES = [
    pytest.param(path, line, source, id=f"{path}:{line}")
    for path in DOC_FILES + SAMPLE_DOCS
    for line, source in _fences((ROOT / path).read_text())
]


def test_the_doc_list_still_matches_the_docs_on_disk():
    """A doc added without being listed here is a doc nothing checks."""
    on_disk = {f"docs/{p.name}" for p in (ROOT / "docs").glob("*.md")}
    assert on_disk <= set(DOC_FILES), f"unlisted docs: {sorted(on_disk - set(DOC_FILES))}"
    for path in DOC_FILES:
        assert (ROOT / path).exists(), f"{path} is listed here but does not exist"


def test_the_samples_site_was_found():
    """The glob above is silent when the directory moves, which would drop every sample page."""
    assert len(SAMPLE_DOCS) > 5, f"only {len(SAMPLE_DOCS)} sample pages found; did docs-site move?"


def test_there_are_fences_to_check():
    """Guards against the regex silently matching nothing after a format change."""
    assert len(DOC_FENCES) > 15, f"only {len(DOC_FENCES)} python fences found; did the fence format change?"


@pytest.mark.parametrize(("path", "line", "source"), DOC_FENCES)
def test_a_doc_snippet_parses_and_its_engine_imports_resolve(path: str, line: int, source: str):
    where = f"{path}:{line}"
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        pytest.fail(f"{where} is not valid Python: {exc}")

    for module, name in _engine_imports(tree):
        if module != "interp_engine":
            assert module in ALLOWED_SUBMODULES, (
                f"{where} imports from {module}, which is not API. Import from `interp_engine` "
                f"directly, or add {module} to ALLOWED_SUBMODULES with a reason."
            )
        try:
            imported = importlib.import_module(module)
        except ImportError as exc:  # pragma: no cover -- a broken install, not doc rot
            pytest.fail(f"{where} imports {module}, which does not import: {exc}")
        if not name:
            continue
        assert hasattr(imported, name), (
            f"{where} imports `{name}` from {module}, which no longer exists. "
            f"The doc needs updating along with the rename."
        )
        if module == "interp_engine":
            assert name in interp_engine.__all__, (
                f"{where} shows `{name}`, which is reachable but not in interp_engine.__all__ -- "
                f"it can change without a major version, so it should not be in a doc a reader "
                f"copies. Export it deliberately, or use the documented equivalent."
            )
