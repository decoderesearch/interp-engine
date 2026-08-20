"""Other frameworks' vocabulary stays at the edges -- enforced by parsing, not grepping.

The engine is the reference implementation in a six-engine comparison, so a canonical point has to
mean one thing defined by the HF forward pass rather than "whatever TransformerLens calls it". Two of
those names collide across conventions and produce plausible wrong numbers (TL's block-level
``hook_mlp_out`` is our ``mlp_out_post`` on a sandwich-norm model; TL's ``mlp.hook_post`` is our
``mlp_act``, a different width in a different position), so keeping the foreign vocabulary in one
module keeps the collisions in one module too. ``AGENTS.md`` states the decision; this is its teeth.

**The line is whether the engine carries a foreign name or merely mentions it.** A
``PointSpec.tlens_name`` field would make every new point restate a vocabulary we do not own. A
comment saying "NOT TransformerLens' ``hook_normalized``" prevents a specific mistranslation at the
code that invites it, and deleting it would make the core less navigable for exactly the audience
``mappers.py`` serves.

So this checks the **code** half and exempts prose *by construction*: it walks the AST and looks at
identifiers, string literals and imports, which means docstrings and comments are not part of what it
sees. A regex over the source could not draw that line, and would either ban the useful warnings or
be silenced with per-line ignores -- the failure mode this file is meant to avoid being edited into.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "interp_engine"

# `mappers.py` is the translator, so foreign names are its subject matter. Nothing else is exempt:
# the fix for a failure here is to move the name into mappers.py, never to extend this tuple.
TRANSLATION_MODULES = ("mappers.py",)

# The translator's own public functions have to say what they translate, so `tlens_hook_to_point` is
# a correct name for it. `__init__.py` is allowed to re-export exactly those -- the package's public
# surface is an edge in the same sense mappers.py is -- and nothing else foreign.
PUBLIC_SURFACE = "__init__.py"

FOREIGN = ("transformer_lens", "transformerlens", "tlens", "nnsight", "nnterp", "hookedtransformer")

# Ours, and older than the frameworks' use of the word: a forward hook is a torch concept.
OURS = ("hook_points", "hook_capture_points", "validate_hook_points", "hook_manager", "hooks")


def _modules() -> list[pathlib.Path]:
    return sorted(p for p in PACKAGE.rglob("*.py") if p.name not in TRANSLATION_MODULES)


def _is_ours(text: str) -> bool:
    return any(own in text.lower() for own in OURS)


def _translator_api() -> frozenset[str]:
    """Top-level names `mappers.py` defines, read from its source rather than listed here."""
    tree = ast.parse((PACKAGE / "mappers.py").read_text())
    return frozenset(
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) and not node.name.startswith("_")
    )


def _offenders(tree: ast.AST, *, allow: frozenset[str] = frozenset()) -> list[str]:
    """Foreign vocabulary reachable at runtime: identifiers, attributes, literals, imports.

    Docstrings are `ast.Constant` nodes too, so they are excluded explicitly -- they are prose that
    happens to be stored as a string, and the whole point is that prose may cross-reference.
    """
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }

    found: list[str] = []
    for node in ast.walk(tree):
        match node:
            case ast.Name(id=text) | ast.Attribute(attr=text) | ast.arg(arg=text) | ast.keyword(arg=str() as text):
                candidates = [text]
            case ast.FunctionDef(name=text) | ast.AsyncFunctionDef(name=text) | ast.ClassDef(name=text):
                candidates = [text]
            case ast.Constant(value=str() as text) if id(node) not in docstrings:
                candidates = [text]
            case ast.Import(names=aliases):
                candidates = [a.name for a in aliases]
            case ast.ImportFrom(module=str() as text, names=aliases):
                candidates = [text, *(a.name for a in aliases)]
            case _:
                continue
        found += [c for c in candidates if any(f in c.lower() for f in FOREIGN) and not _is_ours(c) and c not in allow]
    return found


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_no_foreign_vocabulary_in_engine_code(path: pathlib.Path):
    """Every module except the translator, checked one by one so a failure names the file."""
    allow = _translator_api() if path.name == PUBLIC_SURFACE else frozenset()
    offenders = _offenders(ast.parse(path.read_text()), allow=allow)
    assert not offenders, (
        f"{path.name} carries other frameworks' vocabulary in executable code: {sorted(set(offenders))}. "
        "Translation belongs in interp_engine/mappers.py -- see AGENTS.md. Prose may cross-reference; "
        "identifiers, string literals and imports may not."
    )


def test_nothing_in_the_engine_imports_those_libraries():
    """A hard dependency would make the engine unloadable without them, and it has none.

    Separate from the above because it is a different failure: the check above would catch the import
    statement, but this states the property a reader actually cares about -- `pip install
    interp-engine` pulls in neither framework.
    """
    imports: set[str] = set()
    for path in PACKAGE.rglob("*.py"):  # including mappers.py: it translates names, it does not import
        for node in ast.walk(ast.parse(path.read_text())):
            match node:
                case ast.Import(names=aliases):
                    imports |= {a.name.split(".")[0] for a in aliases}
                case ast.ImportFrom(module=str() as module):
                    imports.add(module.split(".")[0])
    assert not imports & {"transformer_lens", "nnsight", "nnterp"}


def test_the_check_can_actually_fail():
    """Guard the guard: a tripwire that cannot fire is worse than none, since it reads as coverage."""
    assert _offenders(ast.parse("tlens_name: str = ''")), "an annotated field went unnoticed"
    assert _offenders(ast.parse("spec = PointSpec(tlens_name='blocks.5.hook_z')")), "a keyword went unnoticed"
    assert _offenders(ast.parse("import transformer_lens")), "an import went unnoticed"
    assert _offenders(ast.parse("x = {'nnsight': 'mlps_output'}")), "a foreign literal went unnoticed"
    assert not _offenders(ast.parse('"""Prose about TransformerLens hook_z is fine."""')), "prose banned"
    assert not _offenders(ast.parse("def _validate_hook_points(): pass")), "our own word banned"
    # The re-export allowance is scoped, not global: it exempts the name, not the whole file.
    api = _translator_api()
    assert "tlens_hook_to_point" in api, "the translator's public API was not discovered"
    assert not _offenders(ast.parse("from .mappers import tlens_hook_to_point"), allow=api)
    assert _offenders(ast.parse("import tlens_extras"), allow=api), "the allowance leaked past the API"


def test_the_decision_is_written_down_where_an_agent_will_read_it():
    """The test enforces the rule; AGENTS.md is where someone learns *why* before tripping it."""
    agents = (PACKAGE.parent / "AGENTS.md").read_text()
    assert "mappers.py" in agents and pathlib.Path(__file__).name in agents
