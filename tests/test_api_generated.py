"""The API contract in ``api/engine.yaml`` holds: the generated sources are current, and every Python
backend exposes each operation with the manifest's parameters, kinds and defaults.

Signatures are checked here rather than left to pyright alone because pyright does not compare
defaults: a backend whose ``max_tokens`` defaulted to 200 against the manifest's 64 would pass it.
``interp_engine/_conformance.py`` is the other half, where pyright checks the types.
"""

from __future__ import annotations

import importlib.util
import inspect
import pathlib
import sys
from typing import Any

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load(name: str) -> Any:
    path = ROOT / "api" / f"{name}.py"
    if not path.exists():
        pytest.skip("api/ is not in the sdist; this test runs from a checkout")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generate() -> Any:
    return _load("generate")


@pytest.fixture(scope="module")
def manifest(generate) -> Any:
    return generate.load()


def test_committed_sources_are_current(generate) -> None:
    stale = [
        path.relative_to(ROOT)
        for path, text in generate.outputs().items()
        if not path.exists() or path.read_text(encoding="utf-8") != text
    ]
    assert not stale, f"stale: {stale}. Run `uv run python api/generate.py` and commit the result."


def test_every_operation_has_a_case(manifest) -> None:
    """An operation with no case is a promise nothing checks across the backends."""
    bare = [op.name for op in manifest.ops if op.kind != "property" and not op.cases]
    assert not bare, f"operations with no cases in api/engine.yaml: {bare}"


def _backends() -> list[type]:
    from interp_engine.model import EagerModel
    from interp_engine.vllm_backend import VLLMModel

    return [EagerModel, VLLMModel]


def _sets_in_init(cls: type, name: str) -> bool:
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is not None and f"self.{name} =" in inspect.getsource(init):
            return True
    return False


def _conformance_problems(cls: type, manifest: Any) -> list[str]:
    problems = []
    for op in manifest.ops:
        where = f"{cls.__name__}.{op.name}"
        found = inspect.getattr_static(cls, op.name, None)
        if op.kind == "property":
            if not isinstance(found, property) and not _sets_in_init(cls, op.name):
                problems.append(f"{where}: not a property or an attribute set in __init__")
            continue
        if found is None or not callable(found):
            problems.append(f"{where}: missing")
            continue
        fn = found
        is_coro = inspect.iscoroutinefunction(fn)
        is_gen = inspect.isasyncgenfunction(fn)
        if op.kind == "async" and not is_coro:
            problems.append(f"{where}: must be async def")
        if op.kind == "sync" and (is_coro or is_gen):
            problems.append(f"{where}: must be a plain def")
        if op.kind == "stream" and is_coro:
            problems.append(f"{where}: must return an async iterator, not a coroutine")
        params = list(inspect.signature(fn).parameters.values())[1:]
        by_name = {p.name: p for p in params}
        wanted = list(op.params)
        for i, want in enumerate(wanted):
            got = by_name.get(want.python_name)
            if got is None:
                problems.append(f"{where}: no parameter {want.python_name}")
                continue
            if want.positional:
                if got.kind not in (got.POSITIONAL_ONLY, got.POSITIONAL_OR_KEYWORD) or params.index(got) != i:
                    problems.append(f"{where}: {want.python_name} must be positional parameter {i}")
            elif got.kind is not got.KEYWORD_ONLY:
                problems.append(f"{where}: {want.python_name} must be keyword-only")
            if want.has_default:
                if got.default is got.empty or got.default != want.default:
                    problems.append(f"{where}: {want.python_name} defaults to {got.default!r}, not {want.default!r}")
            elif got.default is not got.empty:
                problems.append(f"{where}: {want.python_name} has a default the manifest does not")
        named = {p.python_name for p in wanted}
        for extra in params:
            if extra.name in named or extra.kind in (extra.VAR_POSITIONAL, extra.VAR_KEYWORD):
                continue
            if extra.default is extra.empty:
                problems.append(f"{where}: extra parameter {extra.name} has no default")
    return problems


def test_every_backend_matches_the_manifest(manifest) -> None:
    problems = [p for cls in _backends() for p in _conformance_problems(cls, manifest)]
    assert not problems, "\n".join(problems)


def test_external_records_have_the_manifest_fields(manifest) -> None:
    import dataclasses
    import importlib

    problems = []
    for record in manifest.records.values():
        if not record.external:
            continue
        cls = getattr(importlib.import_module(record.python_module), record.name)
        have = {f.name for f in dataclasses.fields(cls)}
        problems += [f"{record.name}.{f.name}: missing" for f in record.fields if f.name not in have]
    assert not problems, "\n".join(problems)


def test_the_generated_protocol_is_exported() -> None:
    import interp_engine
    from interp_engine.api import EngineAPI

    assert interp_engine.EngineAPI is EngineAPI
