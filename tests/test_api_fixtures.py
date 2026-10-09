"""The manifest's cases, run on each Python backend against the eager reference.

``api/make_fixtures.py`` writes the reference from ``EagerModel`` on the CPU in float32. The CPU test
here writes a fresh set into a temporary directory and reads it back through the same runner the
other backends use, so the plumbing is checked on every run. The vLLM test reads the set on disk
(``IE_FIXTURES_DIR`` or ``~/.cache/interp-engine/fixtures``) and holds its backend to the tolerance
each case names.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import pathlib
import sys
from typing import Any

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL = "HuggingFaceTB/SmolLM2-135M"


def _load(name: str) -> Any:
    path = ROOT / "api" / f"{name}.py"
    if not path.exists():
        pytest.skip("api/ is not in the sdist; this test runs from a checkout")
    sys.path.insert(0, str(path.parent))
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _run_all(model: Any, root: pathlib.Path, run: Any) -> list[str]:
    fixtures = _load("fixtures")
    manifest = _load("generate").load()
    problems = []
    for op in manifest.ops:
        for case in op.cases:
            got = run(fixtures.run_case(model, manifest, op, case, MODEL, root))
            problems += [f"{op.name}/{case.name}: {p}" for p in got]
    return problems


def _on_disk() -> pathlib.Path:
    root = _load("fixtures").DEFAULT_ROOT
    if not (root / MODEL.replace("/", "--") / "meta.json").exists():
        message = f"no fixtures for {MODEL} under {root}; run `uv run python api/make_fixtures.py {MODEL}`"
        if os.environ.get("IE_REQUIRE_FIXTURES") == "1":
            pytest.fail(message)
        pytest.skip(message)
    return root


def test_eager_reads_back_what_it_wrote(tmp_path: pathlib.Path) -> None:
    from interp_engine import EagerModel

    make = _load("make_fixtures")
    model = EagerModel(MODEL, device="cpu", dtype="float32", attn_implementation="eager")
    assert asyncio.run(make.write_all(model, MODEL, tmp_path)) > 0
    problems = _run_all(model, tmp_path, asyncio.run)
    assert not problems, "\n".join(problems)


def test_a_stale_fixture_is_a_failure_not_a_pass(tmp_path: pathlib.Path) -> None:
    """A case edited in the manifest after its fixture was written must not pass on the old one."""
    import json

    from interp_engine import EagerModel

    make = _load("make_fixtures")
    model = EagerModel(MODEL, device="cpu", dtype="float32", attn_implementation="eager")
    asyncio.run(make.write_all(model, MODEL, tmp_path))
    path = next((tmp_path / MODEL.replace("/", "--") / "serves").glob("*.json"))
    payload = json.loads(path.read_text())
    payload["digest"] = "0" * 16
    path.write_text(json.dumps(payload))
    problems = _run_all(model, tmp_path, asyncio.run)
    assert any("older case" in p for p in problems)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the vLLM backend initializes on CUDA")
def test_vllm_matches_the_eager_reference() -> None:
    from tests.harness import require_vllm

    require_vllm()
    _vllm_matches("vllm", gpu_memory_utilization=0.2)


def _vllm_matches(backend: str, **kwargs: Any) -> None:
    from interp_engine import load_model

    root = _on_disk()
    loop = asyncio.new_event_loop()
    try:
        model = load_model(MODEL, backend=backend, dtype="float32", max_model_len=512, **kwargs)
        loop.run_until_complete(model.warmup())
        try:
            problems = _run_all(model, root, loop.run_until_complete)
        finally:
            loop.run_until_complete(model.shutdown())
    finally:
        loop.close()
    assert not problems, "\n".join(problems)
