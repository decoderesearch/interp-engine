"""A hybrid trunk whose Mamba cache holds fewer blocks than ``max_num_seqs`` starts on the retry.

vLLM measures the block count inside its engine-core child and refuses to start when there are
fewer blocks than sequences. The worker records the count, and the parent starts again with
``max_num_seqs`` set to it. Stand-ins replace vLLM: what is under test is the handoff between the
two halves, which needs no GPU.
"""

from __future__ import annotations

import os
import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest

from interp_engine import vllm_seq_fit
from interp_engine.vllm_backend import VLLMModel
from interp_engine.vllm_seq_fit import (
    SEQ_FIT_ENV,
    mamba_block_limit,
    patch_mamba_seq_check,
    read_block_limit,
    record_block_limit,
)


class _KVConfig:
    def __init__(self, num_blocks: int, has_mamba_layers: bool = True) -> None:
        self.num_blocks = num_blocks
        self.has_mamba_layers = has_mamba_layers


def test_block_limit_only_when_mamba_blocks_are_short() -> None:
    assert mamba_block_limit(_KVConfig(88), 128) == 88
    assert mamba_block_limit(_KVConfig(128), 128) is None, "enough blocks"
    assert mamba_block_limit(_KVConfig(88, has_mamba_layers=False), 128) is None, "no Mamba layers"
    assert mamba_block_limit(None, 128) is None
    assert mamba_block_limit(_KVConfig(88), None) is None


def test_record_and_read_round_trip(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    path = str(tmp_path / "fit")
    monkeypatch.setenv(SEQ_FIT_ENV, path)
    record_block_limit(88)
    assert read_block_limit(path) == 88


def test_read_ignores_a_missing_or_empty_file(tmp_path: Any) -> None:
    empty = tmp_path / "empty"
    empty.write_text("")
    assert read_block_limit(str(empty)) is None
    assert read_block_limit(str(tmp_path / "absent")) is None


def test_record_without_the_env_writes_nothing(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SEQ_FIT_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    record_block_limit(88)
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def fake_compilation(monkeypatch: pytest.MonkeyPatch) -> Iterator[type]:
    """Install a stand-in ``vllm.config.compilation`` whose check matches vLLM's signature."""

    class CompilationConfig:
        def resolve_cudagraph_mode_and_sizes(
            self,
            min_cg_support: Any,
            min_cg_attn_backend: str | None,
            uniform_decode_query_len: int = 1,
            use_v2_model_runner: bool = False,
            tensor_parallel_size: int = 1,
            kv_cache_config: Any = None,
            max_num_reqs: int | None = None,
            is_profiling: bool = False,
        ) -> str:
            blocks = mamba_block_limit(kv_cache_config, max_num_reqs)
            if blocks is not None:
                raise ValueError(f"max_num_seqs ({max_num_reqs}) exceeds available Mamba cache blocks ({blocks}).")
            if min_cg_support == "bad":
                raise ValueError("unrelated")
            return "FULL"

    module = types.ModuleType("vllm.config.compilation")
    module.CompilationConfig = CompilationConfig  # pyright: ignore[reportAttributeAccessIssue]
    for name in ("vllm", "vllm.config"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.config.compilation", module)
    yield CompilationConfig


def test_patched_check_records_the_block_count_and_still_raises(
    fake_compilation: type, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = str(tmp_path / "fit")
    monkeypatch.setenv(SEQ_FIT_ENV, path)
    patch_mamba_seq_check()
    patch_mamba_seq_check()

    config = fake_compilation()
    with pytest.raises(ValueError, match="exceeds available Mamba cache blocks"):
        config.resolve_cudagraph_mode_and_sizes("ok", None, 1, kv_cache_config=_KVConfig(88), max_num_reqs=128)
    assert read_block_limit(path) == 88


def test_patched_check_changes_nothing_else(
    fake_compilation: type, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = str(tmp_path / "fit")
    monkeypatch.setenv(SEQ_FIT_ENV, path)
    patch_mamba_seq_check()

    config = fake_compilation()
    assert (
        config.resolve_cudagraph_mode_and_sizes("ok", None, kv_cache_config=_KVConfig(128), max_num_reqs=128) == "FULL"
    )
    with pytest.raises(ValueError, match="unrelated"):
        config.resolve_cudagraph_mode_and_sizes("bad", None)
    assert not os.path.exists(path), "no block count for an unrelated error"


class _FakeAsyncLLM:
    """Fails like vLLM while ``max_num_seqs`` exceeds ``blocks``, recording the count as the worker does."""

    def __init__(self, blocks: int | None) -> None:
        self.blocks = blocks
        self.calls: list[dict[str, Any]] = []

    def from_engine_args(self, args: dict[str, Any]) -> str:
        self.calls.append(dict(args))
        if self.blocks is not None and args.get("max_num_seqs", 128) > self.blocks:
            record_block_limit(self.blocks)
            raise RuntimeError("Engine core initialization failed. See root cause above.")
        return "engine"


def _model(extra: dict[str, Any] | None = None) -> VLLMModel:
    model = object.__new__(VLLMModel)
    model.hf_model_id = "stand-in/hybrid"
    model._engine_kwargs = {"model": "stand-in/hybrid", **(extra or {})}
    model._max_num_seqs_pinned = "max_num_seqs" in (extra or {})
    return model


def test_start_engine_retries_with_the_recorded_block_count() -> None:
    llm = _FakeAsyncLLM(blocks=88)
    model = _model()

    assert model._start_engine(llm, dict) == "engine"
    assert [c.get("max_num_seqs") for c in llm.calls] == [None, 88]
    assert model._engine_kwargs["max_num_seqs"] == 88
    assert SEQ_FIT_ENV not in os.environ


def test_start_engine_starts_once_when_it_fits() -> None:
    llm = _FakeAsyncLLM(blocks=None)

    assert _model()._start_engine(llm, dict) == "engine"
    assert len(llm.calls) == 1


def test_start_engine_reraises_a_failure_with_no_block_count() -> None:
    class Broken(_FakeAsyncLLM):
        def from_engine_args(self, args: dict[str, Any]) -> str:
            self.calls.append(dict(args))
            raise RuntimeError("out of memory")

    llm = Broken(blocks=None)
    with pytest.raises(RuntimeError, match="out of memory"):
        _model()._start_engine(llm, dict)
    assert len(llm.calls) == 1


def test_an_explicit_max_num_seqs_wins_and_is_not_retried() -> None:
    llm = _FakeAsyncLLM(blocks=88)
    model = _model({"max_num_seqs": 128})

    with pytest.raises(RuntimeError, match="initialization failed"):
        model._start_engine(llm, dict)
    assert len(llm.calls) == 1
    assert model._engine_kwargs["max_num_seqs"] == 128


def test_the_worker_plugin_installs_the_patch() -> None:
    import interp_engine.vllm_plugin as plugin

    assert plugin.patch_mamba_seq_check is vllm_seq_fit.patch_mamba_seq_check
