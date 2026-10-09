"""``describe()``: one record of what a model serves, built from ``serves`` and the load config.

The vLLM rows use an engine-less ``VLLMModel``, as ``test_point_capability`` does, since every
answer here is defined to need no engine.
"""

from __future__ import annotations

import pytest

from interp_engine import Address, EagerModel, VLLMModel
from interp_engine.residual_basis import ResidualBasis

GPT2 = "openai-community/gpt2"


def _vllm(
    cls: type[VLLMModel] = VLLMModel,
    *,
    enforce_eager: bool = True,
    reads: tuple[Address, ...] = (),
    extraction: bool = False,
) -> VLLMModel:
    model = object.__new__(cls)
    model._engine_kwargs = {"enforce_eager": enforce_eager}
    model._residual_basis = ResidualBasis()
    model._static_reads = frozenset(reads)
    model._static_writes = frozenset()
    model.tensor_parallel_size = 1
    model.hf_model_id = GPT2
    model.num_hidden_layers = 12
    model._hidden_size = 768
    model._attn_dims = {"n_heads": 12, "n_kv_heads": 12, "head_dim": 64}
    model.enable_extraction = extraction
    return model


@pytest.fixture(scope="module")
def gpt2() -> EagerModel:
    return EagerModel(GPT2, device="cpu", dtype="float32", attn_implementation="eager")


def test_eager_describes_what_serves_answers(gpt2: EagerModel) -> None:
    d = gpt2.describe()
    assert (d.backend, d.n_layers, d.d_model, d.n_heads, d.n_kv_heads, d.head_dim) == ("eager", 12, 768, 12, 12, 64)
    assert d.hooks_available and not d.graph_replay
    assert d.static_points == [] and d.static_writes == []
    assert d.capture_points == sorted(d.capture_points)
    assert {"resid_post", "attn_probs", "mlp_out"} <= set(d.capture_points)
    assert all(gpt2.serves(name, 0) or gpt2.serves(name) for name in d.capture_points)
    assert d.residual_readable and d.attention


def test_a_hooked_vllm_engine_serves_its_table() -> None:
    d = _vllm().describe()
    assert d.backend == "vllm"
    assert "resid_post" in d.capture_points
    assert d.residual_readable


def test_a_static_engine_serves_only_its_taps() -> None:
    d = _vllm(enforce_eager=False, reads=(Address("resid_post", 6),)).describe()
    assert d.backend == "vllm-static"
    assert d.graph_replay and not d.hooks_available
    assert d.static_points == [Address("resid_post", 6)]
    assert d.residual_readable
    assert not d.attention


def test_a_generate_engine_serves_nothing() -> None:
    d = _vllm(enforce_eager=False).describe()
    assert d.backend == "vllm-generate"
    assert d.capture_points == []
    assert not d.residual_readable and not d.attention


def test_native_extraction_makes_the_residual_readable_and_nothing_else() -> None:
    d = _vllm(enforce_eager=False, extraction=True).describe()
    assert d.residual_readable
    assert d.capture_points == []
