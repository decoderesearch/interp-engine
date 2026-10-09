"""``sample_from_embeds``: n completions of one embeds prompt, one vLLM request or a seeded loop.

The vLLM half drives the real method against a fake engine and asserts on what was asked for and on
how the completions come back: CPU-only, no vLLM needed. The loop half runs on eager GPT-2, where
the per-completion seeds must reproduce ``generate_steps_from_embeds`` exactly.
"""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any

import pytest
import torch

from interp_engine import EagerModel, EmbedsSample, sample_from_embeds
from interp_engine.sampling import RecommendedSampling
from interp_engine.vllm_backend import VLLMModel

WIDTH = 8
EOS = 99


class _Completion:
    def __init__(self, index: int, token_ids: tuple[int, ...], finish_reason: str, stop_reason: Any = None) -> None:
        self.index = index
        self.token_ids = token_ids
        self.finish_reason = finish_reason
        self.stop_reason = stop_reason


class _FakeEngine:
    """Answers one request with a fixed set of completions, out of index order as vLLM may."""

    model_config = types.SimpleNamespace(dtype=torch.bfloat16)

    def __init__(self) -> None:
        self.requests: list[tuple[dict, Any, Any]] = []

    async def generate(self, prompt: dict, sampling_params: Any, request_id: str, lora_request: Any = None):
        self.requests.append((prompt, sampling_params, lora_request))
        yield types.SimpleNamespace(
            outputs=[
                _Completion(2, (7, 8, 9), "length"),
                _Completion(0, (1, 2, EOS), "stop"),
                _Completion(1, (3, EOS), "stop", stop_reason=EOS),
            ]
        )


class _Tokenizer:
    def decode(self, ids: list[int], **kw: Any) -> str:
        return " ".join(str(i) for i in ids)


@pytest.fixture
def fake_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    """The method imports ``SamplingParams`` and ``RequestOutputKind`` at call time."""
    vllm = types.ModuleType("vllm")
    params = types.ModuleType("vllm.sampling_params")

    class SamplingParams:
        def __init__(self, **kw: Any) -> None:
            self.kw = kw

    class RequestOutputKind:
        FINAL_ONLY = "final_only"

    lora = types.ModuleType("vllm.lora.request")
    lora.LoRARequest = lambda name, lora_id, path: types.SimpleNamespace(lora_path=path)  # type: ignore[attr-defined]
    vllm.SamplingParams = SamplingParams  # type: ignore[attr-defined]
    params.RequestOutputKind = RequestOutputKind  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", params)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", lora)


def _model(engine: _FakeEngine) -> Any:
    """A VLLMModel with only the attributes this path touches (``__init__`` loads a real model)."""
    model = object.__new__(VLLMModel)
    model.engine = engine
    model._engine_loop = None
    model._hidden_size = WIDTH
    model._global_intervention = None
    model.enable_prompt_embeds = True
    model.max_lora_rank = 16
    model._lora_ids = {}
    model.tokenizer = _Tokenizer()
    model._recommended_sampling = RecommendedSampling()
    return model


def test_vllm_runs_one_request_for_all_n(fake_vllm: None) -> None:
    """One request carries n, the seed, no detokenizing and a final-only report; ids drop the stop token."""
    engine = _FakeEngine()
    rows = torch.zeros(5, WIDTH)
    out = asyncio.run(
        sample_from_embeds(_model(engine), rows, n=3, max_tokens=3, temperature=1.0, seed=40, lora_path="/lora")
    )
    assert len(engine.requests) == 1
    prompt, sampling, lora = engine.requests[0]
    assert sampling.kw["n"] == 3 and sampling.kw["seed"] == 40 and sampling.kw["max_tokens"] == 3
    assert sampling.kw["detokenize"] is False and sampling.kw["output_kind"] == "final_only"
    assert prompt["prompt_embeds"].dtype == torch.bfloat16 and prompt["prompt_embeds"].shape == (5, WIDTH)
    assert lora is not None and lora.lora_path == "/lora"
    assert out == [
        EmbedsSample("1 2", [1, 2], "eos"),
        EmbedsSample("3", [3], "eos"),
        EmbedsSample("7 8 9", [7, 8, 9], "length"),
    ]


def test_vllm_refuses_before_the_request(fake_vllm: None) -> None:
    """n < 1 and an engine without prompt embeds fail by name, with nothing sent."""
    engine = _FakeEngine()
    model = _model(engine)
    with pytest.raises(ValueError, match="n must be at least 1"):
        asyncio.run(model.sample_from_embeds(torch.zeros(2, WIDTH), n=0))
    model.enable_prompt_embeds = False
    with pytest.raises(ValueError, match="enable_prompt_embeds=True"):
        asyncio.run(model.sample_from_embeds(torch.zeros(2, WIDTH), n=2))
    assert engine.requests == []


def test_the_loop_seeds_each_completion_as_vllm_does(gpt2: EagerModel) -> None:
    """Completion j is generate_steps_from_embeds with seed + j; a rerun gives the same set."""

    async def one(rows: torch.Tensor, seed: int) -> list[int]:
        return [
            s.token_id async for s in gpt2.generate_steps_from_embeds(rows, max_tokens=5, temperature=1.0, seed=seed)
        ]

    ids = [int(t) for t in gpt2.to_tokens("The capital of France is")[0]]
    rows = gpt2.arch.embed(torch.tensor([ids], device=gpt2.device))[0]
    got = asyncio.run(sample_from_embeds(gpt2, rows, n=3, max_tokens=5, temperature=1.0, seed=11))
    again = asyncio.run(sample_from_embeds(gpt2, rows, n=3, max_tokens=5, temperature=1.0, seed=11))
    assert got == again
    for j, s in enumerate(got):
        want = asyncio.run(one(rows, 11 + j))
        assert s.token_ids == (want[:-1] if s.finish == "eos" else want)
        assert s.text == gpt2.tokenizer.decode(s.token_ids, clean_up_tokenization_spaces=False)
    with pytest.raises(ValueError, match="lora_path"):
        asyncio.run(sample_from_embeds(gpt2, rows, n=1, lora_path="/lora"))
