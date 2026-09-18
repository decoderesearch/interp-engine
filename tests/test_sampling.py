"""A generation's knobs: the caller's, else the checkpoint's, else neutral; and the presence penalty.

The reader is tested against files written here, so the cases do not depend on what a Hub
checkpoint ships this month. The resolution rules are pure. The penalty is checked on tensors and
then through the eager loop, where a penalty large enough must stop a token from repeating.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import torch
from harness import GPT2, load_model

from interp_engine import (
    EagerModel,
    RecommendedSampling,
    SamplingSettings,
    apply_presence_penalty,
    generate_stream,
    read_recommended_sampling,
    resolve_sampling,
)
from interp_engine.sampling import GENERATION_CONFIG, parse_recommended_sampling

GEMMA_LIKE = RecommendedSampling(temperature=1.0, top_k=64, top_p=0.95, do_sample=True, source="x")


# --- reading ------------------------------------------------------------------


def _checkpoint(tmp_path: Path, config: dict | None) -> Path:
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({"model_type": "gpt2"}))
    if config is not None:
        (directory / GENERATION_CONFIG).write_text(json.dumps(config))
    return directory


def test_a_stated_file_is_read_as_stated(tmp_path: Path) -> None:
    """Gemma 4's file: every field named, the temperature at its nominal value still counts."""
    directory = _checkpoint(tmp_path, {"do_sample": True, "temperature": 1.0, "top_k": 64, "top_p": 0.95})
    got = read_recommended_sampling(str(directory))
    assert got == RecommendedSampling(
        temperature=1.0, top_k=64, top_p=0.95, do_sample=True, source=str(directory / GENERATION_CONFIG)
    )


def test_off_values_read_as_none() -> None:
    """``transformers`` spells "no filtering" as 0, 1.0 and 1.0; the engine spells it ``None``."""
    got = parse_recommended_sampling({"top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0, "temperature": 0.7})
    assert got == RecommendedSampling(temperature=0.7)


def test_a_file_with_only_token_ids_is_empty_but_sourced(tmp_path: Path) -> None:
    directory = _checkpoint(tmp_path, {"bos_token_id": 50256, "eos_token_id": 50256})
    got = read_recommended_sampling(str(directory))
    assert got.is_empty and got.source == str(directory / GENERATION_CONFIG)


def test_no_file_and_no_checkpoint_are_empty_not_errors(tmp_path: Path) -> None:
    assert read_recommended_sampling(str(_checkpoint(tmp_path, None))) == RecommendedSampling()
    assert read_recommended_sampling(str(tmp_path / "nowhere")) == RecommendedSampling()


def test_wrong_types_are_ignored() -> None:
    assert (
        parse_recommended_sampling({"temperature": "hot", "top_k": True, "do_sample": "yes"}) == RecommendedSampling()
    )


# --- resolving ----------------------------------------------------------------


def test_unset_knobs_take_the_checkpoints_recommendation() -> None:
    assert resolve_sampling(GEMMA_LIKE) == SamplingSettings(temperature=1.0, top_k=64, top_p=0.95, presence_penalty=0.0)


def test_a_passed_knob_wins_over_the_recommendation() -> None:
    got = resolve_sampling(GEMMA_LIKE, temperature=0.3, top_k=5, presence_penalty=1.5)
    assert got == SamplingSettings(temperature=0.3, top_k=5, top_p=0.95, presence_penalty=1.5)


def test_a_caller_turns_filtering_off_in_transformers_terms() -> None:
    """``top_k=0`` and ``top_p=1.0`` mean "none", and beat a recommendation of some."""
    got = resolve_sampling(GEMMA_LIKE, top_k=0, top_p=1.0)
    assert got.top_k is None and got.top_p is None


def test_nothing_stated_is_neutral() -> None:
    """Qwen3.5 ships no file: temperature 1, no filtering, no penalty."""
    assert resolve_sampling(RecommendedSampling()) == SamplingSettings(1.0, None, None, 0.0)
    assert resolve_sampling(None) == SamplingSettings(1.0, None, None, 0.0)


def test_do_sample_false_is_a_recommendation_of_greedy() -> None:
    stated = RecommendedSampling(temperature=0.7, do_sample=False)
    assert resolve_sampling(stated).temperature == 0.0
    assert resolve_sampling(stated, temperature=0.7).temperature == 0.7


def test_the_settings_speak_vllm() -> None:
    assert SamplingSettings(0.6, None, None, 1.5).vllm_kwargs() == {
        "temperature": 0.6,
        "top_k": -1,
        "top_p": 1.0,
        "presence_penalty": 1.5,
    }
    assert SamplingSettings(0.6, 20, 0.95, 0.0).as_dict() == {
        "temperature": 0.6,
        "top_k": 20,
        "top_p": 0.95,
        "presence_penalty": 0.0,
    }


# --- the penalty --------------------------------------------------------------


def test_the_penalty_is_flat_per_distinct_token_and_leaves_the_rest() -> None:
    logits = torch.zeros(6)
    out = apply_presence_penalty(logits, [1, 1, 1, 4], 1.5)
    assert out.tolist() == [0.0, -1.5, 0.0, 0.0, -1.5, 0.0]
    assert logits.tolist() == [0.0] * 6, "the input is not written"


def test_no_penalty_or_nothing_generated_returns_the_input_itself() -> None:
    logits = torch.zeros(3)
    assert apply_presence_penalty(logits, [], 1.5) is logits
    assert apply_presence_penalty(logits, [1], 0.0) is logits


def test_a_large_penalty_stops_greedy_repetition_on_eager() -> None:
    """Greedy GPT-2 repeats itself within a few tokens on a bare prompt; a big presence penalty
    forbids any repeat, so every generated id is distinct. Greedy is penalized too, as vLLM does."""
    model = load_model(GPT2, device="cpu")
    assert isinstance(model, EagerModel)
    ids = model.to_tokens("the the the the", prepend_bos=False)
    plain = [s.token_id for s in generate_stream(model, ids, max_tokens=12, temperature=0.0)]
    penalized = [
        s.token_id for s in generate_stream(model, ids, max_tokens=12, temperature=0.0, presence_penalty=100.0)
    ]
    assert len(set(plain)) < len(plain), "the plain greedy run should repeat, or this test proves nothing"
    assert len(set(penalized)) == len(penalized)


def test_the_eager_protocol_methods_take_the_new_knobs() -> None:
    model = load_model(GPT2, device="cpu")
    ids = model.to_tokens("The capital of France is", prepend_bos=False)[0].tolist()
    settings = model.sampling_settings(temperature=0.0, presence_penalty=2.0)
    assert settings == SamplingSettings(temperature=0.0, top_k=None, top_p=None, presence_penalty=2.0)

    async def run() -> tuple[str, list[str]]:
        text = await model.generate_text(ids, max_tokens=4, temperature=0.0, presence_penalty=2.0)
        deltas = [d async for d in model.generate_stream(ids, max_tokens=4, temperature=0.0, presence_penalty=2.0)]
        return text, deltas

    text, deltas = asyncio.run(run())
    assert text == "".join(deltas)


# --- transformers.generate ----------------------------------------------------


def test_the_hf_kwargs_state_every_knob_and_turn_the_files_own_penalty_off() -> None:
    from interp_engine import hf_generate_kwargs

    kwargs = hf_generate_kwargs(SamplingSettings(0.6, 20, 0.95, 0.0), prompt_len=3)
    assert kwargs == {
        "repetition_penalty": 1.0,
        "do_sample": True,
        "temperature": 0.6,
        "top_k": 20,
        "top_p": 0.95,
        "min_p": None,
        "typical_p": 1.0,
    }
    assert hf_generate_kwargs(SamplingSettings(1.0, None, None, 0.0), prompt_len=3)["top_k"] == 0
    assert hf_generate_kwargs(SamplingSettings(1.0, None, None, 0.0), prompt_len=3)["top_p"] == 1.0


def test_greedy_hf_kwargs_leave_the_sampling_knobs_out_but_keep_the_penalty() -> None:
    from interp_engine import hf_generate_kwargs

    kwargs = hf_generate_kwargs(SamplingSettings(0.0, 20, 0.95, 1.5), prompt_len=3)
    assert kwargs["do_sample"] is False
    assert "temperature" not in kwargs and "top_k" not in kwargs
    assert len(kwargs["logits_processor"]) == 1


def test_the_processor_penalizes_the_generation_and_not_the_prompt() -> None:
    from interp_engine import PresencePenaltyProcessor

    scores = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    ids = torch.tensor([[0, 0, 1, 1, 3], [0, 0, 2, 2, 2]])
    out = PresencePenaltyProcessor(0.5, prompt_len=2)(ids, scores)
    assert out.tolist() == [[1.0, 1.5, 3.0, 3.5], [1.0, 2.0, 2.5, 4.0]]


def test_transformers_generate_runs_with_the_kwargs_and_the_penalty_stops_greedy_repeats() -> None:
    """The same run as the eager-loop test, through ``generate``: GPT-2 greedy on a bare prompt
    repeats; with the penalty every generated id is distinct."""
    from interp_engine import hf_generate_kwargs

    model = load_model(GPT2, device="cpu")
    assert isinstance(model, EagerModel)
    ids = model.to_tokens("the the the the", prepend_bos=False)
    prompt_len = int(ids.shape[-1])

    def run(settings: SamplingSettings) -> list[int]:
        out = model.hf_model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            pad_token_id=model.tokenizer.eos_token_id,
            max_new_tokens=12,
            **hf_generate_kwargs(settings, prompt_len=prompt_len),
        )
        return out[0, prompt_len:].tolist()

    plain = run(SamplingSettings(0.0, None, None, 0.0))
    penalized = run(SamplingSettings(0.0, None, None, 100.0))
    assert len(set(plain)) < len(plain)
    assert len(set(penalized)) == len(penalized)
