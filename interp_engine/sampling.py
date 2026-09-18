"""How a generation's sampling knobs are decided, on every backend and for every caller.

Three layers, each a type here:

- :class:`RecommendedSampling` is what the checkpoint's ``generation_config.json`` states. The file
  is Hugging Face's, so it carries only what ``transformers`` can express -- ``temperature``,
  ``top_k``, ``top_p``, ``do_sample`` -- and a presence penalty has no field there.
- :func:`resolve_sampling` turns a caller's arguments into :class:`SamplingSettings`: a knob the
  caller passes is used as passed, a knob left ``None`` takes the checkpoint's recommendation, and
  a knob neither states falls back to the engine's own neutral value. This is what ``transformers``
  and ``vllm serve`` both do, so a checkpoint samples here the way its authors tuned it to.
- :class:`SamplingSettings` is what the generation then runs with, every knob decided. A server
  reports it beside the completion so the reader sees the settings that produced the text.

The penalty here is the presence penalty and nothing else: one flat subtraction from the logit of
every token the generation has produced so far, before the temperature (vLLM's order). It breaks
a loop without scaling with how long the loop ran, which is what a frequency penalty does and why
that one punishes function words in a long reply. ``transformers``' multiplicative
``repetition_penalty`` is read from the file but never applied; no checkpoint this engine serves
states one.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypedDict

import torch

logger = logging.getLogger(__name__)

GENERATION_CONFIG = "generation_config.json"


@dataclass(frozen=True)
class RecommendedSampling:
    """The checkpoint's stated sampling defaults, in the engine's vocabulary.

    ``None`` means the file does not state the value, or states the value ``transformers`` treats
    as "off" (``top_k`` of 0, ``top_p`` of 1.0, ``repetition_penalty`` of 1.0), which is what
    ``None`` means to a generation. ``do_sample`` is kept as stated: ``False`` means the checkpoint
    recommends greedy decoding whatever the other fields say, and :func:`resolve_sampling` reads it
    as a temperature of 0.
    """

    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    do_sample: bool | None = None
    source: str | None = None
    """The path of the file read, or ``None`` when the checkpoint has none."""

    @property
    def is_empty(self) -> bool:
        """No field is stated. True for a checkpoint without the file (Qwen3.5) and for one whose
        file names only token ids (GPT-2)."""
        return (
            self.temperature is None
            and self.top_k is None
            and self.top_p is None
            and self.repetition_penalty is None
            and self.do_sample is None
        )


class VLLMSamplingKwargs(TypedDict):
    """``vllm.SamplingParams`` keywords, which spell "no filtering" as ``top_k=-1`` and
    ``top_p=1.0``."""

    temperature: float
    top_k: int
    top_p: float
    presence_penalty: float


@dataclass(frozen=True)
class SamplingSettings:
    """What a generation runs with, every knob decided; see :func:`resolve_sampling`.

    ``top_k`` and ``top_p`` are ``None`` for no filtering. ``temperature`` of 0 is greedy.
    ``presence_penalty`` of 0 is none.
    """

    temperature: float
    top_k: int | None
    top_p: float | None
    presence_penalty: float

    def as_dict(self) -> dict[str, float | int | None]:
        """The settings as a JSON-ready mapping, for a server to report beside a completion."""
        return {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
        }

    def vllm_kwargs(self) -> VLLMSamplingKwargs:
        """The settings as ``vllm.SamplingParams`` keywords: ``SamplingParams(max_tokens=n,
        **settings.vllm_kwargs())``."""
        return VLLMSamplingKwargs(
            temperature=self.temperature,
            top_k=-1 if self.top_k is None else int(self.top_k),
            top_p=1.0 if self.top_p is None else float(self.top_p),
            presence_penalty=self.presence_penalty,
        )


def resolve_sampling(
    recommended: RecommendedSampling | None,
    *,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    presence_penalty: float | None = None,
) -> SamplingSettings:
    """Decide every knob: the caller's value, else the checkpoint's, else the engine's neutral one.

    A caller that wants no filtering where the checkpoint recommends some says so in
    ``transformers``' terms, ``top_k=0`` or ``top_p=1.0``, and gets ``None`` back for it. A
    ``do_sample`` of ``False`` in the file is a recommendation of greedy decoding, so a caller who
    leaves ``temperature`` unset gets 0 there. The presence penalty has no field in the file, so
    unset means 0: a checkpoint's card may recommend one (Qwen3.5 says 1.5), and the caller who
    reads that card passes it.
    """
    stated = recommended or RecommendedSampling()
    if temperature is None:
        greedy = stated.do_sample is False
        temperature = 0.0 if greedy else (1.0 if stated.temperature is None else stated.temperature)
    if top_k is None:
        top_k = stated.top_k
    if top_p is None:
        top_p = stated.top_p
    return SamplingSettings(
        temperature=float(temperature),
        top_k=None if top_k is None or int(top_k) <= 0 else int(top_k),
        top_p=None if top_p is None or float(top_p) >= 1.0 else float(top_p),
        presence_penalty=0.0 if presence_penalty is None else float(presence_penalty),
    )


def apply_presence_penalty(logits: torch.Tensor, generated: Sequence[int], penalty: float) -> torch.Tensor:
    """``logits`` with ``penalty`` subtracted at every id in ``generated``; the input when there is
    nothing to subtract.

    The prompt's tokens are not penalized, only the generation's, as vLLM has it: a reply that
    quotes its prompt is not a loop. Applied before the temperature, so a penalty means the same
    number of logits whatever the temperature is.
    """
    if not penalty or not generated:
        return logits
    ids = torch.tensor(sorted({int(t) for t in generated}), device=logits.device)
    out = logits.clone()
    out[ids] -= penalty
    return out


class PresencePenaltyProcessor:
    """A ``transformers`` logits processor that subtracts ``penalty`` at every token generated so far.

    ``input_ids`` is the prompt followed by the generation; only the generation counts, which is
    what ``prompt_len`` is for (0 when generating from ``inputs_embeds``, where ``generate`` holds
    no prompt ids). ``generate`` appends its temperature warper after caller-supplied processors,
    so this runs before the temperature: the order every backend here uses. Duck-typed rather than
    subclassing ``LogitsProcessor`` so this module does not import ``transformers`` at load.
    """

    def __init__(self, penalty: float, prompt_len: int) -> None:
        self.penalty = penalty
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        rows = [
            apply_presence_penalty(row_scores, row_ids[self.prompt_len :].tolist(), self.penalty)
            for row_ids, row_scores in zip(input_ids, scores, strict=True)
        ]
        return torch.stack(rows)


def hf_generate_kwargs(settings: SamplingSettings, *, prompt_len: int) -> dict[str, Any]:
    """``settings`` as ``transformers.generate`` keywords, every knob stated so the checkpoint's
    own ``generation_config`` adds nothing.

    Temperature 0 is greedy (``generate`` raises on a literal 0), and the sampling knobs are left
    out on that path because ``generate`` warns about each flag it will not use. The file's
    multiplicative ``repetition_penalty`` is turned off: the presence penalty is the repetition
    control, and it is applied on the greedy path too, as vLLM has it.
    """
    from transformers import LogitsProcessorList

    kwargs: dict[str, Any] = {"repetition_penalty": 1.0}
    if settings.presence_penalty > 0:
        kwargs["logits_processor"] = LogitsProcessorList(
            [PresencePenaltyProcessor(settings.presence_penalty, prompt_len)]  # type: ignore[list-item]
        )
    if settings.temperature <= 0:
        kwargs["do_sample"] = False
        return kwargs
    kwargs["do_sample"] = True
    kwargs["temperature"] = settings.temperature
    # ``transformers`` spells "no filtering" as 0 and 1.0, where the engine spells it None.
    kwargs["top_k"] = 0 if settings.top_k is None else settings.top_k
    kwargs["top_p"] = 1.0 if settings.top_p is None else settings.top_p
    kwargs["min_p"] = None
    kwargs["typical_p"] = 1.0
    return kwargs


def parse_recommended_sampling(config: dict[str, Any], *, source: str | None = None) -> RecommendedSampling:
    """The sampling fields of a parsed ``generation_config.json``; see :class:`RecommendedSampling`."""

    def number(key: str) -> float | None:
        value = config.get(key)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    temperature = number("temperature")
    top_k = number("top_k")
    top_p = number("top_p")
    repetition_penalty = number("repetition_penalty")
    do_sample = config.get("do_sample")
    return RecommendedSampling(
        temperature=temperature,
        top_k=int(top_k) if top_k is not None and top_k > 0 else None,
        top_p=top_p if top_p is not None and top_p < 1.0 else None,
        repetition_penalty=repetition_penalty if repetition_penalty is not None and repetition_penalty != 1.0 else None,
        do_sample=do_sample if isinstance(do_sample, bool) else None,
        source=source,
    )


def read_recommended_sampling(hf_model_id: str) -> RecommendedSampling:
    """Read the checkpoint's ``generation_config.json`` from the Hub cache or a local directory.

    Empty when the checkpoint has no such file, or when it cannot be reached (offline, no such
    repo): the recommendation is a courtesy, and a missing one must not fail a load.
    """
    from transformers.utils.hub import cached_file

    try:
        path = cached_file(
            hf_model_id,
            GENERATION_CONFIG,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
            _raise_exceptions_for_gated_repo=False,
        )
    except (OSError, ValueError) as exc:
        logger.debug("no %s for %s (%s)", GENERATION_CONFIG, hf_model_id, exc)
        return RecommendedSampling()
    if path is None:
        return RecommendedSampling()
    try:
        with open(path, encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, ValueError) as exc:
        logger.debug("unreadable %s at %s (%s)", GENERATION_CONFIG, path, exc)
        return RecommendedSampling()
    if not isinstance(config, dict):
        return RecommendedSampling()
    return parse_recommended_sampling(config, source=str(path))
