"""LoRA reads: one activation read out as text by a LoRA adapter on the model.

The base model plus a PEFT adapter trained to describe one residual vector in words. The vector
goes into a fixed prompt, in place of the embedding of one marker character, and the adapter-on
model writes ``- `` bullets.

The read contract travels together or not at all: the prompt, the marker scan, the transform
(``alpha * h / ||h||``), and the layers the adapter was trained on. :class:`OracleContract` holds
it, and :meth:`OracleContract.from_run` reads it off the checkpoint's ``run.json`` so a caller
cannot mix one checkpoint's alpha with another's prompt.

Two backends read. Eager runs the adapter as forward hooks that exist only while the read
generates, so a capture on the same model always sees the base weights. vLLM runs each layer as
one embeds-prompt request with a ``LoRARequest``, and the requests run together; an engine for
this needs ``enable_prompt_embeds=True`` and ``max_lora_rank``. Every read is greedy and stops at
``max_bullets`` finished bullets, which is where most of its cost goes.

Prefix caching helps a read little: the marker sits about a dozen tokens into the prompt and
everything after it depends on the activation. Batching the layers is what pays.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncGenerator, Iterator, Mapping, Sequence
from contextlib import aclosing, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, overload

import torch

from interp_engine.dispatch import refuse

#: The prompts a checkpoint may name in ``run.json``'s ``extra.prompt``. On Qwen3.6-27B the line
#: breaks of ``concepts_raw`` give a higher greedy log-prob than spaces. ``concepts_raw_inline``
#: is the same text on one line.
PROMPTS: dict[str, str] = {
    "concepts_raw": (
        "An activation vector from layer {layer} of a language model is enclosed in activation tags:\n"
        "<activation>{char}</activation>. Produce distinct concepts that encode this activation, each as\n"
        "a '- ' bullet on its own line."
    ),
    "concepts_raw_inline": (
        "An activation vector from layer {layer} of a language model is enclosed in activation "
        "tags: <activation>{char}</activation>. Produce distinct concepts that encode this "
        "activation, each as a '- ' bullet on its own line."
    ),
}

#: Where the marker scan looks: characters rare enough that none carries meaning of its own.
MARKER_RANGE = (0x3200, 0x3400)

#: The adapter's trained band, and a cheaper subset of it.
ALL_LAYERS: tuple[int, ...] = tuple(range(20, 61, 4))
FAST_LAYERS: tuple[int, ...] = (20, 36, 44, 52, 60)

#: The files of a checkpoint; at a repo root, only these are downloaded.
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors", "run.json")

#: How a checkpoint may name its decoder layers. A vision-language one (Qwen3.5/3.6) keeps its
#: text trunk under ``model.language_model``; a text-only one (Qwen3) under ``model``.
TRUNK_PREFIXES = ("model.language_model.layers.", "model.layers.")

_BULLET = re.compile(r"^\s*[-*]\s+(.*\S)\s*$")


@dataclass(frozen=True)
class OracleContract:
    """How one checkpoint must be read. Reading with another checkpoint's numbers is off-contract."""

    prompt_kind: str = "concepts_raw"
    transform: str = "unit"
    alpha: float = 16000.0
    layers: tuple[int, ...] = ALL_LAYERS

    def __post_init__(self) -> None:
        if self.prompt_kind not in PROMPTS:
            raise ValueError(f"Unknown read prompt {self.prompt_kind!r}; known: {sorted(PROMPTS)}.")
        if self.transform != "unit":
            raise ValueError(f"Read transform {self.transform!r} is not supported; only 'unit' is.")

    @property
    def template(self) -> str:
        return PROMPTS[self.prompt_kind]

    @classmethod
    def from_run(cls, run: Mapping[str, Any]) -> OracleContract:
        """The contract in a checkpoint's ``run.json``."""
        cfg = run["config"]
        return cls(
            prompt_kind=str(cfg.get("extra", {}).get("prompt", "concepts_raw")),
            transform=str(cfg.get("transform", "unit")),
            alpha=float(cfg["alpha"]),
            layers=tuple(int(x) for x in cfg["layers"]),
        )

    def check_layers(self, layers: Sequence[int]) -> tuple[int, ...]:
        """``layers`` as a tuple, refused if any is outside the trained band."""
        out = tuple(int(x) for x in layers)
        off = sorted(set(out) - set(self.layers))
        if off:
            raise ValueError(
                f"Layers {off} are outside this adapter's trained band {list(self.layers)}; the adapter "
                "never saw them, so a read there is off-contract."
            )
        return out


@dataclass(frozen=True)
class OraclePrompt:
    """One layer's rendered prompt, and the row the activation replaces."""

    layer: int
    token_ids: tuple[int, ...]
    slot: int
    char: str
    char_id: int


def _render(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
    )
    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    return [int(t) for t in ids]


def _prompt_with(tokenizer: Any, contract: OracleContract, layer: int, char: str) -> OraclePrompt | None:
    char_ids = tokenizer.encode(char, add_special_tokens=False)
    if len(char_ids) != 1:
        return None
    ids = _render(tokenizer, contract.template.format(layer=int(layer), char=char))
    if ids.count(char_ids[0]) != 1:
        return None
    return OraclePrompt(int(layer), tuple(ids), ids.index(char_ids[0]), char, int(char_ids[0]))


def find_marker(tokenizer: Any, contract: OracleContract) -> str:
    """The first character in :data:`MARKER_RANGE` that stays one token in every layer's prompt.

    Tokenizing the character alone is not enough: BPE is context-sensitive, and a character that is
    one token alone can merge with its neighbours inside the chat template.
    """
    for cp in range(*MARKER_RANGE):
        char = chr(cp)
        if all(_prompt_with(tokenizer, contract, layer, char) is not None for layer in contract.layers):
            return char
    raise ValueError(
        f"No character in U+{MARKER_RANGE[0]:04X}..U+{MARKER_RANGE[1] - 1:04X} stays one token in the "
        "read prompt for this tokenizer, so there is no slot to put an activation in."
    )


def oracle_prompts(
    tokenizer: Any, contract: OracleContract, layers: Sequence[int], marker: str | None = None
) -> dict[int, OraclePrompt]:
    """Each layer's prompt, all with the same marker."""
    char = marker or find_marker(tokenizer, contract)
    out: dict[int, OraclePrompt] = {}
    for layer in contract.check_layers(layers):
        prompt = _prompt_with(tokenizer, contract, layer, char)
        if prompt is None:
            raise ValueError(f"Marker {char!r} does not stay one token in the layer-{layer} read prompt.")
        out[layer] = prompt
    return out


def inject(rows: torch.Tensor, slot: int, activation: torch.Tensor, alpha: float) -> torch.Tensor:
    """A copy of ``rows`` ([T, d]) with row ``slot`` set to ``alpha * activation / ||activation||``."""
    h = activation.detach().float().reshape(-1)
    norm = h.norm()
    if not torch.isfinite(norm) or norm == 0:
        raise ValueError("A LoRA read cannot take a zero or non-finite activation.")
    out = rows.clone()
    out[slot] = (alpha * h / norm).to(out.device, out.dtype)
    return out


def bullets_of(text: str) -> list[str]:
    """The ``- `` bullets in a read's output, in order."""
    return [m.group(1) for line in text.splitlines() if (m := _BULLET.match(line))]


def bullets_done(text: str, max_bullets: int) -> bool:
    """True once ``max_bullets`` bullets are finished (each ends at a newline)."""
    finished = text.split("\n")[:-1]
    return sum(1 for line in finished if _BULLET.match(line)) >= max_bullets


def _weight_names(base_model: str) -> list[str]:
    """The tensor names of ``base_model``'s checkpoint (a local directory or a Hub repo), from its index or header."""
    local = Path(base_model)
    if local.is_dir():
        index = local / "model.safetensors.index.json"
        if index.exists():
            return list(json.loads(index.read_text())["weight_map"])
        from safetensors import safe_open

        with safe_open(str(local / "model.safetensors"), framework="pt") as f:
            return list(f.keys())
    from huggingface_hub import get_safetensors_metadata, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        return list(
            json.loads(Path(hf_hub_download(base_model, "model.safetensors.index.json")).read_text())["weight_map"]
        )
    except EntryNotFoundError:
        return list(get_safetensors_metadata(base_model).weight_map)


def trunk_prefix(base_model: str) -> str:
    """Which of :data:`TRUNK_PREFIXES` ``base_model``'s checkpoint names its decoder layers with."""
    names = _weight_names(base_model)
    for prefix in TRUNK_PREFIXES:
        if any(name.startswith(prefix) for name in names):
            return prefix
    raise ValueError(f"The checkpoint of {base_model} has no decoder layers under {list(TRUNK_PREFIXES)}.")


@dataclass
class OracleAdapter:
    """A downloaded LoRA read checkpoint: its contract, its LoRA shape, and where its files are."""

    repo: str
    subdir: str
    directory: Path
    contract: OracleContract
    rank: int
    lora_alpha: float
    target_modules: tuple[str, ...]
    _weights: dict[str, torch.Tensor] | None = field(default=None, repr=False)
    _vllm_dir: Path | None = field(default=None, repr=False)

    @property
    def scale(self) -> float:
        return self.lora_alpha / self.rank

    @property
    def name(self) -> str:
        return f"{self.repo}:{self.subdir}"

    @classmethod
    def load(cls, repo: str, subdir: str, revision: str | None = None) -> OracleAdapter:
        """Download (or reuse the cached) checkpoint ``repo/subdir``. An empty ``subdir`` is the repo root."""
        from huggingface_hub import snapshot_download

        patterns: list[str] = [f"{subdir}/*"] if subdir else list(ADAPTER_FILES)
        root = Path(snapshot_download(repo, revision=revision, allow_patterns=patterns))
        return cls.from_directory(root / subdir if subdir else root, repo=repo, subdir=subdir)

    @classmethod
    def from_directory(cls, directory: Path, *, repo: str = "", subdir: str = "") -> OracleAdapter:
        directory = Path(directory)
        config = json.loads((directory / "adapter_config.json").read_text())
        run_path = directory / "run.json"
        contract = OracleContract.from_run(json.loads(run_path.read_text())) if run_path.exists() else OracleContract()
        return cls(
            repo=repo or str(directory),
            subdir=subdir or directory.name,
            directory=directory,
            contract=contract,
            rank=int(config["r"]),
            lora_alpha=float(config["lora_alpha"]),
            target_modules=tuple(config.get("target_modules") or ()),
        )

    def weights(self) -> dict[str, torch.Tensor]:
        """The adapter's tensors, keyed as PEFT saved them."""
        if self._weights is None:
            from safetensors.torch import load_file

            self._weights = load_file(str(self.directory / "adapter_model.safetensors"))
        return self._weights

    def layer_pairs(self) -> dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]]:
        """``(decoder layer, module path inside it) -> (A [r, in], B [out, r])``."""
        pattern = re.compile(r"(?:^|\.)layers\.(\d+)\.(.+)\.lora_([AB])\.weight$")
        found: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
        for key, tensor in self.weights().items():
            m = pattern.search(key)
            if m is None:
                raise ValueError(f"Adapter tensor {key!r} is not inside a decoder layer; it is not supported.")
            found.setdefault((int(m.group(1)), m.group(2)), {})[m.group(3)] = tensor
        pairs = {}
        for where, ab in found.items():
            if set(ab) != {"A", "B"}:
                raise ValueError(f"Adapter module {where} has lora_{sorted(ab)} only; it needs both A and B.")
            pairs[where] = (ab["A"], ab["B"])
        return pairs

    def vllm_dir(self, base_model: str | None = None) -> Path:
        """The adapter keyed as ``base_model``'s checkpoint names its layers: what vLLM's mappers expect.

        PEFT saved the text-only class's names (``model.layers.N``). vLLM maps the checkpoint's own
        names. A Qwen3.5/3.6 checkpoint stores its text trunk as ``model.language_model``, which
        its text-only class maps to ``model.`` and its vision-language class to
        ``language_model.model.``, so this is a copy renamed to that layout. A checkpoint that
        stores ``model.layers`` gets the adapter as saved. vLLM gives no error for a LoRA module it
        cannot place: it serves the base weights there. ``base_model`` defaults to the one in
        ``adapter_config.json``.
        """
        if self._vllm_dir is not None:
            return self._vllm_dir
        from safetensors.torch import save_file

        if base_model is None:
            config = json.loads((self.directory / "adapter_config.json").read_text())
            base_model = config.get("base_model_name_or_path")
            if not base_model:
                raise ValueError("adapter_config.json names no base model; pass base_model.")
        if trunk_prefix(str(base_model)) == "model.layers.":
            self._vllm_dir = self.directory
            return self.directory
        out = self.directory.parent / f"{self.directory.name}.vllm"
        target = out / "adapter_model.safetensors"
        if not target.exists():
            out.mkdir(parents=True, exist_ok=True)
            renamed = {
                re.sub(r"^base_model\.model\.model\.layers\.", "base_model.model.model.language_model.layers.", k): v
                for k, v in self.weights().items()
            }
            (out / "adapter_config.json").write_text((self.directory / "adapter_config.json").read_text())
            tmp = out / "adapter_model.safetensors.tmp"
            save_file(renamed, str(tmp))
            tmp.rename(target)
        self._vllm_dir = out
        return out


@dataclass(frozen=True)
class OracleRead:
    """One layer's read."""

    layer: int
    text: str
    bullets: list[str]
    token_ids: list[int]
    finish: str
    """``"bullets"`` (the cap), ``"eos"``, or ``"length"``."""


@dataclass(frozen=True)
class OraclePartial:
    """One layer's text so far, while its read runs. Its :class:`OracleRead` comes after."""

    layer: int
    text: str


def _read(layer: int, token_ids: list[int], text: str, max_bullets: int, eos: bool) -> OracleRead:
    finish = "bullets" if bullets_done(text, max_bullets) else "eos" if eos else "length"
    return OracleRead(layer, text, bullets_of(text)[:max_bullets], token_ids, finish)


@overload
def stream_oracle(
    model: Any,
    adapter: OracleAdapter,
    activations: Mapping[int, torch.Tensor],
    *,
    max_bullets: int = ...,
    max_tokens: int = ...,
    marker: str | None = ...,
    partial: Literal[False] = ...,
) -> AsyncGenerator[OracleRead]: ...


@overload
def stream_oracle(
    model: Any,
    adapter: OracleAdapter,
    activations: Mapping[int, torch.Tensor],
    *,
    max_bullets: int = ...,
    max_tokens: int = ...,
    marker: str | None = ...,
    partial: bool,
) -> AsyncGenerator[OracleRead | OraclePartial]: ...


async def stream_oracle(
    model: Any,
    adapter: OracleAdapter,
    activations: Mapping[int, torch.Tensor],
    *,
    max_bullets: int = 2,
    max_tokens: int = 128,
    marker: str | None = None,
    partial: bool = False,
) -> AsyncGenerator[OracleRead | OraclePartial]:
    """Read each ``activations[layer]`` ([d_model]) with the adapter; yield each layer as it ends.

    The activations must be the base model's (captured without the adapter) at ``resid_post``,
    the output of the decoder layer. Greedy; on vLLM, bf16 noise can flip a near tie in a repeat.
    With ``partial``, vLLM also yields an :class:`OraclePartial` per token. Eager ends
    its layers together, so it yields none.
    """
    if max_bullets < 1:
        raise ValueError("max_bullets must be at least 1.")
    prompts = oracle_prompts(model.tokenizer, adapter.contract, list(activations), marker)
    from interp_engine.model import EagerModel
    from interp_engine.vllm_backend import VLLMModel

    reads: AsyncGenerator[OracleRead | OraclePartial]
    if isinstance(model, VLLMModel):
        reads = _vllm_reads(model, adapter, prompts, activations, max_bullets, max_tokens, partial)
    elif isinstance(model, EagerModel):
        reads = _eager_reads(model, adapter, prompts, activations, max_bullets, max_tokens)
    else:
        raise refuse(model, "A LoRA read", capability="lora_read")
    async with aclosing(reads) as it:
        async for read in it:
            yield read


async def read_oracle(
    model: Any,
    adapter: OracleAdapter,
    activations: Mapping[int, torch.Tensor],
    *,
    max_bullets: int = 2,
    max_tokens: int = 128,
    marker: str | None = None,
) -> dict[int, OracleRead]:
    """:func:`stream_oracle`, collected, in the order of ``activations``."""
    got = {
        r.layer: r
        async for r in stream_oracle(
            model, adapter, activations, max_bullets=max_bullets, max_tokens=max_tokens, marker=marker
        )
    }
    return {layer: got[layer] for layer in activations}


# --- vLLM ----------------------------------------------------------------------------------------


async def _vllm_reads(
    model: Any,
    adapter: OracleAdapter,
    prompts: dict[int, OraclePrompt],
    activations: Mapping[int, torch.Tensor],
    max_bullets: int,
    max_tokens: int,
    partial: bool,
) -> AsyncGenerator[OracleRead | OraclePartial]:
    if model.max_lora_rank is None or model.max_lora_rank < adapter.rank:
        raise ValueError(
            f"The adapter has rank {adapter.rank}; this engine was built with "
            f"max_lora_rank={model.max_lora_rank}. Load it with max_lora_rank>={adapter.rank} and "
            "enable_prompt_embeds=True."
        )
    lora_path = str(adapter.vllm_dir())
    eos = set(_eos_ids(model.tokenizer))
    # Each layer's partials, then its read (or the error that ended it).
    out: asyncio.Queue[OracleRead | OraclePartial | BaseException] = asyncio.Queue()

    async def read(p: OraclePrompt) -> OracleRead:
        rows = inject(await model.embed_rows(p.token_ids), p.slot, activations[p.layer], adapter.contract.alpha)
        ids: list[int] = []
        text = ""
        steps = model.generate_steps_from_embeds(
            rows, max_tokens=max_tokens, temperature=0.0, presence_penalty=0.0, lora_path=lora_path
        )
        async with aclosing(steps) as it:
            async for step in it:
                if step.token_id in eos:
                    return _read(p.layer, ids, text, max_bullets, True)
                ids.append(step.token_id)
                text += step.token_str
                if bullets_done(text, max_bullets):
                    break
                if partial and step.token_str:
                    out.put_nowait(OraclePartial(p.layer, text))
        return _read(p.layer, ids, text, max_bullets, False)

    async def one(p: OraclePrompt) -> None:
        try:
            out.put_nowait(await read(p))
        except Exception as exc:  # noqa: BLE001 - raised again by the loop below
            out.put_nowait(exc)

    tasks = [asyncio.ensure_future(one(p)) for p in prompts.values()]
    try:
        left = len(tasks)
        while left:
            item = await out.get()
            if isinstance(item, BaseException):
                raise item
            if isinstance(item, OracleRead):
                left -= 1
            yield item
    finally:
        for t in tasks:
            t.cancel()


# --- eager ---------------------------------------------------------------------------------------


@contextmanager
def eager_lora(model: Any, adapter: OracleAdapter) -> Iterator[None]:
    """The adapter on ``model``'s decoder layers as forward hooks, for the ``with`` body only."""
    layers = model.arch.decoder_layers
    handles = []
    try:
        for (layer, path), (a, b) in adapter.layer_pairs().items():
            module = layers[layer].get_submodule(path)
            weight = module.weight
            a = a.to(weight.device, weight.dtype)
            b = (b * adapter.scale).to(weight.device, weight.dtype)

            def hook(_m: Any, args: tuple, out: torch.Tensor, a: torch.Tensor = a, b: torch.Tensor = b) -> torch.Tensor:
                return out + (args[0] @ a.T) @ b.T

            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for h in handles:
            h.remove()


def _eos_ids(tokenizer: Any) -> list[int]:
    ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()
    for name in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(name)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            ids.add(tid)
    return sorted(ids)


async def _eager_reads(
    model: Any,
    adapter: OracleAdapter,
    prompts: dict[int, OraclePrompt],
    activations: Mapping[int, torch.Tensor],
    max_bullets: int,
    max_tokens: int,
) -> AsyncGenerator[OracleRead]:
    """One batched greedy ``generate`` per prompt length (Qwen's prompts all share one)."""
    from transformers import StoppingCriteria, StoppingCriteriaList

    tok = model.tokenizer
    eos = _eos_ids(tok)
    by_len: dict[int, list[OraclePrompt]] = {}
    for p in prompts.values():
        by_len.setdefault(len(p.token_ids), []).append(p)

    class _Cap(StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
            texts = tok.batch_decode(input_ids, skip_special_tokens=True)
            return torch.tensor([bullets_done(t, max_bullets) for t in texts], device=input_ids.device)

    for group in by_len.values():
        embed = model.arch.embed
        device = embed.weight.device
        with torch.no_grad():
            rows = torch.stack(
                [
                    inject(
                        embed(torch.tensor(p.token_ids, device=device)),
                        p.slot,
                        activations[p.layer],
                        adapter.contract.alpha,
                    )
                    for p in group
                ]
            )
            with eager_lora(model, adapter):
                out = model.hf_model.generate(
                    inputs_embeds=rows,
                    attention_mask=torch.ones(rows.shape[:2], dtype=torch.long, device=device),
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    eos_token_id=eos,
                    pad_token_id=eos[0],
                    stopping_criteria=StoppingCriteriaList([_Cap()]),
                )
        for p, row in zip(group, out.tolist(), strict=True):
            ids: list[int] = []
            ended = False
            for t in row:
                if t in eos:
                    ended = True
                    break
                ids.append(t)
            text = tok.decode(ids, skip_special_tokens=True)
            if bullets_done(text, max_bullets):
                text = _through_bullets(text, max_bullets)
            yield _read(p.layer, ids, text, max_bullets, ended)


def _through_bullets(text: str, max_bullets: int) -> str:
    """``text`` cut after the line that finishes bullet ``max_bullets`` (a batch runs on past it)."""
    seen = 0
    kept = []
    for line in text.split("\n"):
        kept.append(line)
        if _BULLET.match(line):
            seen += 1
            if seen == max_bullets:
                break
    return "\n".join(kept) + "\n"
