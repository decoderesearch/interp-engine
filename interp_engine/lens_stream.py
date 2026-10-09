"""A lens read-out over a generation: every position's top tokens at each lens's layers.

Behind ``generate_with_lens`` on every backend. A source hands over residual rows as the forwards
produce them -- the eager KV loop here, or any backend's ``capture_generation_stream`` -- and
:func:`read_out` stages each lens's rows (through ``J_bar`` where it has one), decodes them a chunk
at a time and yields one :class:`~interp_engine.api.LensStep` per position. vLLM's fused path reads
out in its worker, and :func:`fused_steps` only pairs what comes back with the token ids.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from interp_engine.address import Address
from interp_engine.api import LensSpec, LensStep
from interp_engine.lens_topk import lens_topk
from interp_engine.residual_basis import reduce_streams

if TYPE_CHECKING:
    from interp_engine.model import EagerModel
    from interp_engine.protocol import InterpModel
    from interp_engine.vllm_backend import VLLMModel

# Positions decoded per unembed matmul. Batching amortizes the lm_head read; the bound is the
# vocab-sized intermediate, chunk * n_layers * vocab (213 MB at gemma-2-2b's 26 layers x 256k).
READOUT_CHUNK = 8

# Positions carried through J_bar per batch. The transport is bound by re-reading each J_bar, and
# its intermediate is only batch * n_layers * d_model, so it takes many read-out chunks at once.
STAGE_BATCH = 128

DEFAULT_JACOBIAN_SET = "default"
"""The set ``LensSpec.jacobian_set`` and ``set_lens_jacobians`` use when not told one."""

JacobianSets = Mapping[str, Mapping[int, torch.Tensor]]
"""J_bar sets by name, each ``{layer: [d_model, d_model]}``."""

TopK = Callable[[torch.Tensor, int], Awaitable[tuple[torch.Tensor, torch.Tensor]]]
"""``(rows [n * n_layers, d_model], n_layers)`` to ``(top_idx, top_probs)``, each ``[n * n_layers, k]``."""


@dataclass
class Rows:
    """Positions ``first`` onward: their token ids, and per layer their ``[n, d_model]`` rows."""

    first: int
    token_ids: list[int]
    rows: dict[int, torch.Tensor]


def lens_max_tokens(max_tokens: int) -> int:
    """Tokens to sample so that each of ``max_tokens`` generated positions has a read-out.

    A position is read from the forward that takes its token as input, and no forward runs after the
    last sample, so one more is sampled than is read. With none to read, one is still sampled to run
    an intervened forward; its position is past the limit and is not read.
    """
    return max_tokens + 1 if max_tokens > 0 else 1


def stop_token_ids(model: Any) -> set[int]:
    """Every id that ends a generation: the tokenizer's EOS and the generation config's.

    gpt-oss ends a turn on ``<|return|>`` or ``<|call|>``, which only the generation config lists.
    """
    ids: set[int] = set()

    def add(value: Any) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, int):
            ids.add(value)
        elif isinstance(value, list | tuple | set):
            for item in value:
                add(item)

    add(getattr(model.tokenizer, "eos_token_id", None))
    add(getattr(getattr(getattr(model, "hf_model", None), "generation_config", None), "eos_token_id", None))
    return ids


def prepare(
    model: InterpModel,
    lenses: Sequence[LensSpec],
    *,
    top_n: int,
    point: str,
    stream_reduce: str,
    stream_index: int | None,
    jacobian_sets: Collection[str],
) -> list[int]:
    """The union of the lenses' layers, after refusing what no backend can read. No forward.

    ``jacobian_sets`` names the J_bar sets that hold at least one layer.
    """
    if not lenses:
        raise ValueError("generate_with_lens needs at least one LensSpec")
    if top_n <= 0:
        raise ValueError(f"top_n must be > 0, got {top_n}")
    n = model.n_layers
    for spec in lenses:
        layers = [int(layer) for layer in spec.layers]
        if not layers:
            raise ValueError("a LensSpec names no layers")
        if layers != sorted(set(layers)):
            raise ValueError(f"a LensSpec's layers must ascend with no repeats; got {layers}")
        if layers[0] < 0 or layers[-1] >= n:
            raise ValueError(f"layers {layers} are outside the model's {n} layers")
    missing = sorted({spec.jacobian_set for spec in lenses if spec.jacobian} - set(jacobian_sets))
    if missing:
        raise ValueError(
            f"A Jacobian lens needs its J_bar set {missing}: pass jacobians= (default set only), or "
            "call set_lens_jacobians first. Read without it, every layer is the logit lens under "
            "the Jacobian lens's name."
        )
    model.residual_basis.require_stream_reduction(stream_reduce, stream_index, point=point)
    return sorted({int(layer) for spec in lenses for layer in spec.layers})


def _reduce(model: InterpModel, stream_reduce: str, stream_index: int | None) -> Callable[[torch.Tensor], torch.Tensor]:
    n_streams = model.residual_basis.n_streams if stream_reduce != "none" else None
    return lambda t: reduce_streams(t, stream_reduce, index=stream_index, n_streams=n_streams)


def _stage(rows: Mapping[int, torch.Tensor], layers: Sequence[int], jacobians: Mapping[int, torch.Tensor] | None):
    """``[n * n_layers, d_model]`` float32, position-major, each layer through its J_bar if it has one.

    The matmul runs at the J_bar's dtype, on its device when that is an accelerator: the step is
    bound by re-reading J_bar, so the residual is the one cast and moved.
    """
    held = list((jacobians or {}).values())
    device = next((j.device for j in held if j.device.type != "cpu"), rows[layers[0]].device)
    blocks = []
    for layer in layers:
        block = rows[layer].to(device=device, dtype=torch.float32)
        j = jacobians.get(int(layer)) if jacobians else None
        if j is not None:
            j = j.to(device)
            block = (block.to(j.dtype) @ j.T).float()
        blocks.append(block)
    return torch.stack(blocks, dim=1).reshape(-1, blocks[0].shape[-1])


def decode_topk(model: InterpModel, *, top_n: int, word_mask: torch.Tensor | None) -> TopK:
    """Top-k through ``model.decode_residuals``, which applies the family's own post-unembed arithmetic."""
    placed: dict[torch.device, torch.Tensor] = {}

    async def topk(rows: torch.Tensor, n_layers: int) -> tuple[torch.Tensor, torch.Tensor]:
        logits = await model.decode_residuals(rows)
        mask = None
        if word_mask is not None:
            mask = placed.get(logits.device)
            if mask is None:
                mask = placed[logits.device] = word_mask.to(device=logits.device, dtype=torch.bool)
        idx, probs = lens_topk(logits, top_n=top_n, mask=mask, rows_per_group=n_layers)
        return idx.cpu(), probs.cpu()

    return topk


def worker_topk(
    model: VLLMModel, *, top_n: int, softcap: float | None, word_mask: torch.Tensor | None, dtype: torch.dtype | None
) -> TopK:
    """Top-k in the vLLM worker, for rows staged here. Sent at ``dtype``: the worker casts to it on arrival."""

    async def topk(rows: torch.Tensor, n_layers: int) -> tuple[torch.Tensor, torch.Tensor]:
        return await model.decode_residuals_topk(
            rows if dtype is None else rows.to(dtype),
            top_n=top_n,
            softcap=softcap,
            word_mask=word_mask,
            rows_per_group=n_layers,
        )

    return topk


async def read_out(
    source: AsyncIterator[Rows],
    lenses: Sequence[LensSpec],
    *,
    prompt_len: int,
    jacobians: JacobianSets,
    topk: TopK,
) -> AsyncIterator[LensStep]:
    """One LensStep per position of ``source``, each chunk's steps as soon as that chunk is decoded."""
    async for batch in source:
        n = len(batch.token_ids)
        for start in range(0, n, STAGE_BATCH):
            end = min(n, start + STAGE_BATCH)
            part = {layer: rows[start:end] for layer, rows in batch.rows.items()}
            staged = [
                _stage(part, spec.layers, jacobians.get(spec.jacobian_set) if spec.jacobian else None)
                for spec in lenses
            ]
            for lo in range(0, end - start, READOUT_CHUNK):
                hi = min(end - start, lo + READOUT_CHUNK)
                results = []
                for spec, rows in zip(lenses, staged, strict=True):
                    g = len(spec.layers)
                    idx, probs = await topk(rows[lo * g : hi * g], g)
                    k = int(idx.shape[-1])
                    results.append((idx.view(hi - lo, g, k), probs.view(hi - lo, g, k)))
                for i in range(hi - lo):
                    position = batch.first + start + lo + i
                    yield LensStep(
                        position=position,
                        token_id=int(batch.token_ids[start + lo + i]),
                        is_generated=position >= prompt_len,
                        top_ids=[r[0][i] for r in results],
                        top_probs=[r[1][i] for r in results],
                    )


def _sample(logits: torch.Tensor, temperature: float) -> int:
    """Greedy at ``temperature <= 0``, else temperature sampling over the full vocab."""
    # `torch.multinomial` on nan/inf is a device-side assert that poisons the CUDA context.
    if not torch.isfinite(logits).all():
        raise ValueError(
            "Non-finite logits during generation (nan/inf), likely from steering that is too strong. "
            "Reduce the steer strength or the number of steered layers."
        )
    if temperature <= 0:
        return int(logits.argmax())
    return int(torch.multinomial(torch.softmax(logits.float() / temperature, dim=-1), num_samples=1))


def _eager_rows(
    model: EagerModel,
    prompt_token_ids: Sequence[int],
    layers: Sequence[int],
    *,
    point: str,
    max_tokens: int,
    temperature: float,
    seed: int | None,
    skip_before: int,
    stream_reduce: str,
    stream_index: int | None,
    steering_spec: Any,
) -> Iterator[Rows]:
    """The KV-cached loop: the prefill's rows, then one row per generated token as it is read.

    Hooks go on and come off around each forward, so nothing is installed while the caller holds a
    step. Read hooks are registered after the steering hooks, so a read sees the steered tensor.
    """
    from interp_engine.hooks import HookManager
    from interp_engine.steer import forward_from

    ids = [int(t) for t in prompt_token_ids]
    prompt = torch.tensor([ids], device=model.device)
    reduce = _reduce(model, stream_reduce, stream_index)
    targets = [(layer, *model.resolve_point(point, layer)) for layer in layers]
    stops = stop_token_ids(model)
    captured: dict[int, torch.Tensor] = {}

    def reader(layer: int) -> Callable[[torch.Tensor], None]:
        def _read(tensor: torch.Tensor) -> None:
            captured[layer] = tensor.detach()

        return _read

    if seed is not None:
        torch.manual_seed(seed)
    cur, past, position, generated = prompt, None, 0, 0
    while True:
        captured.clear()
        with model._maybe_steer(steering_spec, prompt), HookManager() as hm, torch.no_grad():
            for layer, module, side in targets:
                hm.read(module, reader(layer), point=side)  # pyright: ignore[reportArgumentType]
            with forward_from(position):
                out = model.hf_model(cur, past_key_values=past, use_cache=True)
        width = int(cur.shape[1])
        lo = max(skip_before - position, 0)
        if lo < width:
            rows = {layer: reduce(captured[layer][0][lo:]) for layer in layers}
            yield Rows(position + lo, ids[position + lo : position + width], rows)
        position += width
        if generated >= max_tokens or (generated and ids[-1] in stops):
            return
        next_id = _sample(out.logits[0, -1, :], temperature)
        ids.append(next_id)
        generated += 1
        past = out.past_key_values
        cur = torch.tensor([[next_id]], device=model.device)


async def eager_rows(model: EagerModel, prompt_token_ids: Sequence[int], layers: Sequence[int], **kw: Any):
    """:func:`_eager_rows` as the async source :func:`read_out` takes."""
    for rows in _eager_rows(model, prompt_token_ids, layers, **kw):
        yield rows


async def protocol_rows(
    model: InterpModel,
    prompt_token_ids: Sequence[int],
    layers: Sequence[int],
    *,
    point: str,
    max_tokens: int,
    temperature: float,
    seed: int | None,
    skip_before: int,
    stream_reduce: str,
    stream_index: int | None,
    steering_spec: Any,
) -> AsyncIterator[Rows]:
    """Rows through ``capture_generation_stream``, each position as soon as its row and its id are in.

    As often as the backend yields: per token on vLLM. Nothing generated and nothing steered
    is one ``capture``.
    """
    from interp_engine.steer import active_steering

    ids = [int(t) for t in prompt_token_ids]
    prompt_len = len(ids)
    points = [Address(point, int(layer)) for layer in layers]
    reduce = _reduce(model, stream_reduce, stream_index)
    emitted = skip_before
    if max_tokens <= 0 and steering_spec is None and active_steering(model) is None:
        caps = await model.capture(ids, points)
        if emitted < prompt_len:
            got = {layer: reduce(caps[a])[emitted:] for layer, a in zip(layers, points, strict=True)}
            yield Rows(emitted, ids[emitted:], got)
        return

    rows: dict[int, list[torch.Tensor]] = {int(layer): [] for layer in layers}
    limit = prompt_len + max(max_tokens, 0)
    async for new, token_ids in model.capture_generation_stream(
        ids,
        points,
        max_tokens=lens_max_tokens(max_tokens),
        temperature=temperature,
        seed=seed,
        steering_spec=steering_spec,
    ):
        for layer, a in zip(layers, points, strict=True):
            block = new.get(a)
            if block is not None:
                rows[int(layer)].extend(reduce(block).unbind(0))
        known = ids + [int(t) for t in token_ids]
        available = min(min(len(r) for r in rows.values()), len(known), limit)
        if emitted < available:
            got = {layer: torch.stack(r[emitted:available]) for layer, r in rows.items()}
            yield Rows(emitted, known[emitted:available], got)
            emitted = available


async def client_steps(
    model: InterpModel,
    prompt_token_ids: Sequence[int],
    lenses: Sequence[LensSpec],
    *,
    jacobians: JacobianSets,
    point: str,
    top_n: int,
    max_tokens: int,
    temperature: float,
    seed: int | None,
    word_mask: torch.Tensor | None,
    skip_before: int,
    stream_reduce: str,
    stream_index: int | None,
    steering_spec: Any,
) -> AsyncIterator[LensStep]:
    """The read-out on the caller's side: rows from :func:`protocol_rows`, the top-k through
    ``decode_residuals``. For a backend that holds ``J_bar`` here and has no worker read-out.
    """
    layers = prepare(
        model,
        lenses,
        top_n=top_n,
        point=point,
        stream_reduce=stream_reduce,
        stream_index=stream_index,
        jacobian_sets=[name for name, held in jacobians.items() if held],
    )
    source = protocol_rows(
        model,
        prompt_token_ids,
        layers,
        point=point,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
        skip_before=min(max(int(skip_before), 0), len(prompt_token_ids)),
        stream_reduce=stream_reduce,
        stream_index=stream_index,
        steering_spec=steering_spec,
    )
    topk = decode_topk(model, top_n=top_n, word_mask=word_mask)
    async for step in read_out(source, lenses, prompt_len=len(prompt_token_ids), jacobians=jacobians, topk=topk):
        yield step


async def fused_steps(
    model: VLLMModel,
    prompt_token_ids: Sequence[int],
    lenses: Sequence[LensSpec],
    layers: Sequence[int],
    *,
    point: str,
    top_n: int,
    max_tokens: int,
    temperature: float,
    seed: int | None,
    word_mask: torch.Tensor | None,
    skip_before: int,
    stream_reduce: str,
    stream_index: int | None,
    softcap: float | None,
    steering_spec: Any,
) -> AsyncIterator[LensStep]:
    """vLLM's worker-side read-out, paired with the token ids. Generation runs ahead of the read-outs,
    so a position is emitted once both have arrived, and never past the request's limit."""
    ids = [int(t) for t in prompt_token_ids]
    prompt_len = len(ids)
    limit = prompt_len + max(max_tokens, 0)
    specs = [
        {"layers": [int(x) for x in spec.layers], "jacobian": bool(spec.jacobian), "jacobian_set": spec.jacobian_set}
        for spec in lenses
    ]
    slices: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    emitted = skip_before
    stream = model.lens_capture_readout_stream(
        ids,
        [Address(point, int(layer)) for layer in layers],
        specs,
        top_n=top_n,
        softcap=softcap,
        word_mask=word_mask,
        chunk_positions=READOUT_CHUNK,
        skip_before=skip_before,
        max_tokens=lens_max_tokens(max_tokens),
        temperature=temperature,
        seed=seed,
        steering_spec=steering_spec,
        stream_reduce=stream_reduce,
        stream_index=stream_index,
    )
    async for first, idx_list, probs_list, gen_ids in stream:
        n = int(idx_list[0].shape[0]) // len(lenses[0].layers) if idx_list else 0
        for offset in range(n):
            per = []
            for i, spec in enumerate(lenses):
                g, k = len(spec.layers), int(idx_list[i].shape[-1])
                per.append((idx_list[i].view(n, g, k)[offset], probs_list[i].view(n, g, k)[offset]))
            slices[first + offset] = per
        while emitted < limit and emitted in slices and emitted < prompt_len + len(gen_ids):
            per = slices.pop(emitted)
            token_id = ids[emitted] if emitted < prompt_len else int(gen_ids[emitted - prompt_len])
            yield LensStep(
                position=emitted,
                token_id=token_id,
                is_generated=emitted >= prompt_len,
                top_ids=[p[0] for p in per],
                top_probs=[p[1] for p in per],
            )
            emitted += 1


def jacobian_bytes(jacobians: Mapping[int, torch.Tensor] | None) -> int:
    return sum(j.numel() * j.element_size() for j in (jacobians or {}).values())


def install_jacobian_set(
    held: dict[str, dict[int, torch.Tensor]], jacobians: Mapping[int, torch.Tensor] | None, name: str
) -> int:
    """Put ``jacobians`` in ``held`` as set ``name``, or with ``None`` drop that set. Returns its bytes."""
    if jacobians is None:
        held.pop(name, None)
        return 0
    held[name] = {int(k): v for k, v in jacobians.items()}
    return jacobian_bytes(held[name])


def call_jacobian_sets(
    held: Mapping[str, Mapping[int, torch.Tensor]], jacobians: Mapping[int, torch.Tensor] | None
) -> dict[str, Mapping[int, torch.Tensor]]:
    """The sets a call reads: ``held``, with the call's ``jacobians`` as the default set."""
    sets = dict(held)
    if jacobians is not None:
        sets[DEFAULT_JACOBIAN_SET] = jacobians
    return sets
