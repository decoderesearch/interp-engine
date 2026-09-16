"""Capture via interp-engine (raw HF, eager) — the reference engine."""

from __future__ import annotations

import numpy as np

from comparison.dumpio import with_mask_sentinel
from comparison.spec import SaeSpec, dump_key, layers_for_point


def _resolvable(model, point: str, layer: int | None, announced: set[str]) -> bool:
    """Whether this checkpoint has ``point`` at all, asked before the forward rather than after.

    Not every compared point exists on every architecture: `resid_mid` is undefined on a parallel
    block, where interp-engine raises instead of returning the plausible-looking `resid_pre`. One
    unresolvable point would otherwise take the whole capture down with it and cost the reference
    column three cells it can produce — and since `eager` is the reference, an empty cell here reads
    as "no reference" for every engine, not just this one.

    Each distinct refusal is announced once, not once per layer: these are usually facts about the
    architecture, so the per-layer print is the same sentence N times, and in a sweep log that buries
    the lines that are about this capture. Keyed on the message rather than on the point, so a trunk
    that refuses a point for a *layer-specific* reason (a mamba mixer at layer 0, softmax attention
    at layer 5) still says so each time the reason differs.
    """
    from interp_engine import points

    spec = points.point_spec(point)
    # The mHC rows are excluded from the branch below even though the registry marks most of them
    # `module_resolved=False`: that flag is about vLLM, where the deferred post phase leaves those
    # tensors as locals of a kernel call, while on the HF module tree they are ordinary module I/O that
    # `resolve_point` answers for (`model._resolve_hyper_connection`). Stated rather than left to the
    # default above -- `point_spec` is asked about the global table, which returns None for a
    # conditional row -- so that a stream point is never sent down the attention-score path and refused
    # for a reason that is not about it.
    if spec is not None and not spec.module_resolved and point not in points.hyper_connection_names():
        # A point no module boundary carries, so `resolve_point` refuses it by design rather than
        # because this checkpoint lacks it -- `attn_scores` is rebuilt inside `run_with_cache`,
        # which owns that path. Asking `resolve_point` here would read a declaration about the
        # point as a fact about the model and drop a point most checkpoints can produce.
        #
        # It is asked a different way instead, because there are several ways a family can refuse
        # it and getting the *reference* engine wrong costs every cell in the row rather than one:
        # `bloom` has no `eager_attention_forward` to delegate to, a gpt2 with
        # `reorder_and_upcast_attn` picks its attention path by name, a hybrid trunk's linear layers
        # compute no scores at all, and a model not loaded eagerly never forms them. Entering the
        # capture's own context manager runs exactly those checks -- all of them happen before it
        # yields, and its `finally` restores the dispatch either way -- so this is the engine's
        # answer rather than a second copy of it here, and it costs no forward.
        if layer is None:
            return True
        from interp_engine.attn_scores import capture_attn_scores

        try:
            with capture_attn_scores(model, [layer]):
                pass
        except (ValueError, AttributeError, RuntimeError) as exc:
            message = f"[eager/{model.arch.architecture}] point '{point}' unavailable: {exc}"
            if message not in announced:
                announced.add(message)
                print(message)
            return False
        return True
    try:
        model.resolve_point(point, layer)
    except (ValueError, AttributeError) as exc:
        message = f"[eager/{model.arch.architecture}] point '{point}' unavailable: {exc}"
        if message not in announced:
            announced.add(message)
            print(message)
        return False
    return True


def _host_in_use_bytes() -> int:
    """Host memory this cgroup holds that a load cannot reclaim: anonymous pages and shm.

    Not ``memory.current``, which also counts the page cache -- reclaimable, and after a checkpoint
    copy it can be hundreds of GB that would price the host as nearly full. 0 when unreadable.
    """
    try:
        with open("/sys/fs/cgroup/memory.stat") as f:
            stat = dict(line.split() for line in f if " " in line)
        return int(stat["anon"]) + int(stat["shmem"])
    except (OSError, KeyError, ValueError):
        pass
    for path in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            with open(path) as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            continue
    return 0


def offload_budget(hf_id: str, dtype: str, num_gpus: int) -> dict[int | str, int] | None:
    """Per-device byte budgets that spill the reference onto host RAM when the cards cannot hold it.

    transformers dequantizes every routed expert of a compressed-tensors checkpoint at load, so a
    native-INT4 MoE such as Kimi-K2.6 is 2.1 TB of bf16 in eager -- more than 8x H200 hold, while
    the same box has 2 TB of host RAM. accelerate's ``device_map="auto"`` already spills to the CPU
    when the cards fill, but it packs the first card to its limit, and the first card is where every
    offloaded layer executes: it needs room for one full layer's weights on top of what lives there.
    So card 0 keeps 55% for placement and the others 85%, the fraction the validator budgets weights
    at everywhere else -- and not more, because the load dequantizes each expert group on the card it
    lands on and needs a projection-sized scratch there (10.5 GiB on Kimi), which a card packed to
    92% ran out of. Host RAM takes the rest. The capture is the same bf16 arithmetic on the same
    cards, only slower; nothing about the numbers changes.

    None when the weights fit the cards at 85%, so the ordinary path is untouched.
    """
    from interp_engine.model import resolve_trust_remote_code
    from transformers import AutoConfig

    from comparison import sizing

    try:
        cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=resolve_trust_remote_code(hf_id, None))
        need = sizing.weight_bytes(cfg, dtype)
    except Exception:  # noqa: BLE001 - an unreadable config falls back to accelerate's own placement
        return None
    cards = sizing.gpu_memory_bytes(num_gpus)
    if not need or not cards or need <= int(cards * 0.85):
        return None
    per_card = cards // num_gpus
    budget: dict[int | str, int] = {0: int(per_card * 0.55)}
    budget.update({i: int(per_card * 0.85) for i in range(1, num_gpus)})
    host = sizing.host_memory_bytes() - _host_in_use_bytes()
    budget["cpu"] = max(0, int(host * 0.9))
    print(
        f"[eager/offload] {hf_id}: {sizing.gib(need)} of {dtype} weights exceed {sizing.gib(cards)} on "
        f"{num_gpus} cards -> {sizing.gib(sum(v for k, v in budget.items() if k != 'cpu'))} placed on the "
        f"cards, up to {sizing.gib(budget['cpu'])} offloaded to host RAM"
    )
    if need > sum(budget.values()):
        print(f"[eager/offload] {hf_id}: the host cannot hold the remainder either; the load will fail")
    return budget


def capture(
    hf_id: str,
    input_ids: list[int],
    layers: list[int],
    points: list[str],
    saes: tuple[SaeSpec, ...] = (),
    device: str = "cpu",
    dtype: str = "float32",
    num_gpus: int = 1,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    import torch
    from interp_engine import EagerModel, deepgemm_fallback_kwargs, run_with_cache

    # Empty for every checkpoint but an FP8 one on a GPU the `deep-gemm` Hub build does not target,
    # where it is the difference between measuring the row and losing it: the library refuses that
    # combination rather than choosing an experts implementation on a caller's behalf, and `eager` is
    # the reference, so its skip would take every other engine's cell in the row down with it.
    #
    # More than one GPU: accelerate places the layers (`device_map="auto"`), the same route
    # `load_model(num_gpus=N)` takes, and `device` must then stay unset or the load would pull the
    # sharded model back onto one card. Hooks fire on whichever card each module lives on. When even
    # all the cards cannot hold the weights, `offload_budget` lets accelerate spill layers to host RAM.
    model_kwargs = deepgemm_fallback_kwargs(hf_id)
    budget = offload_budget(hf_id, dtype, num_gpus) if num_gpus > 1 else None
    if budget is not None:
        model_kwargs = {**model_kwargs, "max_memory": budget}
    model = EagerModel(
        hf_id,
        dtype=dtype,
        device=None if num_gpus > 1 else device,
        device_map="auto" if num_gpus > 1 else None,
        attn_implementation="eager",
        model_kwargs=model_kwargs,
    )
    ids = torch.tensor([input_ids], device=model.device)

    announced: set[str] = set()
    requests = [
        (point, layer)
        for point in points
        for layer in layers_for_point(point, layers)
        if _resolvable(model, point, layer, announced)
    ]
    # SAE points may differ from the compared points (e.g. resid_pre); capture them too. The layer can
    # differ as well, so this can land a *compared* point outside the layer plan -- which is why the
    # aggregator scores only the planned layers (`aggregate._planned_layers`) rather than every key here.
    for s in saes:
        requests.append((s.point, s.layer))

    cache = run_with_cache(model, ids, requests)

    arrays: dict[str, np.ndarray] = {}
    for point, layer in requests:
        if (point, layer) in cache:
            t = cache.get(point, layer)[0]  # drop batch dim -> [seq, ...]
            array = t.float().cpu().numpy()
            # Scores are the one point whose mask fill is the checkpoint's to spell, and not every
            # checkpoint spells it the way HF's eager attention does: DeepSeek-V4-Flash's own modeling
            # code masks the compressed blocks its `compressed_sparse_attention` layers attend over
            # with -inf, so those layers arrive with a legitimate mask that the non-finite guard in
            # `run_engine` would read as a corrupt reference and refuse the whole capture over.
            arrays[dump_key(point, layer)] = with_mask_sentinel(array) if point == "attn_scores" else array

    sae_summaries: list[dict] = []
    if saes:
        from comparison.sae_check import encode_summary

        for s in saes:
            act = cache.get(s.point, s.layer)[0].float().cpu().numpy()
            summary = encode_summary(act, s.release, s.sae_id, device="cpu", loader=s.loader)
            if summary is not None:
                sae_summaries.append(summary)
    return arrays, sae_summaries
