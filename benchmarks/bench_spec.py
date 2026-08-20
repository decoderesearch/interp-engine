"""What gets benchmarked: models, backend variants, workloads, and the prompt normalization.

Everything here is plain data plus a tokenizer call, so it imports no torch and no vLLM and can be
listed (``python -m benchmarks.run_bench --list``) on a machine with no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------- models --


@dataclass(frozen=True)
class ModelSpec:
    """One checkpoint to benchmark."""

    key: str
    """Short name used in filenames, CLI filters and report rows."""
    hf_id: str
    family: str
    """Architecture family, so the report can show that the set spans more than one."""
    params: str
    """Parameter count as advertised, for the report's size column."""
    dtype: str = "bfloat16"
    """Passed to ``load_model``. Pinned rather than left at ``"auto"``, which is the checkpoint's own
    precision and does not resolve to the same thing on both backends: eager honors a float32
    checkpoint while vLLM's "auto" downcasts it to bfloat16, which would make that model's row a
    precision comparison rather than a backend one."""
    min_gpu_gib: float = 0.0
    """The smallest card this row fits on, in GiB of total VRAM. A sweep that was not given an
    explicit ``--models`` drops the rows the card cannot hold and says which (:func:`default_models`).

    A floor rather than a hardware allowlist, because what makes a row impossible here is one number:
    the weights have to fit beside a KV pool. It is deliberately not consulted when a model is asked
    for by name -- someone who types ``--models deepseek-v4-flash-0731`` on a 96 GiB card wants the OOM and
    the reason, not a sweep that quietly ran nothing."""
    capture_point: str = "resid_post"
    """The point the capture, capture-generation and steering workloads address on this model.

    Overridden only where ``resid_post`` does not name anything on the architecture, which is the
    case for a hyper-connection trunk: it carries several parallel residual streams, so the engine
    refuses the ambiguous name rather than picking one. A substitute has to be the same *shape* of
    work for the row to belong in the table -- one ``d_model``-wide vector per position per layer,
    written by a hook in the same place in the block -- or the capture columns stop being comparable
    across models."""
    per_variant_vllm_kwargs: dict[str, dict[str, object]] = field(default_factory=dict)
    """vLLM arguments one *variant* cannot serve this checkpoint without, keyed by variant key and
    merged over :attr:`extra_vllm_kwargs`. Empty for every model whose configurations all fit the
    sweep's shared budget, which is all but the largest.

    Separate from :attr:`extra_vllm_kwargs` because that is a fact about the weights and applies to
    every vLLM column, so putting a static-only budget there would change three cells that have
    already been measured and are comparable as they stand. The overrides here are recorded in the cell
    and stated by the report, so the one cell that ran on a different budget says so."""
    static_capture_point: str | None = None
    """The point those workloads address instead when the variant under test declared a set, for a
    checkpoint where the two cannot be the same point. ``None`` means :attr:`capture_point` serves both.

    A static engine serves *the set it declared* and refuses anything outside it, so the workload has to
    name a point the static set covers. On a hyper-connection trunk it cannot be the same one:
    ``static_points="auto"`` resolves to ``resid_streams`` -- the whole stack -- while the other columns
    need one ``d_model`` row to stay comparable with the rest of the table. Declared here rather than
    inferred at run time so the cell records it, ``report_bench`` states it under *Where a row differs*,
    and ``cells.nonuniform`` carries it onto the visualizer's card as one line of that row's footnote.

    Only the static cell records it, so read it through ``cells.row_spec`` rather than off a cell.
    """
    extra_eager_kwargs: dict[str, object] = field(default_factory=dict)
    """``load_model`` arguments the eager variants need for this checkpoint, merged under the
    variant's own. The counterpart to :attr:`extra_vllm_kwargs`, and used for the same reason: a
    property of the weights rather than of the configuration under test."""
    extra_vllm_kwargs: dict[str, object] = field(default_factory=dict)
    """vLLM engine arguments this *checkpoint* cannot be served without, merged under the variant's
    own. Not a tuning surface: a knob that only makes a model faster belongs in a variant, where it is
    measured, and everything here is something vLLM refuses to boot without.

    Kept per model rather than passed by whoever runs the sweep because the failure is late and reads
    like a bug in this harness -- ``DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache, got auto``
    arrives from an assertion inside the model class, after the config work but before any weight is
    read."""
    gpu_memory_utilization: float | None = None
    """vLLM's fraction of the card for this row alone, or None for the uniform
    :data:`GPU_MEMORY_UTILIZATION`.

    An exception that has to be declared rather than left to the caller now that the sweep runs this
    row by default: 148-155 GiB of weights do not fit in what 0.8 of a 180 GB card reserves, so the
    cell would OOM during bring-up. The cost is that this row's memory reservation is not the one
    every other row ran under, which is exactly what the uniform default exists to protect -- so it
    stays a per-model override with a reason attached, and the report says the row was measured under
    a different reservation. Overridden in turn by ``--gpu-memory-utilization`` on either script."""


#: Families roughly doubling in size, from the deployed set
#: (``apps/inference/local_scripts/pods.yaml``) rather than invented, so the numbers describe models
#: that are actually served. All five are the default sweep; the first three fit on one 32 GiB card
#: next to a vLLM reservation and the last two need progressively more card, which ``min_gpu_gib``
#: states per row rather than leaving to whoever runs it.
#:
#: The two large rows each break an assumption the small ones let stand, which is the point of having
#: them but also means neither is a drop-in:
#:
#: - ``qwen3.8-27b`` is a *multimodal wrapper* (``Qwen3_5ForConditionalGeneration``, a vision tower
#:   over a nested text config) and a *hybrid*: its 64 layers are three ``linear_attention`` to one
#:   ``full_attention``, so only 16 of them have softmax attention for the attention workloads to read,
#:   and the linear ones want ``causal-conv1d``/``flash-linear-attention`` or transformers falls back to
#:   a slow torch path that would be measured as the model being slow.
#: - ``deepseek-v4-flash-0731`` is 155 GiB of weights, more than the uniform ``GPU_MEMORY_UTILIZATION`` below
#:   reserves on a 180 GB B200, so it carries its own fraction and its own floor. Its memory figures
#:   are therefore not comparable with the rows swept at 0.8.
MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("gemma-2-2b", "google/gemma-2-2b", "Gemma2", "2.6B"),
    ModelSpec("qwen3-4b", "Qwen/Qwen3-4B", "Qwen3", "4.0B"),
    ModelSpec("llama-3.1-8b", "meta-llama/Llama-3.1-8B", "Llama3", "8.0B"),
    # 52 GiB of weights, so the 80 GiB class is the first that holds it beside a KV pool.
    ModelSpec("qwen3.8-27b", "Qwen/Qwen3.8-27B", "Qwen3_5", "27B", min_gpu_gib=80.0),
    # bfloat16 here is the *compute* dtype, not a request to expand the weights: the checkpoint is
    # block-quantized FP8 with UE8M0 scales and transformers keeps it that way (it dequantizes only
    # when asked with `dequantize=True`). Passing "auto" instead would leave the two backends free to
    # disagree about that, which is the thing this field exists to prevent.
    ModelSpec(
        "deepseek-v4-flash-0731",
        "deepseek-ai/DeepSeek-V4-Flash-0731",
        "DeepseekV4",
        "291B (14B active)",
        min_gpu_gib=170.0,
        # `resid_post` does not exist on this trunk: the block carries `hc_mult` (4) parallel residual
        # streams, and the engine refuses the name rather than silently returning stream 0. `mlp_out`
        # is the closest thing that keeps every column meaning what it means elsewhere -- one d_model
        # row per position per layer, so the transport figures compare, and a plain module output, so
        # the steering column has something to write to.
        #
        # The two obvious alternatives both cost a column. `mlp_stream_collapse` (the vector the MLP
        # actually reads) captures fine on both backends but resolves to element 2 of the hyper-
        # connection's output tuple, and steering asserts on a plain input/output side -- so the steer
        # cell would be an error on eager while working on vLLM, which is a difference in what the two
        # cells measured rather than in how fast they are. `resid_streams` is steerable but 4x the
        # bytes, which would price this row's transport as though DeepSeek moved four times the
        # activations to answer the same question.
        capture_point="mlp_out",
        # ...except under static, which serves the set it declared: `"auto"` on this trunk declares
        # `resid_streams`, so `mlp_out` is outside the set and would be refused. That column therefore
        # prices the stack rather than a row, which is declared as a row exception instead of hidden.
        static_capture_point="resid_streams",
        # The static column is the one configuration here that does not fit the shared budget, and both
        # of these are why. 149 GiB of weights at 0.95 of a 178 GiB card leave about 15 GiB for the
        # activation peak, the static buffers, the KV pool and the graph pool together.
        #
        # A static buffer is allocated per batched row, so at the 4096 `max_num_batched_tokens` static
        # lowered vLLM's default to, `resid_streams` costs 4096 rows x 4 streams x 4096 wide x 2 bytes
        # x 43 layers = 5.8 GiB -- and vLLM then sized the KV pool from what was left and refused to
        # start: "No available memory for the cache blocks". 1024 rows costs a quarter of that and is
        # still twice the longest prompt in the sweep (512 tokens, `MAX_MODEL_LEN` 2048).
        #
        # The capture sizes are the same argument for the graph pool: vLLM captures a decode graph at
        # each of ~19 batch sizes by default, and this sweep runs at concurrency 1 and 8. Capturing the
        # sizes it will actually replay keeps the rest of that memory, and the startup time, unspent.
        # Neither changes what the workloads measure: both sizes they use are still captured.
        per_variant_vllm_kwargs={
            "vllm-static": {
                "max_num_batched_tokens": 1024,
                "compilation_config": {"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]},
            }
        },
        # Placed on the card as each shard is read, instead of the eager default of "load to host RAM,
        # then .to(device)". At 149 GiB that default is not a slower road to the same place: it stages
        # the whole checkpoint in host memory and copies it across, which took over an hour here
        # against roughly ten minutes for the same bytes placed directly, and transformers' own FP8
        # quantizer warns for it ("You have loaded an FP8 model on CPU ... pass device_map = 'cuda'").
        # `load_model` sets `device=None` whenever a device_map is given, so the two do not fight.
        #
        # It does mean this row's `construct_s` measures a different load path from the small eager
        # rows, which is recorded in the cell's kwargs. The alternative was to measure a load nobody
        # doing this for real would perform.
        extra_eager_kwargs={"device_map": "cuda"},
        # No `kv_cache_dtype="fp8"` here any more, though every cell in `results/` recorded one: vLLM's
        # V4 attention does assert on anything else -- its compressed KV lives in DeepSeek's own
        # `fp8_ds_mla` layout, which has no 16-bit form -- but that is a fact about the architecture, so
        # the engine derives it (`facts.mandatory_kv_cache_dtype`) instead of each harness remembering.
        # This spec was one of the three copies that kept the gap invisible from inside the repo.
        gpu_memory_utilization=0.95,
    ),
)

# ------------------------------------------------------------------------- variants --


@dataclass(frozen=True)
class VariantSpec:
    """One backend configuration. The unit of process isolation.

    Each variant runs in its own process because vLLM reserves ``gpu_memory_utilization`` of the
    whole card up front and does not give it back on shutdown reliably enough to trust two engines
    in one interpreter.
    """

    key: str
    backend: str
    """``"eager"`` or ``"vllm"``, passed straight to :func:`interp_engine.load_model`."""
    kwargs: dict[str, object] = field(default_factory=dict)
    """Extra ``load_model`` kwargs. Forwarded verbatim to the backend constructor."""
    note: str = ""
    label: str = ""
    """How the report names this column. Separate from :attr:`key`, which stays short because it is
    what ``--variant`` takes and what the result filenames are made of; the label says whose
    configuration it is, which is the thing a reader of the tables needs."""
    models: tuple[str, ...] = ()
    """Model keys this variant applies to; empty means all of them, which is the normal case.

    A restriction is for a configuration that does not *exist* off one checkpoint -- a draft head
    shipped inside the target weights, say. Without it the sweep's cross product attempts the variant
    on every row and fails there, and a failed cell reads as a broken column rather than an
    inapplicable one. It is deliberately not a way to skip a pair that is merely slow or awkward:
    those belong in the table, which is where a reader finds out what a configuration costs."""


#: One of these three exists to price a default that the engine chose for capture's sake, which is the
#: most useful thing a speed benchmark of this library can say. Hooked vLLM capture rules CUDA graphs
#: out, because graph replay does not re-execute the Python forward and so never fires a
#: ``register_forward_hook``. That is what separates the three vLLM backends, and it is why
#: ``vllm-cudagraph`` -- ``backend="vllm-generate"``, vLLM left at its own defaults, hence the
#: "vanilla" label -- is here at all: it measures what ours costs. The capture workloads run there too
#: rather than being skipped, so the report can show *what* a capture returns under replay instead of
#: asserting that it fails.
#:
#: Each variant names its backend rather than deriving it from the kwargs it passes, which is what
#: the engine now expects: ``vllm-static`` and ``vllm-generate`` refuse ``enforce_eager``, and a tap
#: set is only accepted by the backend built to bake one in.
VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec(
        "eager",
        "eager",
        {},
        "raw HF forward; attn_implementation=eager, the engine's default",
        "interp-engine eager",
    ),
    VariantSpec(
        "vllm",
        "vllm",
        {},
        "capture-capable: CUDA graphs and compile OFF",
        "interp-engine vllm",
    ),
    VariantSpec(
        "vllm-cudagraph",
        "vllm-generate",
        {},
        "CUDA graphs + inductor compile ON, no static wraps; vLLM's own defaults for generate-only",
        "vllm (vanilla)",
    ),
    # Breakable CUDA graphs with resid_post static wraps at every layer. Production static
    # (``static_points="auto"``) is this path, not Dynamo piecewise.
    #
    # `qwen3.8-27b` joined the row once static was correct on a hybrid trunk. Breakable graphs turn
    # inductor off, and vLLM's `FULL_AND_PIECEWISE` capture then miscomputes prefill for a gated-delta
    # trunk -- the engine generated fluent nonsense rather than failing, so the cell would have been a
    # plausible number for a broken forward. Static now pins `FULL_DECODE_ONLY` on such a trunk, which
    # runs prefill eagerly and keeps the decode graphs, and the validator's static column agrees with
    # eager on all 28 points. Its prefill figures carry that eager prefill, which is the point of
    # comparing it against the same model's other variants rather than against another model.
    #
    # `deepseek-v4-flash-0731` is in, and is the one row here whose static set is not one point per
    # layer. It is a hyper-connection trunk, so `"auto"` resolves to `resid_streams` -- the whole stack
    # of four parallel residual streams per layer, four times the width of a `resid_post` row. Its
    # capture and transport figures therefore price four times the activations for the same question,
    # which `cells.nonuniform` declares so the report states it and the visualizer's card drops the row
    # rather than publishing it beside three that declared a quarter as much.
    #
    # Its write tap is `mlp_out`, not `resid_post`: the steer workload addresses the model's
    # `capture_point`, and a hyper-connection trunk refuses the default name (`run_bench._steer_site`).
    #
    # `static_writes` was once what made the `steer` cell a number instead of `n/a`: `"auto"` installed
    # reads, and a steering op needs a write tap to land in, so this row priced capture under replay
    # and left the other half of the feature unmeasured -- with a message that blamed graph replay for
    # it. Auto covers writes now, so the cell stands either way, and naming them here has become a
    # *narrowing*: one write buffer at the site the workload steers rather than one per layer, which
    # is what keeps this row's memory comparable with the columns beside it. The value is the sentinel
    # `run_bench.STEER_WRITES`, resolved there to the mid-stack `resid_post` the workload steers,
    # because the layer differs per model and a static write has to be named before the model exists.
    VariantSpec(
        "vllm-static",
        "vllm-static",
        {"static_points": "auto", "static_writes": "steer"},
        "breakable CUDA graphs with resid_post static wraps at every layer, and a write tap mid-stack",
        "interp-engine vllm static",
        models=("gemma-2-2b", "qwen3-4b", "llama-3.1-8b", "qwen3.8-27b", "deepseek-v4-flash-0731"),
    ),
    # DSpark on, against the `vllm` column with it off -- the pair is the measurement, so read the two
    # together and do not read this one alone. Three things make that comparison less clean than the
    # rest of the table, all of them worth knowing before quoting a speedup:
    #
    # `num_speculative_tokens` is the checkpoint's own `dspark_block_size` (5), which vLLM requires it
    # to be at least; DSpark drafts the whole block in one parallel pass and then adds intra-block
    # dependency with a Markov head, so this is a block size rather than a draft depth.
    #
    # It changes the model runner underneath. `method="dspark"` forces vLLM's V2 runner, and
    # DeepseekV4ForCausalLM is MoE and absent from `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES`, so the
    # `vllm` column serves it on V1 while this one serves it on V2. The delta is therefore
    # speculation *plus* a runner, not speculation alone. It does at least exercise both of
    # `vllm_capture._demux`'s metadata seams on one checkpoint, which nothing else here does.
    #
    # The capture workloads run rather than being skipped, as they do on `vllm-cudagraph` and for the
    # same reason: what a capture returns under speculative decoding is the interesting part. Expect
    # `capture_gen` and `steer` to be *wrong* rather than absent -- accumulate=True appends a row per
    # forward, and a verification forward for a rejected draft is a forward, so rows stop corresponding
    # to accepted tokens. Prefill-only captures (`capture_mid`, `capture_all`) are unaffected.
    VariantSpec(
        "vllm-dspark",
        "vllm",
        {
            "extra_vllm_kwargs": {
                "speculative_config": {"method": "dspark", "num_speculative_tokens": 5},
            },
        },
        "DSpark speculative decoding ON; hooked backend, so still capture-capable",
        "interp-engine vllm +DSpark",
        models=("deepseek-v4-flash-0731",),
    ),
    # The same speculation with vLLM's graphs left on, which is the fourth cell of a 2x2 and the one
    # that says which of the two knobs the `vllm-dspark` row is measuring. It exists because that row
    # came back *slower* than plain `vllm` on this checkpoint, and "speculation costs 36% here" and
    # "speculation cannot pay for itself without graphs" are different claims with different remedies:
    # the first is a reason not to ship DSpark, the second a reason not to ship it *with capture*.
    #
    # Speculation multiplies host-side work per step -- a draft pass plus a verify pass, both dispatched
    # from Python -- and amortizing exactly that is what a captured graph does. So this is where a
    # speedup should appear if there is one, and it is also the configuration a serving deployment would
    # actually run. Not capture-capable, for the same reason `vllm-cudagraph` is not: a replayed graph
    # never calls the Python forward a hook is attached to.
    VariantSpec(
        "vllm-dspark-cudagraph",
        "vllm-generate",
        {
            "extra_vllm_kwargs": {
                "speculative_config": {"method": "dspark", "num_speculative_tokens": 5},
            },
        },
        "DSpark ON with vLLM's own CUDA graphs, i.e. how it would be served",
        "vllm (vanilla) +DSpark",
        models=("deepseek-v4-flash-0731",),
    ),
)


# ------------------------------------------------------------------------ workloads --


@dataclass(frozen=True)
class WorkloadSpec:
    """One timed operation, and how many times to run it.

    ``repeats`` is the number of *measured* runs; one unmeasured warmup always precedes them, so
    allocator growth and any lazy import land outside the measurement. The reported figure is the
    median, which is more honest than a mean on a machine that may have a compositor on it.
    """

    key: str
    prompt_tokens: int
    max_new_tokens: int = 0
    concurrency: int = 1
    repeats: int = 3
    summary: str = ""


#: Prompt lengths are token counts, not characters, so every model does the same amount of work
#: regardless of how its tokenizer splits the text (see :func:`build_prompt`).
#:
#: The set is chosen so that differences are attributable. ``capture_mid`` and ``capture_all`` share
#: a prompt length and differ only in how many points come back, which isolates capture transport
#: from the forward. ``capture_gen`` and ``steer`` are identical except for the steering spec, which
#: isolates the cost of steering. ``generate`` and ``generate_x8`` differ only in concurrency, which
#: is where the batching difference between the backends shows up: vLLM batches, eager serves one
#: at a time.
#:
#: ``lens_topk`` is the read-out as serving actually performs it, top-k reduced before anything
#: crosses a process boundary. The unreduced ``decode_residuals`` is deliberately not benchmarked:
#: it ships ``[rows, vocab]`` out of the vLLM worker -- half a gigabyte per call at this row count --
#: and no caller of ours takes that path.
WORKLOADS: tuple[WorkloadSpec, ...] = (
    WorkloadSpec("generate", 512, 128, 1, 3, "512-token prompt, 128 new tokens, greedy, one at a time"),
    WorkloadSpec("generate_x8", 512, 128, 8, 2, "the same request issued 8x concurrently"),
    WorkloadSpec("capture_mid", 512, 0, 1, 3, "resid_post at the middle layer over a 512-token prompt"),
    WorkloadSpec("capture_all", 512, 0, 1, 3, "resid_post at every layer over the same prompt"),
    WorkloadSpec("capture_gen", 128, 32, 1, 3, "generate 32 tokens capturing resid_post at prompt+generated"),
    WorkloadSpec("steer", 128, 32, 1, 3, "capture_gen again, with an add-steering vector at the middle layer"),
    WorkloadSpec("lens_topk", 512, 0, 1, 3, "512 rows of d_model read out to top-10 ids, as lens serving does"),
)

#: How many candidates ``lens_topk`` asks for. Ten is the order the lens UI displays; the exact value
#: barely moves the timing (the cost is the unembed matmul and the transport, not the ``topk``), but it
#: is fixed here so the payload column means one thing.
LENS_TOP_N = 10

#: vLLM refuses to boot unless its KV pool can hold one request at the model's advertised context,
#: and these models advertise 32k-131k. Capping keeps that precondition small and equal across
#: models, so the memory column compares backends rather than context advertisements. It must exceed
#: the longest ``prompt_tokens + max_new_tokens`` above.
MAX_MODEL_LEN = 2048

#: Below vLLM's own 0.9 default, and not a tuning choice -- 0.9 cannot run ``lens_topk`` at all.
#: ``worker_lens_readout`` computes the full ``[rows, vocab]`` logits and then takes a ``logsumexp``
#: over them, whose intermediate is another tensor of that size, so the worker needs roughly twice the
#: vocab-logits in scratch: ~0.5 GiB for a 256k-vocab model at 512 rows. At 0.9 of a 32 GiB card with
#: a desktop session on it there is none left, and the read-out dies with a worker-side CUDA OOM. That
#: is worth knowing about (the fast lens path is the *recommended* one), but it is a property of the
#: engine rather than of the backend comparison, so the sweep leaves headroom and reports it.
#:
#: Uniform across models rather than tuned per model, which is what keeps the memory column comparable.
#: Override per run with ``--gpu-memory-utilization``.
GPU_MEMORY_UTILIZATION = 0.8


def workload(key: str) -> WorkloadSpec:
    for w in WORKLOADS:
        if w.key == key:
            return w
    raise KeyError(f"unknown workload {key!r}; known: {', '.join(w.key for w in WORKLOADS)}")


def model(key: str) -> ModelSpec:
    for m in MODELS:
        if m.key == key:
            return m
    raise KeyError(f"unknown model {key!r}; known: {', '.join(m.key for m in MODELS)}")


def variant(key: str) -> VariantSpec:
    for v in VARIANTS:
        if v.key == key:
            return v
    raise KeyError(f"unknown variant {key!r}; known: {', '.join(v.key for v in VARIANTS)}")


def variant_label(key: str) -> str:
    """The report's name for a variant, falling back to the key so an unknown one still renders."""
    for v in VARIANTS:
        if v.key == key:
            return v.label or v.key
    return key


def default_models(gpu_gib: float | None) -> tuple[ModelSpec, ...]:
    """The models a plain sweep should run on a card of this size, in spec order.

    A row whose weights cannot fit is dropped rather than attempted, because on the largest of them
    the attempt costs a bring-up and several minutes before vLLM reports it, and a sweep that exits
    non-zero for a model the card was never going to hold reads as a broken benchmark rather than as
    a small card. The caller is expected to say which rows it dropped -- silence here would be the
    worse failure, since the report renders whatever cells exist and a missing row looks the same as
    one nobody has run yet.

    ``None`` means the card could not be read (no ``nvidia-smi``), and then nothing is dropped: an
    unknown card is not a small one, and refusing to sweep on a box we simply failed to interrogate
    would be a guess in the expensive direction.
    """
    if gpu_gib is None:
        return MODELS
    return tuple(m for m in MODELS if m.min_gpu_gib <= gpu_gib)


def gpu_memory_utilization_for(model_key: str, override: float | None = None) -> float:
    """vLLM's fraction of the card for this row: an explicit ``--gpu-memory-utilization`` first, then
    the model's own declared exception, then the uniform default."""
    if override is not None:
        return override
    try:
        declared = model(model_key).gpu_memory_utilization
    except KeyError:  # --hf-id benchmarks checkpoints the spec has never heard of
        declared = None
    return declared if declared is not None else GPU_MEMORY_UTILIZATION


def variant_applies(variant_key: str, model_key: str) -> bool:
    """Whether this ``(variant, model)`` pair is one the sweep should attempt.

    An unknown *variant* raises, because that is a typo in a filter and running the wrong column is
    worse than stopping. An unknown *model* does not: ``--hf-id`` benchmarks checkpoints the spec has
    never heard of, and an unrestricted variant applies to those too.
    """
    return not (restricted := variant(variant_key).models) or model_key in restricted


# --------------------------------------------------------------------------- prompt --

#: Public-domain filler (Moby-Dick's opening). Content is irrelevant to speed; what matters is that
#: it is the same text for every model and long enough to reach the longest prompt above after
#: tokenization, so no model is measured on a padded or truncated-short prompt.
_FILLER = (
    "Call me Ishmael. Some years ago, never mind how long precisely, having little or no money in my "
    "purse, and nothing particular to interest me on shore, I thought I would sail about a little and "
    "see the watery part of the world. It is a way I have of driving off the spleen and regulating the "
    "circulation. Whenever I find myself growing grim about the mouth, whenever it is a damp, drizzly "
    "November in my soul, whenever I find myself involuntarily pausing before coffin warehouses, and "
    "bringing up the rear of every funeral I meet, and especially whenever my hypos get such an upper "
    "hand of me that it requires a strong moral principle to prevent me from deliberately stepping "
    "into the street, and methodically knocking people's hats off, then I account it high time to get "
    "to sea as soon as I can. "
)


def build_prompt(tokenizer: object, n_tokens: int) -> list[int]:
    """Return exactly ``n_tokens`` token ids for this tokenizer.

    Normalizing on token count rather than on text is what makes the models comparable: the same
    passage is 20% more tokens under one tokenizer than another, and prompt length is the thing the
    prefill cost is proportional to.

    The filler is repeated until it is long enough and then truncated, so the tail is mid-sentence.
    That is fine for a speed measurement and deliberate: padding to length with a pad token would
    measure attention over padding instead.
    """
    encode = tokenizer.encode  # pyright: ignore[reportAttributeAccessIssue]
    text = _FILLER
    ids: list[int] = encode(text, add_special_tokens=True)
    # Grow geometrically rather than by one copy at a time: tokenizing a long string is not free and
    # the longest prompt here needs ~4 copies of the filler.
    while len(ids) < n_tokens:
        text = text * 2
        ids = encode(text, add_special_tokens=True)
    return ids[:n_tokens]
