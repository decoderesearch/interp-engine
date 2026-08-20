# Internals

What each file in `interp_engine/` owns, and what the engine checks about itself. Read this before
editing the engine; [AGENTS.md](../AGENTS.md) is the companion piece on the boundaries between those
files and why they are where they are.

- [Modules](#modules)
- [Correctness](#correctness)

## Modules

- `model.py` — `EagerModel`: wraps `AutoModelForCausalLM` (eager, `no_processing` semantics),
  holds the tokenizer + config-derived dims, canonical hook-point resolution, and an optional
  `quantization_config` passthrough to `from_pretrained`.
- `facts.py` — the single source of truth for model facts, shared by both backends: structural
  attribute-name vocabularies, config-derived dims, per-layer window/linear-attention predicates,
  and the per-backend tables (fused-QKV layout, parallel-block architectures). Config arithmetic
  and string tables only — no torch, no live model — so the vLLM client can answer dims for a model
  it never builds.
- `arch.py` — the **eager adapter**: binds a live HF module tree to the structural roles in
  `facts.py`, plus the machine-readable known-quirks table (attention sinks, softcapping, hybrid
  attention, ...).
- `protocol.py` — `InterpModel`: the surface both backends implement, and the contract a sync free
  function dispatches against. Adding a method here means adding it to both backends and to
  `sync.py`; `tests/test_sync_parity.py` fails on a missing twin.
- `sync.py` / `_loop.py` — `sync_model(model)`: the protocol without an event loop, one explicit
  wrapper per method over a background loop thread that is created lazily, reused per model, and
  refuses rather than deadlocks when called from inside a running loop.
- `dispatch.py` — the shared plumbing every free function's two arms sit on: token coercion
  (`TokensLike`, batch refusals) and `CAPABILITIES`, the table each `CapabilityUnsupported` message
  is built from.
- `points.py` — the canonical point registry: every point's scope, width and vLLM support, and the
  engine's own sentence for each refusal. [SUPPORTED_POINTS.md](SUPPORTED_POINTS.md) is that table
  as prose, and is checked against this module by `tests/test_points_registry.py`.
- `hooks.py` — the low-level read/write forward-hook substrate.
- `capture.py` — capture context manager returning a cache keyed by canonical names
  (`resid_post`, `resid_mid`, `mlp_in`, `mlp_act`, `attn_probs`, `value`, `router_logits`,
  `embeddings`, ...), plus the post-processing a captured tensor needs to be usable (fused-QKV
  splits, the attention gate, a norm's scale and gain, per-head residual contributions, dense expert
  assignments).
- `attn_scores.py` — the pre-softmax attention scores, which no module boundary carries: it registers
  a wrapping attention implementation for the duration of a capture and delegates to the
  checkpoint's own eager function, so the forward is unchanged.
- `tokenize.py` — `to_tokens`/`to_str_tokens`/`to_string` (TransformerLens-parity), chat
  templating, and per-token span metadata (the single source of truth for message boundaries).
- `chat_conventions.py` — the only per-model chat table: harmony markers, reasoning delimiters,
  turn-end tokens. Selected by tokenizer capability, never by model name (see
  [Where model-specific config lives](ARCHITECTURE_QUIRKS.md#where-model-specific-config-lives)).
- `chat_compose.py` — rebuilds assistant messages from a generation (`compose_assistant_turns`),
  reading the generated text only; callers pair it with the prompt messages they already have.
- `lens.py` — logit + Jacobian lens by calling the real `final_norm` + `lm_head`. Returns **raw**
  logits unless the caller passes a `softcap`. The vLLM path never returns raw logits (see [vLLM
  `compute_logits` is not a bare unembed](ARCHITECTURE_QUIRKS.md#vllm-compute_logits-is-not-a-bare-unembed)).
- `steer.py` — steering, and the per-token generation stream. Each method's arithmetic is one
  `steer_delta` branch, which is what the vLLM worker's modifier computes too and what
  `tests/test_steer_math_parity.py` runs against it on CPU; `steer()` is the context both backends
  take, registering per-request on vLLM rather than installing a global hook.
- `mappers.py` — translation between canonical points and other frameworks' names:
  TransformerLens hook strings and nnsight/nnterp accessors, both directions. See [Porting from
  TransformerLens, nnsight or nnterp](PORTING.md#porting-from-transformerlens-nnsight-or-nnterp).
- `autograd_support.py` — the `GradSupport` verdict: whether a model can give you gradients, and
  which specific thing is blocking it. Pure config arithmetic, so it is safe to call before
  `warmup()`. See [Gradients](GRADIENTS.md#gradients).
- `cuda_preflight.py` — `check_cuda_driver`: compares the host CUDA driver against the CUDA
  version torch was built for and raises with the forward-compat fix (`cuda-compat-<major>-<minor>`
  - `LD_LIBRARY_PATH`) before the first CUDA call, instead of failing ten frames deep in
    `torch.cuda._lazy_init`. Lives here because every app on the engine inherits the same CUDA
    floor — the `[vllm]` wheels link `libcudart.so.13` directly.

## Correctness

The engine's job is to hand back the tensor a module actually produced, so most of what can go wrong
is quiet: a point resolves to a plausible neighbour, the shapes agree, and the numbers are wrong.
The test suite is built around checks that a shape-correct guess cannot pass. The cross-engine half
of the story — the same points scored against TransformerLens, nnsight/nnterp, vLLM and SGLang on
50+ real checkpoints — is [`validator/`](../validator/README.md).

**Golden parity.** `tests/test_parity_gpt2.py` pins every capture point on gpt2 against
TransformerLens, from a committed golden file. It is the one place another framework is loaded, and
CI treats a skip as a failure (`IE_REQUIRE_PARITY=1`) so a missing dependency or a cold cache cannot
quietly retire the gate.

**Invariants over attribute names.** Three identities hold on any model of a family, so they catch a
misresolved point without needing a reference implementation: `probs @ value == z` for the per-head
value and DFA (`tests/test_qkv_layout.py`, which also asserts the _wrong_ layout fails — otherwise
the test would pass on a single-head model), `resid_pre + attn_out_post + mlp_out_post == resid_post`
for sandwich norms and residual multipliers, and `down_proj(mlp_act) == mlp_out` for the neuron
basis. Where a point genuinely does not exist — a Mamba block's attention, a latent-attention model's
`value`, the residual between the sublayers of a parallel block — it is refused with an explanation
rather than returned as a plausible tensor.

**Self-consistency on real weights.** `tests/test_new_models_gpu.py` decodes the last layer's
residual through the real `final_norm` + `lm_head` and requires the model's true next-token argmax
back, which validates the whole arch map end-to-end without a second framework.
`tests/test_sliding_window_attn.py` pins the vLLM attention recompute's band and sink terms, and
`tests/test_vllm_only_families.py` checks the spellings for families `transformers` has no class for
against a synthetic tree in the shape their own modeling file describes.

**The two backends against each other.** `tests/test_vllm_capture_gpu.py` runs a real vLLM engine and
requires each captured point to match the eager backend's — including that vLLM's positional layer
index names the layer HF's does, which nothing checked before and which would fail silently rather
than raise. It needs `interp-engine[vllm]`, so it self-skips elsewhere; note that running it via
`.venv-vllm/bin/python` needs that directory on `PATH` too, because vLLM shells out to `ninja` to
build a sampler kernel at startup. `tests/test_vllm_wire_grammar.py` covers the same process
boundary on CPU, over a synthetic demux.

**The docs are parsed, not maintained.** The point table in
[SUPPORTED_POINTS.md](SUPPORTED_POINTS.md) and the footnote markers in
[ENGINE_HOOK_MAPPINGS.md](ENGINE_HOOK_MAPPINGS.md) are read back and compared against `points.py` by
`tests/test_points_registry.py`, and every `interp_engine` name in a doc code fence has to still
import and still be in `__all__` (`tests/test_doc_code_fences.py`). A point that quietly became
servable while the tick stayed put is a claim the engine does not make.

**How CI is split.** Two jobs, both on every non-Markdown change: a CPU job
(`-m "not gpu and not xl"`) that owns the golden gate, the lint/format/type gates and the small
models eagerly, and a managed-L4 GPU job (`-m "gpu and not xl"`) running the same models on
CUDA/bf16. The `xl` models are tens of GB and run nowhere automatically — `pytest -m xl` on a big
box. Locally the GPU tests self-skip without CUDA and model loads skip when weights aren't cached, so
a plain `pytest tests` on a laptop runs the fast suite.

[← back to the interp-engine README](../README.md)
